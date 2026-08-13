%% =========================================================================
%  s04_source_addback_destrieux.m
%
%  Source-localise the component add-back preprocessing variants onto the
%  Destrieux atlas (148 cortical ROIs), i.e. repeat the source reconstruction
%  of the main analysis on datasets in which one class of artefact components
%  has been reinstated before source projection.
%
%  INPUT
%    cfg.eeg           cleaned, epoched EEG of the requested variant
%                      (S####_..._rejectchan56_u1<suffix>.set; 56 channels,
%                      average reference, 500 samples per trial)
%    cfg.bst_protocol  Brainstorm protocol root holding, per participant, the
%                      anatomy (anat/) and the individual sLORETA imaging
%                      kernel (data/)
%    cfg.source        published source directory, read only to obtain the
%                      participant list
%
%  OUTPUT
%    One .mat file per participant, in the layout of the published source data:
%      condition_data       cell array, one 148 x 500 (ROI x sample) matrix per trial
%      condition_data_type  trial labels
%      roiindex             {vertex indices, ROI label} per ROI
%      kernel_name, freq_band, variant, atlas_name, source_set, kernel_file
%    Destination: cfg.addback(variant) for the add-back variants, and
%    cfg.out/verify_standard_dest for the control variant.
%
%  METHOD
%    For every trial the sLORETA imaging kernel is applied to the good channels
%    of the 500-sample epoch, and the resulting source time courses are
%    averaged over the vertices of each Destrieux scout. Selection is by name
%    throughout: the atlas is selected by name and asserted to contain 148
%    scouts, participants are selected by name from the published source
%    directory, and the Brainstorm study folder is matched by its
%    '_rejectchan_u1_only' suffix and asserted to hold exactly one sLORETA
%    kernel.
%
%  VARIANTS  (they differ only in which cleaned .set file is read)
%    standard : *_u1.set          brain components only; control run, which
%                                 must reproduce the published source data
%    lateye   : *_u1_lateye.set   brain + lateral eye components
%    mus      : *_u1_mus.set      brain + muscle components
%    blink    : *_u1_blink.set    brain + blink components
%
%  ENV
%    AB_VARIANT   standard | lateye | mus | blink   (default lateye)
%    AB_SUBJ      1-based index into the participant list, for single runs
%    AB_OUTDIR    absolute path overriding the output directory
%    AB_WORKERS   parallel pool size (default cfg.n_workers)
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

% EEGLAB supplies pop_loadset. eeglab('nogui') declares its dataset variables
% in the workspace; they are cleared again so that the parfor loop below sees a
% plain local EEG variable.
if exist('pop_loadset', 'file') ~= 2
    addpath(cfg.eeglab);
    eeglab('nogui');
end
clear ALLEEG EEG CURRENTSET ALLCOM LASTCOM STUDY CURRENTSTUDY;

variant = getenv('AB_VARIANT');
if isempty(variant), variant = 'lateye'; end
suffixes = struct('standard', '', 'lateye', '_lateye', 'mus', '_mus', 'blink', '_blink');
assert(isfield(suffixes, variant), ...
    'AB_VARIANT must be standard|lateye|mus|blink (got "%s")', variant);
setSuffix = suffixes.(variant);

STUDY_SUFFIX = '_rejectchan_u1_only';
N_ROI_EXPECT = 148;
trial_start  = 1;
trial_end    = 500;
kernel_name  = 'sLORETA';
freq_band    = 'raw';

bstRoot = cfg.bst_protocol;
eegDir  = cfg.eeg;
assert(isfolder(bstRoot), 'Brainstorm protocol not found: %s', bstRoot);

% ---- participants BY NAME, taken from the published source data ----------
pub = dir(fullfile(cfg.source, 'Subject*_sLORETA_raw.mat'));
subjects = cellfun(@(s) s(1:strfind(s, '_sLORETA') - 1), {pub.name}, 'UniformOutput', false);
assert(numel(subjects) == 57, 'expected 57 published participants, found %d', numel(subjects));

outDir = getenv('AB_OUTDIR');
if isempty(outDir)
    if strcmp(variant, 'standard')
        outDir = fullfile(cfg.out, 'verify_standard_dest');   % control run
    else
        outDir = cfg.addback(variant);
    end
end
if ~isfolder(outDir), mkdir(outDir); end

onlySubj = str2double(getenv('AB_SUBJ'));
if isnan(onlySubj), subjRange = 1:numel(subjects); else, subjRange = onlySubj; end

nw = str2double(getenv('AB_WORKERS'));
if isnan(nw), nw = cfg.n_workers; end
if nw > 1 && isempty(gcp('nocreate')), parpool('Processes', nw); end

fprintf('==== add-back source localisation ====\n');
fprintf('variant      : %s   (.set suffix "%s")\n', variant, setSuffix);
fprintf('protocol     : %s\n', bstRoot);
fprintf('output       : %s\n', outDir);
fprintf('participants : %d of %d\n\n', numel(subjRange), numel(subjects));

parfor ii = subjRange
    subj = subjects{ii};
    tok = regexp(subj, '(\d+)', 'tokens', 'once');
    sid = sprintf('S%04d', str2double(tok{1}));

    % ---- atlas BY NAME ---------------------------------------------------
    anatFile = fullfile(bstRoot, 'anat', subj, 'tess_cortex_pial_low.mat');
    assert(isfile(anatFile), '%s: missing anatomy %s', subj, anatFile);
    anatData = load(anatFile, 'Atlas');
    names = {anatData.Atlas.Name};
    ai = find(strcmp(names, 'Destrieux'));
    assert(isscalar(ai), '%s: Destrieux not found (have: %s)', subj, strjoin(names, ', '));
    atl = anatData.Atlas(ai);
    assert(numel(atl.Scouts) == N_ROI_EXPECT, ...
        '%s: Destrieux has %d scouts, expected %d', subj, numel(atl.Scouts), N_ROI_EXPECT);

    roiindex = cell(numel(atl.Scouts), 2);
    for i = 1:numel(atl.Scouts)
        roiindex{i,1} = atl.Scouts(i).Vertices;
        roiindex{i,2} = atl.Scouts(i).Label;
    end

    % ---- study folder BY NAME -------------------------------------------
    sd = dir(fullfile(bstRoot, 'data', subj));
    sd = sd([sd.isdir]);
    hit = find(endsWith({sd.name}, STUDY_SUFFIX));
    assert(isscalar(hit), '%s: %d study folders end with %s', subj, numel(hit), STUDY_SUFFIX);
    address = fullfile(bstRoot, 'data', subj, sd(hit).name);

    kf = dir(fullfile(address, ['results_' kernel_name '*']));
    assert(isscalar(kf), '%s: %d sLORETA kernels in %s', subj, numel(kf), sd(hit).name);
    kernelFile = fullfile(kf.folder, kf.name);
    loretaData = load(kernelFile, 'ImagingKernel', 'GoodChannel');

    % ---- EEG -------------------------------------------------------------
    eeg_file = fullfile(eegDir, [sid '_Filters_processed_trials_precut_ICA_ICACUT_' ...
                                 'rejectchan56_u1' setSuffix '.set']);
    assert(isfile(eeg_file), '%s: missing input %s', subj, eeg_file);
    EEG = pop_loadset(eeg_file);
    eeg_data = EEG.data;
    eeg_events = EEG.event;

    % ---- kernel application and ROI averaging ----------------------------
    condition_data = cell(100,1);
    condition_data_type = cell(100,1);
    idx = 0;
    for trial_idx = 1:length(eeg_events)
        event = eeg_events(trial_idx);
        trial_data_segment = eeg_data(:, trial_start:trial_end, trial_idx);
        source_data = loretaData.ImagingKernel * trial_data_segment(loretaData.GoodChannel,:);
        roi_means = zeros(size(roiindex, 1), size(source_data, 2));
        for roi = 1:size(roiindex, 1)
            roi_means(roi, :) = mean(source_data(roiindex{roi, 1}, :), 1);
        end
        idx = idx + 1;
        condition_data{idx} = roi_means;
        condition_data_type{idx} = event.type;
    end
    condition_data = condition_data(1:idx);
    condition_data_type = condition_data_type(1:idx);

    s = struct();
    s.condition_data = condition_data;
    s.condition_data_type = condition_data_type;
    s.roiindex = roiindex;
    s.kernel_name = kernel_name;
    s.freq_band = freq_band;
    s.variant = variant;
    s.atlas_name = atl.Name;
    s.source_set = eeg_file;
    s.kernel_file = kernelFile;

    parsave(fullfile(outDir, [subj '_' kernel_name '_' freq_band '.mat']), s);
    fprintf('%-12s %-9s %3d trials  %d ROIs  <- %s\n', subj, variant, idx, size(roiindex,1), sd(hit).name);
end

fprintf('\nDONE  -> %s\n', outDir);

function parsave(fname, s)
save(fname, '-fromstruct', s, '-v7.3');
end
