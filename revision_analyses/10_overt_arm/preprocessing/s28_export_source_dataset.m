%% ========================================================================
%  s28_export_source_dataset.m -- assemble the overt source dataset
%
%  Builds the overt counterpart of the covert source dataset, one
%  Subject##_sLORETA_raw.mat per subject with the same variables, the same
%  ROI order and the same epoch window, so that every downstream analysis
%  reads it without modification.
%
%  The projection recipe is the one used for the covert dataset: each
%  cleaned 56-channel epoch is multiplied by the subject's sLORETA imaging
%  kernel and averaged within the 148 Destrieux scouts over samples
%  1:500 (-0.5 to +1.496 s at 250 Hz).
%
%  Exactly one input differs from the covert dataset, namely the EEG:
%      covert  cfg.eeg/S00##_..._rejectchan56_u1.set
%      overt   <cfg.out>/overt_clean/S00##_..._rejectchan56_o1.set
%  The imaging kernel is the same one that produced the covert source data.
%  The covert and overt Brainstorm studies hold bit-identical kernels (same
%  head model, same 56 channels, GoodChannel 1:56 in both); the covert study
%  is preferred and the overt study is used as a fallback.
%
%  GoodChannel indexes the channel list the kernel was built for. If the
%  overt channel list ever differed in content or order, the multiplication
%  would silently use the wrong rows and produce plausible but wrong ROI
%  time series, so the overt channel labels are compared against the covert
%  ones for every subject.
%
%  The atlas is selected by name in tess_cortex_pial_low.mat and the scout
%  count is asserted, because the order of the Atlas array varies between
%  subjects.
%
%  KEEP_U1_ONLY keeps only the first utterance of each trial, matching the
%  covert dataset. The overt epoching in s24 already expands u_1 alone, so
%  this acts as a safety net rather than a filter.
%
%  ENVIRONMENT (optional)
%    SUBJ_SUBSET   space-separated Brainstorm subject names, e.g.
%                  "Subject09 Subject40", for a smoke test
%
%  INPUT   <cfg.out>/overt_clean/*rejectchan56_o1.set   (from s26)
%          cfg.eeg                                      covert channel lists
%          cfg.source                                   subject roster
%          cfg.bst_protocol/{anat,data}                 imaging kernels
%  OUTPUT  <cfg.out>/overt_source_dataset/Subject##_sLORETA_raw.mat
%          <cfg.out>/overt_source_dataset/GENERATION_LOG.txt
%% ========================================================================

clearvars; clc;

addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'config'));
cfg = set_paths();

%% ---------------- configuration ----------------
bsD  = fullfile(cfg.bst_protocol, 'data', filesep);
bsA  = fullfile(cfg.bst_protocol, 'anat', filesep);
setO = fullfile(cfg.out, 'overt_clean', filesep);
setC = fullfile(cfg.eeg, filesep);                     % covert, for the channel check
pubD = fullfile(cfg.source, filesep);                  % covert source data, for the roster
outD = fullfile(cfg.out, 'overt_source_dataset', filesep);

trial_start  = 1;            % same window as covert: samples 1:500
trial_end    = 500;          % = -0.5 .. +1.496 s at 250 Hz
freq_bands   = {'raw'};
kernel_types = {'sLORETA'};
atlas_name   = 'Destrieux';
n_roi_expect = 148;
KEEP_U1_ONLY = true;         % false -> keep u_1..u_5
SUBJ_SUBSET  = {};
if ~isempty(getenv('SUBJ_SUBSET'))
    SUBJ_SUBSET = strsplit(strtrim(getenv('SUBJ_SUBSET')));
end

%% ---------------- subject list ----------------
% Taken from the covert source dataset, so that the two arms stay one to one.
pf = dir([pubD 'Subject*_sLORETA_raw.mat']);
assert(~isempty(pf), 'no covert source files in %s', pubD);
subjects = cellfun(@(x) x(1:strfind(x, '_sLORETA')-1), {pf.name}, 'UniformOutput', false);
if ~isempty(SUBJ_SUBSET), subjects = intersect(subjects, SUBJ_SUBSET, 'stable'); end
fprintf('subjects: %d (%s .. %s)\n', numel(subjects), subjects{1}, subjects{end});

if ~isfolder(outD), mkdir(outD); end
addpath(cfg.eeglab); eeglab nogui;

if isempty(gcp('nocreate')), parpool('Processes', cfg.n_workers); end

%% ---------------- generate ----------------
n = numel(subjects);
report = cell(n, 1);
parfor ii = 1:n
    subj = subjects{ii};
    sid  = ['S00' subj(8:9)];
    t0   = tic;

    % ---- Destrieux scouts, selected by name ----
    anatData = load([bsA subj '/tess_cortex_pial_low.mat'], 'Atlas');
    di = find(strcmpi({anatData.Atlas.Name}, atlas_name));
    assert(numel(di) == 1, '%s: %s atlas not unique', subj, atlas_name);
    dk = anatData.Atlas(di);
    assert(numel(dk.Scouts) == n_roi_expect, '%s: %d scouts, expected %d', ...
        subj, numel(dk.Scouts), n_roi_expect);
    roiindex = cell(length(dk.Scouts), 2);
    for i = 1:length(dk.Scouts)
        roiindex{i,1} = dk.Scouts(i).Vertices;
        roiindex{i,2} = dk.Scouts(i).Label;
    end

    % ---- Brainstorm study folder holding the kernel ----
    % The covert study is preferred: its kernel is the matrix behind the
    % covert source data. The overt study holds a bit-identical kernel and
    % is used only when the covert study is absent.
    address = [bsD subj filesep sid ...
        '_Filters_processed_trials_precut_ICA_ICACUT_rejectchan_u1_only' filesep];
    if ~isfolder(address)
        address = [bsD subj filesep sid ...
            '_Filters_processed_trials_precut_ICA_ICACUT_rejectchan_o' filesep];
    end
    assert(isfolder(address), '%s: no kernel study folder', subj);

    for f_idx = 1:length(freq_bands)
        freq_band_current = freq_bands{f_idx};
        eeg_file = [setO sid ...
            '_Filters_overt_processed_trials_precut_ICA_ICACUT_rejectchan56_o1.set'];
        EEG = pop_loadset(eeg_file);
        eeg_data   = EEG.data;
        eeg_events = EEG.event;

        assert(EEG.pnts >= trial_end, '%s: pnts=%d < %d', subj, EEG.pnts, trial_end);
        assert(numel(eeg_events) == EEG.trials, '%s: %d events vs %d epochs', ...
            subj, numel(eeg_events), EEG.trials);

        % The kernel rows follow the covert channel list; verify per subject.
        cov_file = [setC sid '_Filters_processed_trials_precut_ICA_ICACUT_rejectchan56_u1.set'];
        Cc = load(cov_file, '-mat', 'chanlocs');
        assert(isequal({Cc.chanlocs.labels}, {EEG.chanlocs.labels}), ...
            ['%s: the overt channel list differs from the covert one, so the ' ...
             'kernel rows would not line up'], subj);

        sel = 1:numel(eeg_events);
        if KEEP_U1_ONLY
            sel = find(~cellfun(@isempty, ...
                regexp({eeg_events.type}, '_u_1_b_', 'once')));
        end

        for k_idx = 1:length(kernel_types)
            kernel_name_current = kernel_types{k_idx};
            save_filename = [outD subj '_' kernel_name_current '_' ...
                freq_band_current '.mat'];

            loretaFile = dir([address 'results_' kernel_name_current '*']);
            assert(~isempty(loretaFile), '%s: no %s kernel in the study folder', ...
                subj, kernel_name_current);
            loretaData = load([loretaFile(1).folder '/' loretaFile(1).name], ...
                'ImagingKernel', 'GoodChannel');

            condition_data      = cell(numel(sel), 1);
            condition_data_type = cell(numel(sel), 1);
            idx = 0;
            for t = reshape(sel, 1, [])
                event = eeg_events(t);
                trial_data_segment = eeg_data(:, trial_start:trial_end, t);
                source_data = loretaData.ImagingKernel * ...
                    trial_data_segment(loretaData.GoodChannel, :);
                roi_means = zeros(size(roiindex, 1), size(source_data, 2));
                for roi = 1:size(roiindex, 1)
                    roi_vertices = roiindex{roi,1};
                    roi_means(roi,:) = mean(source_data(roi_vertices, :), 1);
                end
                idx = idx + 1;
                condition_data{idx}      = roi_means;
                condition_data_type{idx} = event.type;
            end
            condition_data      = condition_data(1:idx);
            condition_data_type = condition_data_type(1:idx);

            kernel_name = kernel_name_current;
            freq_band   = freq_band_current;
            data_struct = struct();
            data_struct.condition_data      = condition_data;
            data_struct.condition_data_type = condition_data_type;
            data_struct.roiindex            = roiindex;
            data_struct.kernel_name         = kernel_name;
            data_struct.freq_band           = freq_band;
            save(save_filename, '-fromstruct', data_struct, '-v7.3');

            report{ii} = sprintf(['%-10s %-9s nEvent=%-4d kept=%-4d ' ...
                'nbchan=%-3d pnts=%-4d nGood=%-3d kernel=%s  %.0fs'], ...
                subj, kernel_name_current, numel(eeg_events), idx, ...
                EEG.nbchan, EEG.pnts, numel(loretaData.GoodChannel), ...
                loretaFile(1).name, toc(t0));
            fprintf('%s\n', report{ii});
        end
    end
end

%% ---------------- provenance log ----------------
logf = [outD 'GENERATION_LOG.txt'];
fid = fopen(logf, 'w');
fprintf(fid, 'overt source dataset\n');
fprintf(fid, 'generated by 10_overt_arm/preprocessing/s28_export_source_dataset.m\n');
fprintf(fid, 'recipe   : identical projection and scout averaging as the covert dataset\n');
fprintf(fid, 'eeg      : %sS00##_Filters_overt_processed_trials_precut_ICA_ICACUT_rejectchan56_o1.set\n', setO);
fprintf(fid, '           (overt arm rebuilt from raw, own ICA, manual ocular layer\n');
fprintf(fid, '            confirmed by a trained rater)\n');
fprintf(fid, 'kernel   : %sSubject##/S00##_..._rejectchan_u1_only/results_sLORETA*\n', bsD);
fprintf(fid, '           the same ImagingKernel as the covert source data\n');
fprintf(fid, 'checked  : per subject, overt channel labels == covert channel labels, same order\n');
fprintf(fid, 'atlas    : %s (%d ROIs), selected by name from tess_cortex_pial_low.mat\n', atlas_name, n_roi_expect);
fprintf(fid, 'window   : samples %d:%d  (-0.5 .. +1.496 s at 250 Hz)\n', trial_start, trial_end);
fprintf(fid, 'u1 only  : %d\n', KEEP_U1_ONLY);
fprintf(fid, 'subjects : %d\n\n', n);
for ii = 1:n
    if ~isempty(report{ii}), fprintf(fid, '%s\n', report{ii}); end
end
fclose(fid);
fprintf('\nwrote %s\nDONE\n', logf);
