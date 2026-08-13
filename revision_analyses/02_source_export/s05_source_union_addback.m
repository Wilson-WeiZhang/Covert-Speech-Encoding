%% =========================================================================
%  s05_source_union_addback.m
%
%  Source-localise the retained (brain) components together with a union
%  ocular component set onto the Destrieux atlas (148 cortical ROIs),
%  producing datasets with the same shape as the published source data
%  (500 samples per trial).
%
%  WHY THE PROJECTION IS RECOMPUTED HERE
%    The cleaned .set files fix which components have been reinstated: one
%    manually labelled blink and one manually labelled lateral component per
%    participant. The union sets built in 01_ocular_component_sets are larger
%    than that and cannot be expressed in those files, so the reconstruction is
%    redone from the full decomposition (*_precut_ICA.set).
%
%  INPUT
%    cfg.eeg                            full ICA decompositions
%    cfg.out/union_component_sets.mat   union component sets (script s01)
%    cfg.bst_protocol                   Brainstorm anatomy and sLORETA kernels
%
%  OUTPUT
%    One .mat file per participant:
%      condition_data       cell array, one 148 x 500 (ROI x sample) matrix per trial
%      condition_data_type  trial labels
%      roiindex             {vertex indices, ROI label} per ROI
%      kernel_name, freq_band, variant, atlas_name, ics_brain, ics_added, source_set
%    Destination: cfg.addback('lateye_union') or cfg.addback('blink_union'),
%    and cfg.out/verify_union_brain_dest for the control set.
%
%  RECONSTRUCTION
%    act   = (icaweights * icasphere) * data
%    recon = icawinv(:,S) * act(S,:)
%    drop  {Fp1,Fp2,Fpz,ECG,TP9,TP10,FT9,FT10} -> 56 channels, THEN average reference
%    src   = ImagingKernel * recon(GoodChannel,:)
%    ROI   = mean over the vertices of each Destrieux scout -> 148 ROIs
%    This is the operation EEGLAB's pop_subcomp performs, written out so that an
%    arbitrary component set can be retained. The order of the last two
%    preprocessing steps matters: re-referencing before the channels are
%    dropped changes the reconstructed signal by several microvolts.
%
%  TWO INDEPENDENT CHECKS
%    1. AB_SET=brain reinstates nothing and must therefore reproduce the
%       published source data (external check, see 03_pipeline_verification).
%    2. X_union = X_brain + X_added, asserted for every participant: each step
%       after the component projection is linear and the two component sets are
%       disjoint, so the identity must hold exactly up to rounding. It fails if
%       the two branches ever disagree on the source file, channel set or
%       reference (internal check, always on).
%
%  ENV
%    AB_SET     brain | blink | lateral   (default brain)
%    AB_SUBJ    1-based index into the participant list, for single runs
%    AB_WORKERS parallel pool size (default cfg.n_workers)
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

setName = getenv('AB_SET');
if isempty(setName), setName = 'brain'; end
assert(ismember(setName, {'brain','blink','lateral'}), ...
    'AB_SET must be brain|blink|lateral (got "%s")', setName);

STUDY_SUFFIX = '_rejectchan_u1_only';
DROP_CHANS = {'Fp1','Fp2','Fpz','ECG','TP9','TP10','FT9','FT10'};
N_CHAN_KEEP = 56;   % channels entering the head model
N_ROI = 148;        % Destrieux scouts
N_SAMP = 500;       % samples per trial (2 s at 250 Hz)
TH = 0.9;           % ICLabel posterior above which a component counts as an artefact

eegDir  = cfg.eeg;
bstRoot = cfg.bst_protocol;
assert(isfolder(bstRoot), 'Brainstorm protocol not found: %s', bstRoot);

switch setName
    case 'brain',   outDir = fullfile(cfg.out, 'verify_union_brain_dest');
    case 'blink',   outDir = cfg.addback('blink_union');
    case 'lateral', outDir = cfg.addback('lateye_union');
end
if ~isfolder(outDir), mkdir(outDir); end

U = load(fullfile(cfg.out, 'union_component_sets.mat'), 'S');  U = U.S;
assert(numel(U) == 57, 'union table has %d rows', numel(U));

% ---- participant list ------------------------------------------------------
% The participant-specific cases below are indexed by position in this list, so
% the ordering is asserted against the published cohort before it is used.
d = dir(fullfile(eegDir, '*_Filters_processed_trials_precut_ICA.set'));
% One participant (S0013) has no T1 MRI and is absent from the source-space cohort.
d(startsWith({d.name}, 'S0013')) = [];
assert(numel(d) == 57, 'expected 57 participants, got %d', numel(d));
subjIDs = arrayfun(@(k) d(k).name(1:5), 1:57, 'UniformOutput', false)';
EXPECTED = arrayfun(@(n) sprintf('S%04d', n), [9 11 12 14:18 20:68], 'UniformOutput', false)';
assert(isequal(subjIDs, EXPECTED), 'participant ordering does not match the published cohort');

% participants for which the preprocessing pipeline applies a manual correction
manualCases = [1 2 5 7 9 11 12 14 16 18 26 28 30 32 36 40 41 42 43 47 48 50];

only = str2double(getenv('AB_SUBJ'));
if isnan(only), range = 1:57; else, range = only; end
nw = str2double(getenv('AB_WORKERS'));
if isnan(nw), nw = cfg.n_workers; end
if nw > 1 && isempty(gcp('nocreate')), parpool('Processes', nw); end

fprintf('==== union add-back source localisation ====\nset          : %s\noutput       : %s\nparticipants : %d\n\n', ...
    setName, outDir, numel(range));

parfor sub = range
    sid = subjIDs{sub};
    EEG = load(fullfile(eegDir, d(sub).name), '-mat');
    if isfield(EEG, 'EEG'), EEG = EEG.EEG; end
    nIC = size(EEG.icawinv, 2);
    nTr = EEG.trials;
    pnts = EEG.pnts;
    nbchan = EEG.nbchan;
    assert(pnts == N_SAMP, '%s: pnts=%d, expected %d', sid, pnts, N_SAMP);

    % ---- retained-component list, replaying the artefact-rejection rules of
    %      the preprocessing pipeline. Only `brain` is consumed downstream; the
    %      remaining assignments are kept so that this replay follows the
    %      preprocessing script step for step.
    C = EEG.etc.ic_classification.ICLabel.classifications;
    threshold = TH;
    t_mus   = find(C(:,2) > threshold)';
    t_eye   = find(C(:,3) > threshold)';
    t_heart = find(C(:,4) > threshold)';
    t_line  = find(C(:,5) > threshold)';
    t_chan  = find(C(:,6) > threshold)';
    if isempty(t_heart) == 1
        [~, t_heart] = max(C(1:5, 4));
    end
    all_artifacts = unique([t_mus, t_eye, t_heart, t_line, t_chan]);
    speak = find(C(:,1) < 0.1);
    speak = setdiff(speak, all_artifacts);
    brain = setdiff(1:nIC, [all_artifacts, speak']);
    if size(speak, 1) > 1, speak = speak'; end
    brain_auto = brain;

    % Participant-specific corrections. The indices are positions within the
    % current retained list and are applied in this order; for participant 16
    % the second removal refers to the already-shortened list.
    switch sub
        case 1,  t_eye = [t_eye, 12];        speak = [speak, brain([1])];    brain([1]) = [];
        case 2,                              speak = [speak, brain([1, 2])]; brain([1, 2]) = [];
        case 5,                              speak = [speak, brain([1])];    brain([1]) = [];
        case 7,  t_heart = [t_heart, brain([1])]; speak = [speak, brain([1])]; brain([1]) = [];
        case 9,  t_eye = [t_eye, brain(3)];  speak = [speak, brain([3])];    brain([3]) = [];
        case 11, t_eye = [t_eye, brain(4)];  speak = [speak, brain([4])];    brain([4]) = [];
        case 12, t_eye = [t_eye, brain(5)];  speak = [speak, brain([5])];    brain([5]) = [];
        case 14,                             speak = [speak, brain([1])];    brain([1]) = [];
        case 16, t_eye = [t_eye, brain(6)];  speak = [speak, brain([6])];    brain([6]) = [];
                                             speak = [speak, brain([2])];    brain([2]) = [];
        case 18, t_eye = [t_eye, brain([2, 3])]; speak = [speak, brain([2, 3])]; brain([2, 3]) = [];
        case 26, t_eye = [t_eye, brain(2)];  speak = [speak, brain([2])];    brain([2]) = [];
        case 28, t_eye = [t_eye, brain(7)];  speak = [speak, brain([7])];    brain([7]) = [];
        case 30, t_eye = [t_eye, brain(3)];  speak = [speak, brain([3])];    brain([3]) = [];
        case 32, t_eye = [t_eye, brain(6)];  speak = [speak, brain([6])];    brain([6]) = [];
        case 36, t_eye = [t_eye, brain(4)];  speak = [speak, brain([4])];    brain([4]) = [];
        case 40,                             speak = [speak, brain([2, 3])]; brain([2, 3]) = [];
        case 41,                             speak = [speak, brain([1, 2])]; brain([1, 2]) = [];
        case 42, t_eye = [t_eye, brain(5)];  speak = [speak, brain([5])];    brain([5]) = [];
        case 43, t_eye = [t_eye, brain([5])]; t_heart = [t_heart, brain([1])];
                                             speak = [speak, brain([1, 5])]; brain([1, 5]) = [];
        case 47,                             speak = [speak, brain([2, 3])]; brain([2, 3]) = [];
        case 48,                             speak = [speak, brain([3])];    brain([3]) = [];
        case 50,                             speak = [speak, brain([1])];    brain([1]) = [];
    end
    if ismember(sub, manualCases)
        assert(numel(brain) < numel(brain_auto), '%s: manual case removed nothing', sid);
    else
        assert(isequal(brain, brain_auto), '%s: not a manual case but the retained list changed', sid);
    end

    % ---- union component set for this participant --------------------------
    urow = find([U.sid] == str2double(sid(2:end)), 1);
    assert(~isempty(urow), '%s: not in the union table', sid);
    switch setName
        case 'brain',   addIC = [];
        case 'blink',   addIC = U(urow).blink(:)';
        case 'lateral', addIC = U(urow).lateral(:)';
    end
    addIC = unique(addIC);
    assert(all(addIC >= 1 & addIC <= nIC), '%s: component index outside 1..%d', sid, nIC);
    overlap = intersect(brain, addIC);
    assert(isempty(overlap), ...
        '%s: component(s) %s are already retained -- adding them back would count them twice', ...
        sid, mat2str(overlap));

    Sbrain = sort(brain(:))';
    Sadd   = sort(addIC(:))';
    Suse   = sort(union(Sbrain, Sadd));

    % ---- signal ------------------------------------------------------------
    fdt = fullfile(eegDir, char(EEG.data));
    fid = fopen(fdt, 'r');
    dat = fread(fid, [nbchan, pnts * nTr], 'float32');
    fclose(fid);
    assert(size(dat,2) == pnts * nTr, '%s: .fdt size mismatch', sid);
    icaind = EEG.icachansind(:)';
    assert(isequal(icaind, 1:nbchan), '%s: icachansind is not 1:%d', sid, nbchan);
    act = (EEG.icaweights * EEG.icasphere) * dat(icaind, :);

    labs = {EEG.chanlocs.labels};
    keepIdx = find(~ismember(labs, DROP_CHANS));
    assert(numel(keepIdx) == N_CHAN_KEEP, '%s: %d channels kept', sid, numel(keepIdx));

    % ---- head model --------------------------------------------------------
    bsSubj = ['Subject' sid(4:5)];
    kd = dir(fullfile(bstRoot, 'data', bsSubj, ['*' STUDY_SUFFIX], 'results_sLORETA*'));
    assert(isscalar(kd), '%s: %d sLORETA kernels', sid, numel(kd));
    K = load(fullfile(kd.folder, kd.name), 'ImagingKernel', 'GoodChannel');
    good = K.GoodChannel(:)';
    A = load(fullfile(bstRoot, 'anat', bsSubj, 'tess_cortex_pial_low.mat'), 'Atlas');
    ai = find(strcmp({A.Atlas.Name}, 'Destrieux'));
    assert(isscalar(ai) && numel(A.Atlas(ai).Scouts) == N_ROI, '%s: atlas check failed', sid);
    scouts = A.Atlas(ai).Scouts;
    verts = {scouts.Vertices};
    roiindex = cell(N_ROI, 2);
    for i = 1:N_ROI
        roiindex{i,1} = scouts(i).Vertices;
        roiindex{i,2} = scouts(i).Label;
    end

    % ---- reconstruct the set in use, and separately the retained and the
    %      added components for the linearity check
    sets = {Suse, Sbrain, Sadd};
    outs = cell(1,3);
    for c = 1:3
        Sc = sets{c};
        if isempty(Sc), outs{c} = []; continue, end
        R = EEG.icawinv(:, Sc) * act(Sc, :);
        R = R(keepIdx, :);
        R = R - mean(R, 1);
        R = reshape(R, N_CHAN_KEEP, pnts, nTr);
        Z = zeros(N_ROI, pnts, nTr);
        for tr = 1:nTr
            src = K.ImagingKernel * R(good, :, tr);
            for r = 1:N_ROI
                Z(r,:,tr) = mean(src(verts{r}, :), 1);
            end
        end
        outs{c} = Z;
    end

    if ~isempty(Sadd)
        sc = sqrt(mean(outs{1}(:).^2));
        rs = max(abs(outs{1}(:) - outs{2}(:) - outs{3}(:)));
        assert(rs <= 1e-6 * sc, '%s: X_union ~= X_brain + X_added (%.3g vs %.3g)', sid, rs, sc);
    end

    Z = outs{1};
    condition_data = cell(nTr,1);
    condition_data_type = cell(nTr,1);
    for tr = 1:nTr
        condition_data{tr} = Z(:,:,tr);
        condition_data_type{tr} = EEG.event(tr).type;
    end

    s = struct();
    s.condition_data = condition_data;
    s.condition_data_type = condition_data_type;
    s.roiindex = roiindex;
    s.kernel_name = 'sLORETA';
    s.freq_band = 'raw';
    s.variant = ['union_' setName];
    s.atlas_name = 'Destrieux';
    s.ics_brain = Sbrain;
    s.ics_added = Sadd;
    s.source_set = fullfile(eegDir, d(sub).name);

    parsave(fullfile(outDir, [bsSubj '_sLORETA_raw.mat']), s);
    fprintf('%-10s %-8s retained %3d + added %2d = %3d components   %3d trials\n', ...
        bsSubj, setName, numel(Sbrain), numel(Sadd), numel(Suse), nTr);
end

fprintf('\nDONE -> %s\n', outDir);

function parsave(f, s)
save(f, '-fromstruct', s, '-v7.3');
end
