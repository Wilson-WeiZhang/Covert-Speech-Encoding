%% ========================================================================
%  s25_run_ica.m -- Stage 2 of the overt arm: independent component analysis
%
%  Decomposes the overt epochs from s24 with the same call sequence used for
%  the covert arm:
%      pop_runica(EEG, 'icatype', 'runica')   extended infomax, EEGLAB defaults
%      pop_iclabel(EEG, 'default')            ICLabel class probabilities
%      processMARA(ALLEEG, EEG, CURRENTSET)   MARA artefact scores
%
%  The overt arm gets its own decomposition rather than reusing the covert
%  weights: the two conditions were epoched from different parts of the
%  recording, so the component subspaces are not interchangeable.
%
%  REPRODUCIBILITY NOTE
%    runica starts from a random initialisation and the random generator is
%    not seeded here, so re-running this step yields a different but
%    statistically equivalent decomposition: component order, sign and exact
%    scalp maps are not preserved across runs. The decomposition used in
%    this study is distributed together with the data, and the component
%    indices recorded in s26_remove_artefact_components.m refer to that
%    decomposition. If ICA is re-run, the ocular components have to be
%    identified again on the new decomposition before s26 is used.
%
%  The step is resumable: subjects whose output already exists are skipped,
%  so an interrupted run can simply be repeated.
%
%  ENVIRONMENT (optional)
%    SUBJ_LIST   indices into the discovered input list, e.g. "1" or "1:5"
%
%  INPUT   <cfg.out>/overt_epochs/*_Filters_overt_processed_trials.set
%  OUTPUT  <cfg.out>/overt_ica/*_Filters_overt_processed_trials_precut_ICA.set
%% ========================================================================

clearvars
clc

addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'config'));
cfg = set_paths();

inDir  = fullfile(cfg.out, 'overt_epochs');
outDir = fullfile(cfg.out, 'overt_ica');
if ~isfolder(outDir), mkdir(outDir); end

addpath(cfg.eeglab); eeglab nogui;

fileslist = dir(fullfile(inDir, '*_Filters_overt_processed_trials.set'));
assert(~isempty(fileslist), 'no epoched overt datasets in %s (run s24 first)', inDir);
names = {fileslist.name};
fprintf('epoched datasets found: %d\n', numel(names));

idx = 1:numel(names);
subj_list_str = getenv('SUBJ_LIST');
if ~isempty(subj_list_str)
    idx = intersect(idx, str2num(subj_list_str)); %#ok<ST2NM>
    fprintf('SUBJ_LIST override: %s\n', mat2str(idx));
end

% resume: keep only the subjects without an output file
todo = idx(arrayfun(@(i) ~isfile(fullfile(outDir, strrep(names{i}, '.set', '_precut_ICA.set'))), idx));
fprintf('already done: %d   to run: %d\n', numel(idx) - numel(todo), numel(todo));
if isempty(todo), fprintf('nothing to do\n'); return; end

if isempty(gcp('nocreate')), parpool('Processes', cfg.n_workers); end

t0 = tic;
parfor k = 1:numel(todo)
    name = names{todo(k)};
    sid  = name(1:5);
    ts = tic;

    EEG = pop_loadset('filename', name, 'filepath', inDir);

    EEG = pop_runica(EEG, 'icatype', 'runica');
    EEG = pop_iclabel(EEG, 'default');

    ALLEEG = [];
    CURRENTSET = [];
    [ALLEEG, EEG, CURRENTSET] = processMARA(ALLEEG, EEG, CURRENTSET); %#ok<ASGLU>

    EEG = pop_saveset(EEG, 'filename', strrep(name, '.set', '_precut_ICA.set'), ...
        'filepath', outDir);

    fprintf('%s  nbchan=%d trials=%d nIC=%d  %.1f min\n', ...
        sid, EEG.nbchan, EEG.trials, size(EEG.icaweights, 1), toc(ts)/60);
end
fprintf('\nfinished in %.1f min -> %s\n', toc(t0)/60, outDir);
