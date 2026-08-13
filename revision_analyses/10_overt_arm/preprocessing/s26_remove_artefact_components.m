%% ========================================================================
%  s26_remove_artefact_components.m -- Stage 3 of the overt arm
%
%  Removes artefact components from the overt ICA decompositions and writes
%  the cleaned 56-channel, average-referenced datasets that the source
%  projection consumes. The rule set is the one used for the covert arm, so
%  that the two conditions are cleaned identically.
%
%  AUTOMATIC LAYER
%    - ICLabel probability > 0.9 in any of the muscle, eye, heart, line-noise
%      or channel-noise classes marks a component as an artefact
%    - if no component reaches 0.9 for heart, the component with the highest
%      heart probability among the first five is taken instead, so that every
%      subject contributes one cardiac component
%    - components with a brain probability below 0.1 that are not already
%      artefacts are collected separately ("speak"); they are removed as well
%    - everything else is kept, back-projected with pop_subcomp, reduced to
%      56 channels by dropping {Fp1, Fp2, Fpz, ECG, TP9, TP10, FT9, FT10},
%      and re-referenced to the average
%
%  MANUAL LAYER
%    ICLabel is reliable for blinks but not for lateral (horizontal saccade)
%    components, which frequently score below the 0.9 threshold. The
%    automatic layer is therefore followed by a manual one, filled in for
%    this dataset in the `switch sub` block below: the ocular candidates of
%    every subject were inspected on their scalp maps and single-trial time
%    courses, and confirmed blink or lateral-saccade components that had
%    survived the automatic rule were moved out of the keep-list. Seven
%    subjects carry manual entries -- S0034, S0036, S0039, S0043, S0044,
%    S0047 and S0054 -- and the remaining fifty were inspected and needed no
%    change. Components are named by absolute number, that is by the number
%    their scalp map is labelled with, rather than by position within the
%    keep-list, so the entries do not shift if the automatic layer changes.
%
%  REPRODUCIBILITY NOTE
%    The manual component numbers refer to the ICA decomposition distributed
%    with this study. runica is randomly initialised (see s25_run_ica.m), so
%    a freshly computed decomposition assigns different numbers, orders and
%    signs to the same physiological sources. Running this script against a
%    re-computed ICA requires the ocular components to be identified again on
%    that decomposition. The automatic layer needs no such adjustment.
%
%  S0013 is excluded: it has no structural MRI and therefore no source
%  reconstruction, so it is absent from every downstream dataset. It is
%  dropped by identifier and the remaining 57 identifiers are asserted
%  against the expected list, so an incomplete Stage-2 output fails here
%  instead of shifting every `sub` index and silently misapplying the
%  manual entries.
%
%  INPUT   <cfg.out>/overt_ica/*_Filters_overt_processed_trials_precut_ICA.set
%  OUTPUT  <cfg.out>/overt_clean/
%              *_precut_ICA_ICACUT_rejectchan56_o1.set (+ .fdt)
%              s26_clean_report.csv     per subject: component counts and keep-list
%              s26_kept_components.mat  the keep-list per subject
%% ========================================================================

clearvars; clc; close all;

addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'config'));
cfg = set_paths();

address    = fullfile(cfg.out, 'overt_ica');
outaddress = fullfile(cfg.out, 'overt_clean');
if ~isfolder(outaddress), mkdir(outaddress); end

addpath(cfg.eeglab); eeglab nogui;

d = dir(fullfile(address, '*_Filters_overt_processed_trials_precut_ICA.set'));
assert(~isempty(d), 'no ICA datasets in %s (run s25 first)', address);

% Reference channel description for the ECG channel: the first six datasets
% take their ECG chanlocs entry from the seventh dataset, so that the channel
% description is the same for every subject. Read before any dataset is
% excluded, so that the index is stable.
EEG = pop_loadset('filename', d(7).name, 'filepath', address);
ecginfo = EEG.chanlocs(32);

% ---- exclude S0013 and pin down the sub -> subject mapping ----
sids = cellfun(@(x) x(1:5), {d.name}, 'UniformOutput', false);
d(strcmp(sids, 'S0013')) = [];
sids = cellfun(@(x) x(1:5), {d.name}, 'UniformOutput', false);
expected = {'S0009','S0011','S0012','S0014','S0015','S0016','S0017','S0018', ...
            'S0020','S0021','S0022','S0023','S0024','S0025','S0026','S0027', ...
            'S0028','S0029','S0030','S0031','S0032','S0033','S0034','S0035', ...
            'S0036','S0037','S0038','S0039','S0040','S0041','S0042','S0043', ...
            'S0044','S0045','S0046','S0047','S0048','S0049','S0050','S0051', ...
            'S0052','S0053','S0054','S0055','S0056','S0057','S0058','S0059', ...
            'S0060','S0061','S0062','S0063','S0064','S0065','S0066','S0067','S0068'};
assert(isequal(sids(:)', expected), ...
    ['subject list does not match the expected 57 identifiers (found %d). ' ...
     'The Stage-2 output is incomplete; fix that before running, otherwise ' ...
     'the manual entries below apply to the wrong subjects.'], numel(sids));

n_sub = numel(expected);
kept_components = cell(n_sub, 1);
rep             = cell(n_sub, 1);

for sub = 1:n_sub
    name = d(sub).name(1:5);
    EEG  = pop_loadset('filename', d(sub).name, 'filepath', address);

    if ismember(sub, 1:6)
        for i = 1:size(EEG.chanlocs, 2)
            if isequal(EEG.chanlocs(i).labels, 'ECG')
                EEG.chanlocs(i) = ecginfo;
                break
            end
        end
    end

    %% ---------------- automatic layer ----------------
    threshold = 0.9;
    cls = EEG.etc.ic_classification.ICLabel.classifications;
    t_mus   = find(cls(:, 2) > threshold)';
    t_eye   = find(cls(:, 3) > threshold)';
    t_heart = find(cls(:, 4) > threshold)';
    t_line  = find(cls(:, 5) > threshold)';
    t_chan  = find(cls(:, 6) > threshold)';

    if isempty(t_heart)
        [~, t_heart] = max(cls(1:5, 4));
    end

    all_artifacts = unique([t_mus, t_eye, t_heart, t_line, t_chan]);

    speak = find(cls(:, 1) < 0.1);
    speak = setdiff(speak, all_artifacts);
    brain = setdiff(1:size(EEG.icawinv, 2), [all_artifacts, speak']);

    if size(speak, 1) > 1
        speak = speak';
    end

    n_auto = numel(brain);

    %% ---------------- manual layer ----------------
    % Absolute component numbers of the confirmed ocular components per
    % subject; blink and lateral saccade are listed separately for the
    % record but treated identically -- both join the eye set and both leave
    % the keep-list. Subjects that do not appear here were inspected and
    % required no change.
    manual_blink = []; manual_lateral = [];
    switch sub
        case 23, manual_blink = 2;     manual_lateral = [4 7];   % S0034
        case 25, manual_blink = 2;     manual_lateral = 5;       % S0036
        case 28, manual_blink = 2;     manual_lateral = 15;      % S0039
        case 32, manual_blink = 2;     manual_lateral = [3 4];   % S0043
        case 33, manual_blink = 2;     manual_lateral = [4 7];   % S0044
        case 36, manual_blink = 1;     manual_lateral = 8;       % S0047
        case 43, manual_blink = [1 4]; manual_lateral = 5;       % S0054
    end

    manual_eye = unique([manual_blink, manual_lateral]);
    if ~isempty(manual_eye)
        n_ic_total = size(EEG.icawinv, 2);
        bad = manual_eye(manual_eye < 1 | manual_eye > n_ic_total | ...
                         manual_eye ~= fix(manual_eye));
        assert(isempty(bad), ...
            'sub %d (%s): manual component %s is outside 1..%d.', ...
            sub, name, mat2str(bad), n_ic_total);

        already = setdiff(manual_eye, brain);   % already caught automatically
        newly   = intersect(manual_eye, brain); % removed by the manual layer
        t_eye = unique([t_eye, manual_eye]);
        brain = setdiff(brain, manual_eye);
        fprintf(['  manual: blink %s lateral %s | %d already removed automatically (%s), ' ...
                 '%d removed here (%s)\n'], ...
            mat2str(manual_blink), mat2str(manual_lateral), ...
            numel(already), mat2str(already), numel(newly), mat2str(newly));
    end

    %% ---------------- back-projection and re-referencing ----------------
    EEG = pop_subcomp(EEG, brain, 0, 1);
    EEG = pop_select(EEG, 'nochannel', {'Fp1', 'Fp2', 'Fpz', 'ECG', 'TP9', 'TP10', 'FT9', 'FT10'});
    EEG = eeg_checkset(EEG);
    EEG = pop_reref(EEG, []);

    kept_components{sub} = brain;
    EEG = pop_saveset(EEG, ...
        'filename', [name, '_Filters_overt_processed_trials_precut_ICA_ICACUT_rejectchan56_o1.set'], ...
        'filepath', outaddress);

    rep{sub} = sprintf('%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"%s"', ...
        sub, name, size(EEG.icawinv, 2), n_auto, numel(brain), n_auto - numel(brain), ...
        numel(t_mus), numel(t_eye), numel(t_heart), numel(t_line), numel(t_chan), ...
        numel(speak), num2str(brain));
    fprintf('sub %2d %s : automatic keep %d -> final keep %d (manual removed %d)\n', ...
        sub, name, n_auto, numel(brain), n_auto - numel(brain));
end

fid = fopen(fullfile(outaddress, 's26_clean_report.csv'), 'w');
fprintf(fid, ['sub,subject,n_ic_total,n_keep_auto,n_keep_final,n_manual_removed,' ...
              'n_mus,n_eye,n_heart,n_line,n_chan,n_speak,kept_components\n']);
for sub = 1:n_sub
    if ~isempty(rep{sub}), fprintf(fid, '%s\n', rep{sub}); end
end
fclose(fid);
save(fullfile(outaddress, 's26_kept_components.mat'), 'kept_components');
fprintf('\nreport: %s\n', fullfile(outaddress, 's26_clean_report.csv'));
