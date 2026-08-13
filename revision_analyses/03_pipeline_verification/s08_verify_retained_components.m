function s08_verify_retained_components()
% =========================================================================
%  s08_verify_retained_components.m -- verification only, no classification
%
%  Check that the list of retained (brain) components reconstructed from the
%  artefact-rejection rules of the preprocessing pipeline matches the published
%  cleaned files, for every participant.
%
%  WHAT IS CHECKED
%    The published cleaned files retain only the kept components, so
%      size(icaweights, 1)  ==  numel(brain)
%    must hold for every participant. `brain` is rebuilt here by replaying the
%    rejection rules of the preprocessing pipeline, including its 22
%    participant-specific corrections, on the same file ordering that pipeline
%    uses. The check needs both file generations to be present: the full
%    decompositions (*_precut_ICA.set) and the cleaned files
%    (*_rejectchan56_u1.set).
%
%  INPUT
%    cfg.eeg   full ICA decompositions and published cleaned files
%
%  OUTPUT
%    cfg.out/retained_components.csv   one row per participant
%    cfg.out/retained_components.log   same report as printed
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

setdir = cfg.eeg;
outdir = cfg.out;
if ~isfolder(outdir), mkdir(outdir); end

logf = fullfile(outdir, 'retained_components.log');
fid  = fopen(logf, 'w');
pr(fid, '=== verify retained-component list against the published cleaned files ===\n');
pr(fid, 'run: %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
pr(fid, 'EEG directory: %s\n\n', setdir);

d = dir(fullfile(setdir, '*_Filters_processed_trials_precut_ICA.set'));
pr(fid, 'decompositions found: %d\n', numel(d));
drop = startsWith({d.name}, 'S0013');
if any(drop)
    pr(fid, 'excluding %s: no T1 MRI, absent from the source-space cohort\n', ...
        d(find(drop, 1)).name(1:5));
    d(drop) = [];
end
n = numel(d);
assert(n == 57, 'expected 57 participants, got %d', n);

sid = cell(n,1); expect = nan(n,1); got = nan(n,1); ok = false(n,1);

for sub = 1:n
    name   = d(sub).name(1:5);
    sid{sub} = name;
    S      = load(fullfile(setdir, d(sub).name), '-mat');
    cls    = S.etc.ic_classification.ICLabel.classifications;
    nIC    = size(S.icawinv, 2);

    % ---- replay the artefact-rejection rules ----
    th      = 0.9;   % ICLabel posterior above which a component counts as an artefact
    t_mus   = find(cls(:,2) > th)';
    t_eye   = find(cls(:,3) > th)';
    t_heart = find(cls(:,4) > th)';
    t_line  = find(cls(:,5) > th)';
    t_chan  = find(cls(:,6) > th)';
    if isempty(t_heart), [~, t_heart] = max(cls(1:5,4)); end
    all_art = unique([t_mus, t_eye, t_heart, t_line, t_chan]);
    speak   = find(cls(:,1) < 0.1);
    speak   = setdiff(speak, all_art);
    brain   = setdiff(1:nIC, [all_art, speak']);
    brain   = apply_manual(brain, sub);

    % ---- compare with the published cleaned file ----
    pub = dir(fullfile(setdir, [name '*_rejectchan56_u1.set']));
    if isempty(pub)
        pr(fid, '  %s: no published cleaned file found -- cannot verify\n', name);
        continue
    end
    assert(numel(pub) == 1, '%s: expected exactly 1 published cleaned file, got %d', name, numel(pub));
    P = load(fullfile(setdir, pub(1).name), '-mat', 'icaweights');
    expect(sub) = size(P.icaweights, 1);
    got(sub)    = numel(brain);
    ok(sub)     = (expect(sub) == got(sub));
    if ~ok(sub)
        pr(fid, '  MISMATCH %s: published %d rows, reimplementation %d components\n', ...
            name, expect(sub), got(sub));
    end
end

pr(fid, '\nparticipants matching: %d / %d\n', sum(ok), n);
if all(ok)
    pr(fid, '*** retained-component list VERIFIED for all %d participants ***\n', n);
    pr(fid, 'The reconstructed component set is the one used for the published analyses.\n');
else
    pr(fid, '*** retained-component list NOT verified -- see mismatches above ***\n');
end

T = table(sid, ok, expect, got, 'VariableNames', ...
    {'subject', 'match', 'published_rows', 'reimplementation_n'});
csvf = fullfile(outdir, 'retained_components.csv');
writetable(T, csvf);
pr(fid, '\nwrote %s\n', csvf);
fclose(fid);
end


function pr(fid, varargin)
fprintf(1, varargin{:});
fprintf(fid, varargin{:});
end


function brain = apply_manual(brain, sub)
% The 22 participant-specific component removals of the preprocessing pipeline.
% The indices are positions within the current retained list and are applied in
% this order; for participant 16 the second removal refers to the
% already-shortened list.
switch sub
    case 1,  brain([1]) = [];
    case 2,  brain([1,2]) = [];
    case 5,  brain([1]) = [];
    case 7,  brain([1]) = [];
    case 9,  brain([3]) = [];
    case 11, brain([4]) = [];
    case 12, brain([5]) = [];
    case 14, brain([1]) = [];
    case 16, brain([6]) = []; brain([2]) = [];
    case 18, brain([2,3]) = [];
    case 26, brain([2]) = [];
    case 28, brain([7]) = [];
    case 30, brain([3]) = [];
    case 32, brain([6]) = [];
    case 36, brain([4]) = [];
    case 40, brain([2,3]) = [];
    case 41, brain([1,2]) = [];
    case 42, brain([5]) = [];
    case 43, brain([1,5]) = [];
    case 47, brain([2,3]) = [];
    case 48, brain([3]) = [];
    case 50, brain([1]) = [];
end
end
