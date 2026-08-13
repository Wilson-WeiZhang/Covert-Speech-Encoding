function s11_artefact_component_decoding()
%S11_ARTEFACT_COMPONENT_DECODING  Phrase decoding from artefact components alone.
%
%   Removing an independent component that is classified as an artefact does not
%   by itself establish that the component carried no phrase-discriminative
%   signal. The direct test is to classify phrase identity from those components
%   ALONE and to compare the result against a permutation null built from the
%   same data.
%
%   THREE FEATURE SETS, from the same participants and the same 0-600 ms window
%   used by the published classifier:
%     ocular  the rater-confirmed blink and lateral eye-movement components
%             (59 blink + 61 lateral = 120 components across the cohort).
%             These are exactly the components the published preprocessing
%             removed.
%     muscle  components with ICLabel Muscle probability > 0.9, i.e. the rule
%             used by the published preprocessing.
%     brain   the components the published preprocessing KEPT. This is the
%             positive control and the set the main analysis was built from.
%
%   RECOVERING THE KEPT SET. The published cleaning script computes the kept
%   component list but does not store it. This function re-implements that
%   script, including its per-participant manual adjustments, and then VERIFIES
%   the result: the published cleaned file retains only the kept components, so
%   size(icaweights, 1) in that file must equal the number of components in the
%   reconstructed list. Any mismatch is reported per participant, because a
%   silently wrong list would make the positive control meaningless.
%
%   Feature counts differ across the three sets (a participant has roughly 2-4
%   ocular, 5-20 muscle and 15-48 brain components), so accuracies are NOT
%   compared across sets. Each set is compared against ITS OWN permutation null,
%   which is what "carries phrase information" means here.
%
%   INPUT
%     cfg.eeg   preprocessed EEG, *_Filters_processed_trials_precut_ICA.set with
%               the accompanying .fdt and the ICLabel classification, plus the
%               published cleaned *_rejectchan56_u1.set used for verification
%
%   OUTPUT (in cfg.out)
%     s11_artefact_component_decoding.csv           one row per participant
%     s11_artefact_component_decoding_summary.txt   cohort summary
%     s11_artefact_component_decoding.mat           full null matrices
%
%   PARAMETERS
%     window 0-600 ms after phrase onset; component activations z-scored across
%     trials; five-class linear SVM (one-vs-all ECOC), leave-one-block-out
%     cross-validation; within-block label permutation, which preserves the
%     fixed number of trials per phrase per block imposed by the design.
%
%   ENVIRONMENT
%     N_PERM     permutations per participant (default 1000)
%     N_WORKERS  parallel workers (default cfg.n_workers)
%     USE_SVD    1 classifies on the economy-SVD reduction of the features,
%                which is lossless for a linear SVM; 0 (default) classifies on
%                the raw features

repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo, 'config'));
cfg = set_paths();

setdir = fullfile(cfg.eeg, filesep);
outdir = fullfile(cfg.out, filesep);

fs = 250; pre_smp = 0.5*fs; win_smp = round(0.6*fs);   % 0-600 ms
n_perm    = env_num('N_PERM', 1000);
n_workers = env_num('N_WORKERS', cfg.n_workers);
use_svd   = env_num('USE_SVD', 0);

% Fixed seed offset: participant ii draws its permutations from
% rng(SEED_BASE + ii), so the null distributions are reproducible.
SEED_BASE = 20260803;

% Rater-confirmed ocular components, one row per participant:
% {numeric participant ID, blink component(s), lateral component(s)}.
cov_manual = { ...
     9 2 12; 11 2 11; 12 2 5; 14 1 5; 15 2 3; 16 2 9; 17 2 20; 18 3 14; 20 1 5; ...
    21 2 3; 22 2 6; 23 2 7; 24 2 6; 25 2 13; 26 2 7; 27 4 11; 28 2 7; 29 2 [4,5]; ...
    30 2 5; 31 1 7; 32 3 19; 33 3 4; 34 2 5; 35 1 4; 36 4 15; 37 [1,2] [4,6]; ...
    38 2 3; 39 2 12; 40 [3,5] 2; 41 4 5; 42 1 6; 43 2 13; 44 2 [4,8]; 45 1 5; ...
    46 1 3; 47 1 6; 48 2 4; 49 2 5; 50 2 6; 51 4 2; 52 2 10; 53 5 13; 54 2 10; ...
    55 1 4; 56 1 3; 57 1 6; 58 1 8; 59 1 9; 60 2 3; 61 2 4; 62 2 5; 63 1 7; ...
    64 1 5; 65 1 3; 66 2 [11,14]; 67 1 4; 68 2 7};
man_ids = cell2mat(cov_manual(:, 1));

d = dir([setdir '*_Filters_processed_trials_precut_ICA.set']);
sids = arrayfun(@(x) x.name(1:5), d, 'UniformOutput', false);
% S0013 has no structural MRI and therefore no source reconstruction, so it is
% not part of the analysed cohort. It is dropped by identifier, never by
% position in the directory listing.
if numel(d) == 58, d(strcmp(sids, 'S0013')) = []; end
n = numel(d);

fprintf('=== Phrase decoding from artefact components ===\n');
fprintf('participants: %d   permutations: %d   workers: %d\n', n, n_perm, n_workers);
fprintf('window: 0-600 ms (%d samples), same as the published classifier\n', win_smp);
fprintf('seed base: %d   use_svd: %d\n\n', SEED_BASE, use_svd);

if isempty(gcp('nocreate')), parpool('Processes', n_workers); end

sets = {'ocular', 'muscle', 'brain'};
acc  = nan(n, 3); nic = nan(n, 3);
nullc = nan(n, 3); null95 = nan(n, 3); pperm = nan(n, 3);
kept_ok = false(n, 1); kept_expect = nan(n, 1); kept_got = nan(n, 1);

% The full permutation matrix is retained so that the null distribution of the
% COHORT MEAN can be formed (mean across participants within each permutation),
% which is the reference a cohort mean must be judged against.
nullall = nan(n, 3, n_perm);

parfor ii = 1:n
    rng(SEED_BASE + ii, 'twister');
    name = d(ii).name(1:5);
    S = load([setdir d(ii).name], '-mat');
    cls = S.etc.ic_classification.ICLabel.classifications;
    nIC = size(S.icawinv, 2);

    % ---------- reconstruct the published component selection ----------
    th = 0.9;
    t_mus   = find(cls(:,2) > th)';
    t_eye   = find(cls(:,3) > th)';
    t_heart = find(cls(:,4) > th)';
    t_line  = find(cls(:,5) > th)';
    t_chan  = find(cls(:,6) > th)';
    if isempty(t_heart), [~, t_heart] = max(cls(1:5, 4)); end
    all_art = unique([t_mus, t_eye, t_heart, t_line, t_chan]);
    speak = find(cls(:,1) < 0.1);
    speak = setdiff(speak, all_art);
    brain = setdiff(1:nIC, [all_art, speak']);
    brain = apply_manual(brain, ii);

    % ---------- verify the kept set against the published cleaned file ----------
    pub = dir([setdir name '*_rejectchan56_u1.set']);
    e = NaN; g = NaN; okk = false;
    if ~isempty(pub)
        P = load([setdir pub(1).name], '-mat', 'icaweights');
        e = size(P.icaweights, 1); g = numel(brain); okk = (e == g);
    end
    kept_expect(ii) = e; kept_got(ii) = g; kept_ok(ii) = okk;

    % ---------- component sets ----------
    r = find(man_ids == str2double(name(2:end)), 1);
    ocular = [];
    if ~isempty(r)
        ocular = unique([tolist(cov_manual{r,2}), tolist(cov_manual{r,3})]);
        ocular = ocular(ocular >= 1 & ocular <= nIC);
    end
    sets_ic = {ocular, t_mus, brain};

    % ---------- component activations ----------
    fdt = [setdir d(ii).name(1:end-4) '.fdt'];
    fid = fopen(fdt, 'r');
    raw = fread(fid, [double(S.nbchan), double(S.pnts)*double(S.trials)], 'float32');
    fclose(fid);
    ici = double(S.icachansind(:))';
    act = (S.icaweights * S.icasphere) * raw(ici, :);
    act = reshape(act, size(act,1), double(S.pnts), double(S.trials));

    % ---------- trial labels ----------
    nt = double(S.trials);
    y = zeros(nt, 1); b = zeros(nt, 1);
    for t = 1:nt
        ev = S.epoch(t).eventtype;    if ~iscell(ev), ev = {ev}; end
        lt = S.epoch(t).eventlatency; if ~iscell(lt), lt = {lt}; end
        for k = 1:numel(ev)
            % Named evs rather than s: s is the feature-set loop variable below,
            % and parfor cannot classify a variable used in both roles.
            evs = ev{k}; if ~ischar(evs), continue, end
            tk = regexp(evs, '^[CO]\s*(\d)_u_\d_b_(\d+)', 'tokens', 'once');
            if ~isempty(tk) && abs(lt{k}) < 1e-6
                y(t) = str2double(tk{1}); b(t) = str2double(tk{2}); break
            end
        end
    end

    a_i = nan(1,3); nc = nan(1,3); c_i = nan(1,3); p95_i = nan(1,3); pp_i = nan(1,3);
    nul_i = nan(3, n_perm);
    for s = 1:3
        ics = sets_ic{s};
        nc(s) = numel(ics);
        if isempty(ics) || all(y == 0), continue, end
        seg = act(ics, (pre_smp+1):(pre_smp+win_smp), :);   % components x 150 x trials
        X = reshape(permute(seg, [3 2 1]), nt, []);         % trials x (150 * components)
        ok = y >= 1 & y <= 5;
        X = zscore(X(ok, :)); yy = y(ok); bb = b(ok);
        if use_svd
            [U, Sv, ~] = svd(X, 'econ'); Xr = U * Sv;   % lossless for a linear SVM
        else
            Xr = X;
        end
        a_i(s) = lobo(Xr, yy, bb);
        nul = zeros(1, n_perm);
        ub = unique(bb);
        for q = 1:n_perm
            yp = yy;
            for k = 1:numel(ub)
                m = find(bb == ub(k));
                yp(m) = yy(m(randperm(numel(m))));          % within-block shuffle
            end
            nul(q) = lobo(Xr, yp, bb);
        end
        nul_i(s, :) = nul;
        c_i(s) = mean(nul); p95_i(s) = prctile(nul, 95);
        pp_i(s) = (1 + sum(nul >= a_i(s))) / (n_perm + 1);
    end
    acc(ii,:) = a_i; nic(ii,:) = nc;
    nullc(ii,:) = c_i; null95(ii,:) = p95_i; pperm(ii,:) = pp_i;
    nullall(ii,:,:) = nul_i;
    fprintf('%-6s keep %d/%d %s | ocular %d IC %.1f%% | muscle %d IC %.1f%% | brain %d IC %.1f%%\n', ...
        name, g, e, string(okk), nc(1), a_i(1)*100, nc(2), a_i(2)*100, nc(3), a_i(3)*100);
end

%% ---------------- verification gate ----------------
fprintf('\n--- kept-set verification against the published cleaned files ---\n');
fprintf('participants matching: %d / %d\n', sum(kept_ok), n);
if ~all(kept_ok)
    bad = find(~kept_ok);
    for k = bad(:)'
        fprintf('  MISMATCH %s: reconstruction %d, published file %d\n', ...
                d(k).name(1:5), kept_got(k), kept_expect(k));
    end
    fprintf('*** kept list not verified: the brain row below is not the published set ***\n');
end

%% ---------------- summary ----------------
L = {sprintf('Artefact-component decoding, N = %d, %d within-block permutations', n, n_perm)
     'window 0-600 ms; each set tested against ITS OWN null (feature counts differ)'
     ''
     sprintf('%-8s %6s %9s %9s %9s %9s %s', 'set', 'nIC', 'acc', 'null', 'null95', 'p', 'participants p<.05')};
for s = 1:3
    v = ~isnan(acc(:, s));
    L{end+1} = sprintf('%-8s %6.1f %8.2f%% %8.2f%% %8.2f%% %9.4g %d/%d', sets{s}, ...
        mean(nic(v,s)), mean(acc(v,s))*100, mean(nullc(v,s))*100, mean(null95(v,s))*100, ...
        signrank_safe(acc(v,s), nullc(v,s)), sum(pperm(v,s) < 0.05), sum(v)); %#ok<AGROW>
end
fid = fopen([outdir 's11_artefact_component_decoding_summary.txt'], 'w');
for k = 1:numel(L), fprintf(fid, '%s\n', L{k}); fprintf('%s\n', L{k}); end
fclose(fid);

sid_col = string(arrayfun(@(x) x.name(1:5), d, 'UniformOutput', false));
sid_col = sid_col(:);                      % force a column for table()
T = table(sid_col, kept_ok, ...
    nic(:,1), acc(:,1)*100, nullc(:,1)*100, null95(:,1)*100, pperm(:,1), ...
    nic(:,2), acc(:,2)*100, nullc(:,2)*100, null95(:,2)*100, pperm(:,2), ...
    nic(:,3), acc(:,3)*100, nullc(:,3)*100, null95(:,3)*100, pperm(:,3), ...
    'VariableNames', {'subject', 'kept_verified', ...
      'n_ocular', 'acc_ocular', 'null_ocular', 'null95_ocular', 'p_ocular', ...
      'n_muscle', 'acc_muscle', 'null_muscle', 'null95_muscle', 'p_muscle', ...
      'n_brain', 'acc_brain', 'null_brain', 'null95_brain', 'p_brain'});
writetable(T, [outdir 's11_artefact_component_decoding.csv']);
save([outdir 's11_artefact_component_decoding.mat'], ...
     'acc', 'nic', 'nullc', 'null95', 'pperm', 'sets', 'kept_ok', ...
     'nullall', 'SEED_BASE', 'use_svd', 'n_perm', '-v7.3');
fprintf('\nwrote %s\nDONE\n', outdir);
end


%% ---------------- helpers ----------------
function brain = apply_manual(brain, subject_pos)
%APPLY_MANUAL  Per-participant manual adjustments of the published cleaning step.
%   The adjustments are expressed as POSITIONS in the current kept-component
%   list and are applied in the original order, so for participant 16 the second
%   removal indexes the already-shortened list. subject_pos is the position of
%   the participant in the directory listing, which is the indexing the
%   published script used; the reconstruction is verified per participant
%   against the published cleaned file.
switch subject_pos
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


function a = lobo(X, y, b)
%LOBO  Leave-one-block-out five-class linear SVM accuracy.
pred = zeros(numel(y), 1); ub = unique(b);
for k = 1:numel(ub)
    tr = b ~= ub(k); te = ~tr;
    if ~any(tr) || ~any(te), continue, end
    mdl = fitcecoc(X(tr,:), y(tr), 'Learners', ...
        templateSVM('KernelFunction', 'linear'), 'Coding', 'onevsall', 'verbose', 0);
    pred(te) = predict(mdl, X(te,:));
end
a = sum(pred == y) / numel(y);
end


function v = tolist(x)
%TOLIST  Normalise a scalar, vector or cell entry to a row vector.
if iscell(x), v = [x{:}]; else, v = x(:)'; end
end


function p = signrank_safe(a, b)
%SIGNRANK_SAFE  Wilcoxon signed-rank test, falling back to a paired t-test.
try
    p = signrank(a, b);
catch
    [~, p] = ttest(a, b);
end
end


function v = env_num(name, default_value)
%ENV_NUM  Numeric environment-variable override, or the supplied default.
s = getenv(name);
if isempty(s), v = default_value; else, v = str2double(s); end
end
