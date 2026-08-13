function s10_empirical_chance_level()
%S10_EMPIRICAL_CHANCE_LEVEL  Empirical chance level of the phrase classifier.
%
%   Establishes the chance level of the five-way phrase classifier by label
%   permutation, rather than assuming the theoretical value of 1/5 = 20%.
%
%   The script runs three stages, in this order, because a permutation test is
%   interpretable only if the pipeline being permuted is the published one:
%
%     A  Reproduce the published classifier at full dimensionality
%        (148 ROIs x 150 samples = 22200 features per trial).
%
%     B  Verify that an economy-SVD reduction is lossless. A linear SVM depends
%        on the data only through the Gram matrix X*X'. With n trials and
%        p >> n features, the economy SVD X = U*S*V' gives Xr = U*S with
%        Xr*Xr' = U*S*S'*U' = X*X' exactly, so the SVM solution is unchanged
%        while the feature count drops from 22200 to at most n. This stage
%        asserts that every participant's accuracy is identical to stage A.
%        Without the reduction, 1000 permutations x 57 participants is
%        computationally prohibitive.
%
%     C  Run N_PERM label permutations per participant on the reduced features,
%        rebuilding the complete leave-one-block-out procedure for every
%        permutation.
%
%   Two null models are computed because they answer different questions:
%     global  labels shuffled freely within a participant, i.e. the standard
%             "labels carry no information" null.
%     within  labels shuffled within each block, which preserves the fixed
%             number of trials per phrase per block imposed by the design.
%
%   Two quantities are derived from each null and they are not interchangeable:
%   the CENTRE of the null is the empirical chance level, whereas its 95th
%   percentile is the per-participant significance threshold. Both are reported,
%   at the participant level and for the cohort mean.
%
%   INPUT
%     cfg.source   source-localised covert data, Subject*_sLORETA_raw.mat,
%                  Destrieux atlas, 148 ROIs, 250 Hz, -0.5 to +1.5 s per trial
%
%   OUTPUT (in cfg.out)
%     s10_empirical_chance_per_subject.csv   one row per participant
%     s10_empirical_chance_null.mat          full null matrices
%
%   PARAMETERS
%     window 0-600 ms after phrase onset, baseline-corrected on the 500 ms
%     pre-onset interval, features z-scored across trials; five-class linear
%     SVM (one-vs-all ECOC), leave-one-block-out cross-validation.
%
%   ENVIRONMENT
%     N_PERM     permutations per participant (default 1000)
%     N_WORKERS  parallel workers (default cfg.n_workers)
%     SUBJ_LIST  subset of participant indices, e.g. '1:5', for smoke tests

repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo, 'config'));
cfg = set_paths();

srcdir = cfg.source;
outdir = cfg.out;

fs        = 250;                    % Hz
pre_smp   = 0.5 * fs;               % 125 baseline samples
post_smp  = 1.5 * fs;               % 375 post-onset samples
win_smp   = round(0.6 * fs);        % 150 samples = 0-600 ms
n_roi     = 148;
n_perm    = env_num('N_PERM', 1000);
n_workers = env_num('N_WORKERS', cfg.n_workers);

d = dir(fullfile(srcdir, 'Subject*_sLORETA_raw.mat'));
subs = {d.name};
if ~isempty(getenv('SUBJ_LIST'))
    k = str2num(getenv('SUBJ_LIST')); %#ok<ST2NM>
    subs = subs(k(k >= 1 & k <= numel(subs)));
end
n = numel(subs);

fprintf('=== Empirical chance level of the phrase classifier ===\n');
fprintf('source  : %s\n', srcdir);
fprintf('subjects: %d   permutations: %d   workers: %d\n', n, n_perm, n_workers);
fprintf('window  : 0-600 ms, %d ROIs x %d samples = %d features\n\n', ...
        n_roi, win_smp, n_roi * win_smp);

if isempty(gcp('nocreate')), parpool('Processes', n_workers); end

acc_full = nan(n, 1); acc_red = nan(n, 1); gram_err = nan(n, 1);
null_g = nan(n, n_perm); null_w = nan(n, n_perm); ntr = nan(n, 1);

parfor ii = 1:n
    t0 = tic;
    S = load(fullfile(srcdir, subs{ii}), 'condition_data', 'condition_data_type');
    cd_ = S.condition_data; ct = S.condition_data_type;
    keep = ~cellfun(@isempty, cd_); cd_ = cd_(keep); ct = ct(keep);
    nt = numel(cd_);

    % Trial labels, e.g. 'C 1_u_1_b_2': character 3 is the phrase identity and
    % the last character is the block. Covert blocks are 2, 4, 6, 8 and 10, so
    % the last character (2, 4, 6, 8, 0) still identifies the block uniquely.
    y = zeros(nt, 1); b = zeros(nt, 1);
    for t = 1:nt
        y(t) = str2double(ct{t}(3));
        b(t) = str2double(ct{t}(end));
    end

    % ---- features, as in the published classifier ----
    X = zeros(nt, n_roi * win_smp);
    for t = 1:nt
        nd = cd_{t};
        assert(size(nd, 1) == n_roi, '%s: %d ROIs, expected %d', ...
               subs{ii}, size(nd, 1), n_roi);
        base = mean(nd(1:n_roi, 1:pre_smp), 2);
        ev   = nd(1:n_roi, (pre_smp+1):(pre_smp+post_smp)) - base;
        X(t, :) = reshape(ev(:, 1:win_smp).', 1, []);
    end
    ok = y >= 1 & y <= 5 & ~isnan(y);
    X = zscore(X(ok, :)); y = y(ok); b = b(ok);
    ntr(ii) = numel(y);

    % ---- A: published pipeline, full dimensionality ----
    acc_full(ii) = lobo_acc(X, y, b);

    % ---- B: economy-SVD reduction, Gram matrix preserved ----
    [U, Sv, ~] = svd(X, 'econ');
    Xr = U * Sv;
    gram_err(ii) = max(abs(X*X.' - Xr*Xr.'), [], 'all') / max(abs(X*X.'), [], 'all');
    acc_red(ii)  = lobo_acc(Xr, y, b);

    % ---- C: permutations on the reduced features ----
    ng = zeros(1, n_perm); nw = zeros(1, n_perm);
    ub = unique(b);
    for q = 1:n_perm
        ng(q) = lobo_acc(Xr, y(randperm(numel(y))), b);
        yw = y;
        for k = 1:numel(ub)
            m = find(b == ub(k));
            yw(m) = y(m(randperm(numel(m))));
        end
        nw(q) = lobo_acc(Xr, yw, b);
    end
    null_g(ii, :) = ng; null_w(ii, :) = nw;

    fprintf('%-28s full %.2f%%  reduced %.2f%%  gram_err %.1e  null_g %.2f%%  %.0fs\n', ...
            subs{ii}, acc_full(ii)*100, acc_red(ii)*100, gram_err(ii), ...
            mean(ng)*100, toc(t0));
end

%% ---------------- stage A / B verdicts ----------------
fprintf('\n--- A: reproduction of the published classifier ---\n');
fprintf('mean %.2f%% +/- %.2f%%   (published: 27.26%% +/- 6.97%%)\n', ...
        mean(acc_full)*100, std(acc_full)*100);

fprintf('\n--- B: the SVD reduction is lossless ---\n');
dmax = max(abs(acc_full - acc_red));
fprintf('max |acc_full - acc_reduced| = %.3g over %d participants\n', dmax, n);
fprintf('max relative Gram error      = %.3g\n', max(gram_err));
if dmax > 1e-12
    fprintf('*** reduction is not lossless: the permutation results below are not valid ***\n');
end

%% ---------------- stage C: nulls ----------------
p95_g = prctile(null_g, 95, 2);  p95_w = prctile(null_w, 95, 2);
mu_g  = mean(null_g, 2);         mu_w  = mean(null_w, 2);

fprintf('\n--- C: empirical null, %d permutations per participant ---\n', n_perm);
fprintf('%-34s %8s %8s\n', '', 'global', 'within-block');
fprintf('%-34s %7.2f%% %7.2f%%\n', 'null CENTRE (= chance level)', ...
        mean(mu_g)*100, mean(mu_w)*100);
fprintf('%-34s %7.2f%% %7.2f%%\n', 'null 95th pct (= signif. thresh.)', ...
        mean(p95_g)*100, mean(p95_w)*100);
fprintf('%-34s %7d  %7d\n', 'participants above own 95th pct', ...
        sum(acc_full > p95_g), sum(acc_full > p95_w));

% Cohort-level null: the mean accuracy across participants within each
% permutation, which is the reference the observed cohort mean is judged
% against.
gm_g = mean(null_g, 1); gm_w = mean(null_w, 1);
obs  = mean(acc_full);
fprintf('\ngroup mean accuracy %.2f%%\n', obs*100);
fprintf('group null (global)      : centre %.2f%%, 95th pct %.2f%%, p = %.4g\n', ...
        mean(gm_g)*100, prctile(gm_g, 95)*100, (1 + sum(gm_g >= obs)) / (n_perm + 1));
fprintf('group null (within-block): centre %.2f%%, 95th pct %.2f%%, p = %.4g\n', ...
        mean(gm_w)*100, prctile(gm_w, 95)*100, (1 + sum(gm_w >= obs)) / (n_perm + 1));

T = table(string(subs(:)), ntr, acc_full*100, acc_red*100, mu_g*100, p95_g*100, ...
          mu_w*100, p95_w*100, acc_full > p95_g, acc_full > p95_w, ...
    'VariableNames', {'file', 'n_trials', 'acc_pct', 'acc_reduced_pct', ...
                      'null_mean_global_pct', 'null_p95_global_pct', ...
                      'null_mean_within_pct', 'null_p95_within_pct', ...
                      'sig_global', 'sig_within'});
writetable(T, fullfile(outdir, 's10_empirical_chance_per_subject.csv'));
save(fullfile(outdir, 's10_empirical_chance_null.mat'), ...
     'null_g', 'null_w', 'acc_full', 'acc_red', 'subs', 'n_perm', '-v7');
fprintf('\nwrote %s\nDONE\n', outdir);
end


%% ---------------- helpers ----------------
function a = lobo_acc(X, y, b)
%LOBO_ACC  Leave-one-block-out five-class linear SVM accuracy.
pred = zeros(numel(y), 1);
ub = unique(b);
for k = 1:numel(ub)
    tr = b ~= ub(k); te = ~tr;
    if ~any(tr) || ~any(te), continue, end
    mdl = fitcecoc(X(tr, :), y(tr), ...
        'Learners', templateSVM('KernelFunction', 'linear'), ...
        'Coding', 'onevsall', 'verbose', 0);
    pred(te) = predict(mdl, X(te, :));
end
a = sum(pred == y) / numel(y);
end


function v = env_num(name, default_value)
%ENV_NUM  Numeric environment-variable override, or the supplied default.
s = getenv(name);
if isempty(s), v = default_value; else, v = str2double(s); end
end
