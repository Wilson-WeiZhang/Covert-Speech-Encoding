function s12_cohort_level_null()
%S12_COHORT_LEVEL_NULL  Cohort-mean permutation null for the phrase classifier.
%
%   Computes the permutation null of the COHORT MEAN accuracy, for the full
%   148-parcel classifier and for the classifier that excludes every occipital
%   and ventral occipitotemporal parcel.
%
%   A per-participant null summarised by its centre and percentiles does not
%   determine the null of the cohort mean: averaging across participants reduces
%   the spread by roughly sqrt(N). This script therefore keeps the FULL
%   permutation matrix (nullall, participants x conditions x permutations), so
%   the null of the cohort mean can be formed by averaging across participants
%   WITHIN each shuffle. The centre, standard deviation, central 95% interval,
%   95th percentile, maximum and permutation p value reported below are all
%   computed from that distribution.
%
%   TWO CONDITIONS, same participants, same 0-600 ms window:
%     all         148 parcels - reproduction gate, must return the published
%                 27.26% cohort mean
%     non_visual  114 parcels - the 34 occipital and ventral occipitotemporal
%                 parcels removed
%
%   The excluded parcel set, the baseline correction, the window, the z-scoring
%   and economy-SVD reduction, the leave-one-block-out cross-validation and the
%   within-block label permutation are identical to
%   s13_exclude_visual_rois.m, which documents the parcel set in full.
%
%   INPUT
%     cfg.source                     Subject*_sLORETA_raw.mat, Destrieux 148 ROIs
%     <repo>/config/EEG_ROI_LABELS.csv   atlas label table (ROI_LABEL_FILE overrides)
%
%   OUTPUT (in cfg.out)
%     s12_cohort_level_null.txt   cohort summary
%     s12_cohort_level_null.mat   full null matrix and per-condition statistics
%
%   ENVIRONMENT
%     N_PERM     permutations per participant (default 1000)
%     N_WORKERS  parallel workers (default cfg.n_workers)
%     SUBJ_LIST  subset of participant indices, e.g. '1:5', for smoke tests

repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo, 'config'));
cfg = set_paths();

srcdir = fullfile(cfg.source, filesep);
outdir = fullfile(cfg.out, filesep);
labf = getenv('ROI_LABEL_FILE');
if isempty(labf), labf = fullfile(repo, 'config', 'EEG_ROI_LABELS.csv'); end
assert(isfile(labf), ['ROI label table not found: %s\n' ...
    'Set ROI_LABEL_FILE to point at EEG_ROI_LABELS.csv.'], labf);

fs = 250; pre_smp = 0.5*fs; post_smp = 1.5*fs; win_smp = round(0.6*fs);
n_perm    = env_num('N_PERM', 1000);
n_workers = env_num('N_WORKERS', cfg.n_workers);

% Fixed seed offset: participant ii draws its permutations from
% rng(SEED_BASE + ii), so the null distributions are reproducible.
SEED_BASE = 20260811;

% Occipital and ventral occipitotemporal parcel families, by Destrieux base
% name. Membership is an exact base-name match, never a substring match.
VISUAL_BASES = {'G_and_S_occipital_inf', 'G_cuneus', 'G_oc-temp_lat-fusifor', ...
        'G_oc-temp_med-Lingual', 'G_oc-temp_med-Parahip', 'G_occipital_middle', ...
        'G_occipital_sup', 'Pole_occipital', 'S_calcarine', 'S_collat_transv_ant', ...
        'S_collat_transv_post', 'S_oc-temp_lat', 'S_oc-temp_med_and_Lingual', ...
        'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'S_occipital_ant', ...
        'S_parieto_occipital'};

T = readtable(labf, 'TextType', 'string');
assert(height(T) == 148, 'label file has %d rows, expected 148', height(T));
[~, ord] = sort(T.eeg_idx);
names = T.eeg_name(ord);                          % 1..148 in source-data order
base_names = regexprep(names, '\s+[LR]$', '');    % strip the hemisphere suffix
is_visual = ismember(base_names, VISUAL_BASES);
vis_idx = find(is_visual)'; non_idx = find(~is_visual)';
assert(numel(vis_idx) == 34, 'expected 34 excluded parcels, got %d', numel(vis_idx));
assert(~any(contains(base_names(vis_idx), 'precuneus', 'IgnoreCase', true)), ...
       'G_precuneus is medial parietal cortex and must not be in the excluded set');

d = dir([srcdir 'Subject*_sLORETA_raw.mat']);
subs = {d.name};
if ~isempty(getenv('SUBJ_LIST'))
    k = str2num(getenv('SUBJ_LIST')); %#ok<ST2NM>
    subs = subs(k(k >= 1 & k <= numel(subs)));
end
n = numel(subs);

fprintf('=== Cohort-mean permutation null ===\n');
fprintf('excluded %d parcels, retained %d\n', numel(vis_idx), numel(non_idx));
fprintf('participants: %d   permutations: %d   workers: %d   seed base: %d\n\n', ...
        n, n_perm, n_workers, SEED_BASE);

if ~isempty(ver('parallel'))
    if isempty(gcp('nocreate')), parpool('Processes', n_workers); end
end

conds = {'all', 'non_visual'};
sel = {1:148, non_idx};
acc = nan(n, 2); nullall = nan(n, 2, n_perm); label_ok = false(n, 1);

t0 = tic;
parfor ii = 1:n
    rng(SEED_BASE + ii, 'twister');
    S = load([srcdir subs{ii}], 'condition_data', 'condition_data_type', 'roiindex');
    cd_ = S.condition_data; ct = S.condition_data_type;
    keep = ~cellfun(@isempty, cd_); cd_ = cd_(keep); ct = ct(keep);
    nt = numel(cd_);

    % ROI order is verified, not assumed: the parcel labels stored with the
    % source data are compared row by row against the atlas label table.
    ri = string(S.roiindex(:, 2));
    label_ok(ii) = numel(ri) == 148 && isequal(strtrim(ri), strtrim(names));

    % Trial labels, e.g. 'C 1_u_1_b_2': character 3 is the phrase identity and
    % the last character is the block. Covert blocks are 2, 4, 6, 8 and 10, so
    % the last character (2, 4, 6, 8, 0) still identifies the block uniquely.
    y = zeros(nt, 1); b = zeros(nt, 1);
    for t = 1:nt
        y(t) = str2double(ct{t}(3));
        b(t) = str2double(ct{t}(end));
    end

    F = zeros(nt, 148, win_smp);
    for t = 1:nt
        nd = cd_{t};
        bs = mean(nd(1:148, 1:pre_smp), 2);
        ev = nd(1:148, (pre_smp+1):(pre_smp+post_smp)) - bs;
        F(t, :, :) = ev(:, 1:win_smp);
    end
    ok = y >= 1 & y <= 5;
    y = y(ok); b = b(ok); F = F(ok, :, :);

    a_i = nan(1, 2); nl_i = nan(2, n_perm);
    for s = 1:2
        X = reshape(F(:, sel{s}, :), size(F, 1), []);
        X = zscore(X);
        [U, Sv, ~] = svd(X, 'econ'); Xr = U * Sv;   % lossless for a linear SVM
        a_i(s) = lobo(Xr, y, b);
        ub = unique(b);
        for q = 1:n_perm
            yp = y;
            for k = 1:numel(ub)
                m = find(b == ub(k));
                yp(m) = y(m(randperm(numel(m))));    % within-block shuffle
            end
            nl_i(s, q) = lobo(Xr, yp, b);
        end
    end
    acc(ii, :) = a_i; nullall(ii, :, :) = nl_i;
    fprintf('%-28s labels %d | all %.2f%% | non-visual %.2f%%\n', ...
            subs{ii}, label_ok(ii), a_i(1)*100, a_i(2)*100);
end
fprintf('\nelapsed %.1f min\n', toc(t0)/60);

% ---- cohort-level quantities, computed from the full permutation matrix ----
L = {sprintf('Cohort-mean null, N = %d, %d permutations, seed base %d', n, n_perm, SEED_BASE)
     'window 0-600 ms; conditions: all (148 parcels), non_visual (114 parcels)'
     sprintf('ROI order verified for %d / %d participants', sum(label_ok), n)
     ''};
res = struct();
for s = 1:2
    obs = mean(acc(:, s));
    cm = squeeze(mean(nullall(:, s, :), 1));      % n_perm x 1 cohort means
    ci = prctile(cm, [2.5 97.5]);
    p95 = prctile(cm, 95);
    pv = (1 + sum(cm >= obs)) / (n_perm + 1);
    res.(conds{s}) = struct('acc', obs, 'centre', mean(cm), 'ci', ci, 'p95', p95, ...
                            'max', max(cm), 'p', pv, 'sd', std(cm));
    L{end+1} = sprintf(['%-11s acc %.2f%% | null centre %.2f%% (SD %.2f) | ' ...
                        'central 95%% [%.2f%%, %.2f%%] | 95th pct %.2f%% | max %.2f%% | p %.3g'], ...
                       conds{s}, obs*100, mean(cm)*100, std(cm)*100, ci(1)*100, ci(2)*100, ...
                       p95*100, max(cm)*100, pv); %#ok<AGROW>
end
L{end+1} = '';
L{end+1} = 'Reproduction gate: the ''all'' row must return the published 27.26%.';

fid = fopen([outdir 's12_cohort_level_null.txt'], 'w');
for k = 1:numel(L), fprintf(fid, '%s\n', L{k}); fprintf('%s\n', L{k}); end
fclose(fid);
save([outdir 's12_cohort_level_null.mat'], 'acc', 'nullall', 'conds', ...
     'vis_idx', 'non_idx', 'names', 'label_ok', 'SEED_BASE', 'n_perm', 'res', '-v7.3');
fprintf('\nwrote %s\nDONE\n', outdir);
end


%% ---------------- helpers ----------------
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


function v = env_num(name, default_value)
%ENV_NUM  Numeric environment-variable override, or the supplied default.
s = getenv(name);
if isempty(s), v = default_value; else, v = str2double(s); end
end
