function s13_exclude_visual_rois()
%S13_EXCLUDE_VISUAL_ROIS  Phrase decoding after removing visual cortex parcels.
%
%   Asks whether phrase decoding survives when every occipital and ventral
%   occipitotemporal parcel is removed from the feature set. This is a control
%   that only source space allows: at the scalp there is no way to delete a
%   cortical region. It bounds how much of the discrimination can be carried by
%   low-level visual and orthographic processing of the written cue.
%
%   Scope of the inference: the analysis bounds the visual contribution and
%   nothing more. It does not separate reading from production planning, since
%   silent reading is also represented in the motor and auditory regions that
%   remain (Kunz et al., 2025).
%
%   THREE CONDITIONS, same participants, same 0-600 ms window, same pipeline as
%   the published classifier:
%     all         148 parcels - reproduction control, must return the published
%                 27.26% +/- 6.97%
%     non_visual  114 parcels - the surviving set
%     visual       34 parcels - the complement, which quantifies what the
%                 removed tissue carries
%
%   EXCLUDED SET (17 bilateral families = 34 parcels), by Destrieux base name:
%     G_and_S_occipital_inf, G_cuneus, G_oc-temp_lat-fusifor,
%     G_oc-temp_med-Lingual, G_oc-temp_med-Parahip, G_occipital_middle,
%     G_occipital_sup, Pole_occipital, S_calcarine, S_collat_transv_ant,
%     S_collat_transv_post, S_oc-temp_lat, S_oc-temp_med_and_Lingual,
%     S_oc_middle_and_Lunatus, S_oc_sup_and_transversal, S_occipital_ant,
%     S_parieto_occipital
%
%   Membership is decided by an exact base-name match, never by a substring
%   match, and the size of the resulting set is asserted. G_precuneus is
%   deliberately not part of the set: despite the shared string it is medial
%   parietal cortex, not visual cortex. Boundary structures that are plausibly
%   visual (parahippocampal gyrus, parieto-occipital sulcus, collateral sulcus)
%   are excluded, because over-excluding makes the surviving set a conservative
%   test.
%
%   ROI order is verified, not assumed: the parcel labels stored with each
%   participant's source data are compared row by row against the atlas label
%   table, so a parcel selection can never be applied to a differently ordered
%   atlas.
%
%   INPUT
%     cfg.source                     Subject*_sLORETA_raw.mat, Destrieux 148 ROIs
%     <repo>/config/EEG_ROI_LABELS.csv   atlas label table (ROI_LABEL_FILE overrides)
%
%   OUTPUT (in cfg.out)
%     s13_exclude_visual_rois.csv           one row per participant
%     s13_exclude_visual_rois_summary.txt   cohort summary
%     s13_exclude_visual_rois.mat           accuracies and null summaries
%
%   PARAMETERS
%     window 0-600 ms after phrase onset, baseline-corrected on the 500 ms
%     pre-onset interval, features z-scored across trials and reduced by an
%     economy SVD (lossless for a linear SVM); five-class linear SVM (one-vs-all
%     ECOC), leave-one-block-out cross-validation; within-block label
%     permutation, which preserves the fixed number of trials per phrase per
%     block imposed by the design.
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

VISUAL_BASES = {'G_and_S_occipital_inf', 'G_cuneus', 'G_oc-temp_lat-fusifor', ...
        'G_oc-temp_med-Lingual', 'G_oc-temp_med-Parahip', 'G_occipital_middle', ...
        'G_occipital_sup', 'Pole_occipital', 'S_calcarine', 'S_collat_transv_ant', ...
        'S_collat_transv_post', 'S_oc-temp_lat', 'S_oc-temp_med_and_Lingual', ...
        'S_oc_middle_and_Lunatus', 'S_oc_sup_and_transversal', 'S_occipital_ant', ...
        'S_parieto_occipital'};

% ---- label table ----
T = readtable(labf, 'TextType', 'string');
assert(height(T) == 148, 'label file has %d rows, expected 148', height(T));
[~, ord] = sort(T.eeg_idx);
names = T.eeg_name(ord);                          % 1..148 in source-data order
base_names = regexprep(names, '\s+[LR]$', '');    % strip the hemisphere suffix
is_visual = ismember(base_names, VISUAL_BASES);
vis_idx = find(is_visual)'; non_idx = find(~is_visual)';
fprintf('=== Phrase decoding without occipital / ventral occipitotemporal parcels ===\n');
fprintf('excluded %d parcels (%d name families), retained %d\n', ...
        numel(vis_idx), numel(VISUAL_BASES), numel(non_idx));
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
fprintf('participants: %d   permutations: %d   workers: %d\n\n', n, n_perm, n_workers);

if isempty(gcp('nocreate')), parpool('Processes', n_workers); end

conds = {'all', 'non_visual', 'visual'};
sel = {1:148, non_idx, vis_idx};
acc = nan(n, 3); nullc = nan(n, 3); null95 = nan(n, 3); pperm = nan(n, 3);
label_ok = false(n, 1);

parfor ii = 1:n
    S = load([srcdir subs{ii}], 'condition_data', 'condition_data_type', 'roiindex');
    cd_ = S.condition_data; ct = S.condition_data_type;
    keep = ~cellfun(@isempty, cd_); cd_ = cd_(keep); ct = ct(keep);
    nt = numel(cd_);

    % ---- verify ROI order against the label table ----
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

    % Build the full feature array once, then take parcel blocks per condition.
    F = zeros(nt, 148, win_smp);
    for t = 1:nt
        nd = cd_{t};
        bs = mean(nd(1:148, 1:pre_smp), 2);
        ev = nd(1:148, (pre_smp+1):(pre_smp+post_smp)) - bs;
        F(t, :, :) = ev(:, 1:win_smp);
    end
    ok = y >= 1 & y <= 5;
    y = y(ok); b = b(ok); F = F(ok, :, :);

    a_i = nan(1,3); c_i = nan(1,3); p95_i = nan(1,3); pp_i = nan(1,3);
    for s = 1:3
        rois = sel{s};
        X = reshape(F(:, rois, :), size(F, 1), []);
        X = zscore(X);
        [U, Sv, ~] = svd(X, 'econ'); Xr = U * Sv;   % lossless for a linear SVM
        a_i(s) = lobo(Xr, y, b);
        nul = zeros(1, n_perm); ub = unique(b);
        for q = 1:n_perm
            yp = y;
            for k = 1:numel(ub)
                m = find(b == ub(k));
                yp(m) = y(m(randperm(numel(m))));    % within-block shuffle
            end
            nul(q) = lobo(Xr, yp, b);
        end
        c_i(s) = mean(nul); p95_i(s) = prctile(nul, 95);
        pp_i(s) = (1 + sum(nul >= a_i(s))) / (n_perm + 1);
    end
    acc(ii,:) = a_i; nullc(ii,:) = c_i; null95(ii,:) = p95_i; pperm(ii,:) = pp_i;
    fprintf('%-28s labels %d | all %.1f%% | non-visual %.1f%% | visual %.1f%%\n', ...
            subs{ii}, label_ok(ii), a_i(1)*100, a_i(2)*100, a_i(3)*100);
end

fprintf('\n--- ROI order verification ---\n');
fprintf('participants whose roiindex matches the label table: %d / %d\n', sum(label_ok), n);
if ~all(label_ok)
    fprintf('*** ROI order not verified: the parcel selection may be wrong ***\n');
end

L = {sprintf(['Phrase decoding without occipital / ventral occipitotemporal parcels, ' ...
              'N = %d, %d permutations'], n, n_perm)
     sprintf('excluded %d parcels, retained %d; window 0-600 ms', ...
             numel(vis_idx), numel(non_idx))
     ''
     sprintf('%-12s %8s %9s %9s %10s %s', 'condition', 'nROI', 'acc', 'null', 'null95', 'p / individually sig')};
for s = 1:3
    L{end+1} = sprintf('%-12s %8d %8.2f%% %8.2f%% %9.2f%% %.3g / %d of %d', conds{s}, ...
        numel(sel{s}), mean(acc(:,s))*100, mean(nullc(:,s))*100, mean(null95(:,s))*100, ...
        signrank_safe(acc(:,s), nullc(:,s)), sum(pperm(:,s) < 0.05), n); %#ok<AGROW>
end
[~, pd, ~, sd] = ttest(acc(:,1), acc(:,2));
L{end+1} = '';
L{end+1} = sprintf('all vs non_visual (paired): %.2f pp, t(%d) = %.2f, p = %.4g', ...
                   (mean(acc(:,1)) - mean(acc(:,2)))*100, sd.df, sd.tstat, pd);
L{end+1} = sprintf('excluded parcels: %s', strjoin(cellstr(names(vis_idx))', ', '));

fid = fopen([outdir 's13_exclude_visual_rois_summary.txt'], 'w');
for k = 1:numel(L), fprintf(fid, '%s\n', L{k}); fprintf('%s\n', L{k}); end
fclose(fid);

Tout = table(string(subs(:)), label_ok, acc(:,1)*100, acc(:,2)*100, acc(:,3)*100, ...
    nullc(:,2)*100, null95(:,2)*100, pperm(:,2), pperm(:,3), ...
    'VariableNames', {'file', 'roi_order_verified', 'acc_all', 'acc_non_visual', 'acc_visual', ...
                      'null_non_visual', 'null95_non_visual', 'p_non_visual', 'p_visual'});
writetable(Tout, [outdir 's13_exclude_visual_rois.csv']);
save([outdir 's13_exclude_visual_rois.mat'], 'acc', 'nullc', 'null95', 'pperm', ...
     'conds', 'vis_idx', 'non_idx', 'names', '-v7');
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
