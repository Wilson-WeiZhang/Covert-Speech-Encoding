function s15_ocular_roi_decoding()
%S15_OCULAR_ROI_DECODING  Phrase decoding from orbital / frontopolar parcels alone.
%
%   Decoding counterpart of s14_ocular_spatial_enrichment.py, and the exact
%   analogue of s13_exclude_visual_rois.m for the ocular question.
%
%   Ocular artefact projects maximally onto the cortex directly above the
%   orbits. If residual ocular leakage drove the phrase discrimination observed
%   in source space, the 18 parcels nearest the orbits should themselves decode
%   phrase identity. This script tests that with the classifier rather than with
%   the group-level significance map.
%
%   THREE CONDITIONS, same participants, same 0-600 ms window, same pipeline as
%   the published classifier: all 148 parcels, the 130 remaining parcels, and
%   the 18 ocular-proximal parcels. Each condition is compared against ITS OWN
%   within-block permutation null, because the three differ in feature count and
%   accuracies are not comparable across them.
%
%   A size-matched control is not required for the interpretation here. With
%   p >> n a smaller parcel set is not intrinsically less accurate, so a low
%   accuracy obtained from an 18-parcel set cannot be attributed to its
%   dimensionality: that bias runs against the finding rather than for it.
%
%   OCULAR-PROXIMAL SET (9 bilateral families = 18 parcels), by Destrieux base
%   name: G_and_S_frontomargin, G_and_S_transv_frontopol, G_front_inf-Orbital,
%   G_orbital, G_rectus, S_orbital-H_Shaped, S_orbital_lateral,
%   S_orbital_med-olfact, S_suborbital. Membership is decided by an exact
%   base-name match, never by a substring match; the set size is asserted, and
%   dorsal frontal parcels are asserted to be absent, since including cortex far
%   from the orbits would blunt the test.
%
%   ROI order is verified, not assumed: the parcel labels stored with each
%   participant's source data are compared row by row against the atlas label
%   table.
%
%   INPUT
%     cfg.source                     Subject*_sLORETA_raw.mat, Destrieux 148 ROIs
%     <repo>/config/EEG_ROI_LABELS.csv   atlas label table (ROI_LABEL_FILE overrides)
%
%   OUTPUT (in cfg.out)
%     s15_ocular_roi_decoding.csv           one row per participant
%     s15_ocular_roi_decoding_summary.txt   cohort summary
%     s15_ocular_roi_decoding.mat           accuracies and null summaries
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

% Orbital and frontopolar surface: the cortex directly above the orbits, where
% ocular artefact projects most strongly.
OCULAR_BASES = {'G_and_S_frontomargin', 'G_and_S_transv_frontopol', 'G_front_inf-Orbital', ...
        'G_orbital', 'G_rectus', 'S_orbital-H_Shaped', 'S_orbital_lateral', ...
        'S_orbital_med-olfact', 'S_suborbital'};

% ---- label table ----
T = readtable(labf, 'TextType', 'string');
assert(height(T) == 148, 'label file has %d rows, expected 148', height(T));
[~, ord] = sort(T.eeg_idx);
names = T.eeg_name(ord);                          % 1..148 in source-data order
base_names = regexprep(names, '\s+[LR]$', '');    % strip the hemisphere suffix
is_ocular = ismember(base_names, OCULAR_BASES);
ocu_idx = find(is_ocular)'; non_idx = find(~is_ocular)';
fprintf('=== Phrase decoding from ocular-proximal parcels ===\n');
fprintf('ocular-proximal %d parcels (%d name families), remaining %d\n', ...
        numel(ocu_idx), numel(OCULAR_BASES), numel(non_idx));
assert(numel(ocu_idx) == 18, 'expected 18 ocular-proximal parcels, got %d', numel(ocu_idx));
% Dorsal frontal parcels are far from the orbits and must not be in the set;
% every membership is asserted explicitly because the set is defined by exact
% base-name match.
for bb = {'G_front_sup', 'G_front_middle', 'S_front_sup', 'G_precentral'}
    assert(~any(strcmp(base_names(ocu_idx), bb{1})), '%s leaked into the set', bb{1});
end

d = dir([srcdir 'Subject*_sLORETA_raw.mat']);
subs = {d.name};
if ~isempty(getenv('SUBJ_LIST'))
    k = str2num(getenv('SUBJ_LIST')); %#ok<ST2NM>
    subs = subs(k(k >= 1 & k <= numel(subs)));
end
n = numel(subs);
fprintf('participants: %d   permutations: %d   workers: %d\n\n', n, n_perm, n_workers);

if isempty(gcp('nocreate')), parpool('Processes', n_workers); end

conds = {'all', 'non_ocular', 'ocular'};
sel = {1:148, non_idx, ocu_idx};
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
    fprintf('%-28s labels %d | all %.1f%% | non-ocular %.1f%% | ocular-proximal %.1f%%\n', ...
            subs{ii}, label_ok(ii), a_i(1)*100, a_i(2)*100, a_i(3)*100);
end

fprintf('\n--- ROI order verification ---\n');
fprintf('participants whose roiindex matches the label table: %d / %d\n', sum(label_ok), n);
if ~all(label_ok)
    fprintf('*** ROI order not verified: the parcel selection may be wrong ***\n');
end

L = {sprintf(['Phrase decoding from ocular-proximal (orbital + frontopolar) parcels, ' ...
              'N = %d, %d permutations'], n, n_perm)
     sprintf('ocular-proximal %d parcels, remaining %d; window 0-600 ms', ...
             numel(ocu_idx), numel(non_idx))
     ''
     sprintf('%-12s %8s %9s %9s %10s %s', 'condition', 'nROI', 'acc', 'null', 'null95', 'p / individually sig')};
for s = 1:3
    L{end+1} = sprintf('%-12s %8d %8.2f%% %8.2f%% %9.2f%% %.3g / %d of %d', conds{s}, ...
        numel(sel{s}), mean(acc(:,s))*100, mean(nullc(:,s))*100, mean(null95(:,s))*100, ...
        signrank_safe(acc(:,s), nullc(:,s)), sum(pperm(:,s) < 0.05), n); %#ok<AGROW>
end
[~, pd, ~, sd] = ttest(acc(:,1), acc(:,2));
L{end+1} = '';
L{end+1} = sprintf('all vs non_ocular (paired): %.2f pp, t(%d) = %.2f, p = %.4g', ...
                   (mean(acc(:,1)) - mean(acc(:,2)))*100, sd.df, sd.tstat, pd);
L{end+1} = sprintf('ocular-proximal parcels: %s', strjoin(cellstr(names(ocu_idx))', ', '));

fid = fopen([outdir 's15_ocular_roi_decoding_summary.txt'], 'w');
for k = 1:numel(L), fprintf(fid, '%s\n', L{k}); fprintf('%s\n', L{k}); end
fclose(fid);

Tout = table(string(subs(:)), label_ok, acc(:,1)*100, acc(:,2)*100, acc(:,3)*100, ...
    nullc(:,2)*100, null95(:,2)*100, pperm(:,2), pperm(:,3), ...
    'VariableNames', {'file', 'roi_order_verified', 'acc_all', 'acc_non_ocular', 'acc_ocular', ...
                      'null_non_ocular', 'null95_non_ocular', 'p_non_ocular', 'p_ocular'});
writetable(Tout, [outdir 's15_ocular_roi_decoding.csv']);
save([outdir 's15_ocular_roi_decoding.mat'], 'acc', 'nullc', 'null95', 'pperm', ...
     'conds', 'ocu_idx', 'non_idx', 'names', '-v7');
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
