%% =========================================================================
%  s30_overt_vs_covert_classification.m
%
%  Phrase decoding accuracy in the overt and the covert blocks of the same
%  participants, computed with one and the same classifier so that the two
%  values are comparable by construction:
%
%    features   148 ROIs x N samples, each ROI minus its own 0.5 s pre-onset
%               baseline mean, then column-wise z-scored across trials
%    CV         leave-one-block-out over the five blocks of the condition
%    model      fitcecoc, linear SVM, one-vs-all
%
%  The covert arm is recomputed here rather than read from the main
%  analysis: this reproduces the published covert accuracy from the same
%  code path and yields a per-subject covert value to pair with the overt
%  one.
%
%  WINDOWS
%    0-600 ms    the window of the main analysis
%    200-400 ms  the interval of peak phrase discrimination
%
%  The baseline mean is subtracted from the full ROI time course BEFORE the
%  analysis window is cut, so that a shifted window still has the same
%  per-trial offset removed. Reversing the two operations changes every
%  accuracy.
%
%  ENVIRONMENT (optional)
%    WINDOWS_MS   window list, e.g. "0 600;200 400"
%
%  INPUT   cfg.source        covert source data
%          cfg.source_overt  overt source data
%  OUTPUT  <cfg.out>/overt_analysis/s30_classification_<win>.csv
%          <cfg.out>/overt_analysis/s30_classification_<win>_summary.txt
% =========================================================================

clearvars; clc;

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

src_cov = fullfile(cfg.source, filesep);
src_ovt = fullfile(cfg.source_overt, filesep);
outdir  = fullfile(cfg.out, 'overt_analysis', filesep);
if ~isfolder(outdir), mkdir(outdir); end

fs        = 250;
pre_smp   = 0.5 * fs;              % 125 baseline samples
post_smp  = 1.5 * fs;              % 375 post-onset samples
num_words = 5;
roilist   = 1:148;

wins = [0 600; 200 400];
if ~isempty(getenv('WINDOWS_MS'))
    wins = str2num(getenv('WINDOWS_MS')); %#ok<ST2NM>
end

% subjects present in both datasets, keyed by the Subject## in the filename
fc = dir([src_cov 'Subject*_sLORETA_raw.mat']);
fo = dir([src_ovt 'Subject*_sLORETA_raw.mat']);
sc = cellfun(@(x) x(1:strfind(x, '_sLORETA')-1), {fc.name}, 'UniformOutput', false);
so = cellfun(@(x) x(1:strfind(x, '_sLORETA')-1), {fo.name}, 'UniformOutput', false);
subjects = intersect(sc, so, 'stable');
n = numel(subjects);
fprintf('covert %d, overt %d, paired %d subjects\n', numel(sc), numel(so), n);
assert(n >= 50, 'only %d paired subjects -- check the two source directories', n);

if isempty(gcp('nocreate')), parpool('Processes', cfg.n_workers); end

for wi = 1:size(wins, 1)
    w0 = wins(wi, 1); w1 = wins(wi, 2);
    s0 = round(w0 / 1000 * fs) + 1;          % 0 ms -> first post-onset sample
    s1 = round(w1 / 1000 * fs);
    nsmp = s1 - s0 + 1;
    fprintf('\n===== window %d-%d ms : samples %d..%d (%d per ROI, %d features) =====\n', ...
            w0, w1, s0, s1, nsmp, numel(roilist) * nsmp);

    acc_cov = nan(n, 1); acc_ovt = nan(n, 1);
    ntr_cov = nan(n, 1); ntr_ovt = nan(n, 1);

    parfor ii = 1:n
        subj = subjects{ii};
        [acc_cov(ii), ntr_cov(ii)] = classify_one([src_cov subj '_sLORETA_raw.mat'], ...
                                                  s0, s1, pre_smp, post_smp, roilist);
        [acc_ovt(ii), ntr_ovt(ii)] = classify_one([src_ovt subj '_sLORETA_raw.mat'], ...
                                                  s0, s1, pre_smp, post_smp, roilist);
        fprintf('%-10s covert %.2f%%  overt %.2f%%\n', subj, acc_cov(ii)*100, acc_ovt(ii)*100);
    end

    % ---- paired comparison ----
    d = acc_ovt - acc_cov;
    [~, p, ~, st] = ttest(acc_ovt, acc_cov);
    dz = mean(d) / std(d);
    [~, p_cov] = ttest(acc_cov, 1/num_words);
    [~, p_ovt] = ttest(acc_ovt, 1/num_words);

    T = table(subjects(:), ntr_cov, acc_cov*100, ntr_ovt, acc_ovt*100, d*100, ...
        'VariableNames', {'subject','n_trials_covert','acc_covert_pct', ...
                          'n_trials_overt','acc_overt_pct','diff_pct'});
    csv = sprintf('%ss30_classification_%d-%dms.csv', outdir, w0, w1);
    writetable(T, csv);

    lines = {
        sprintf('window %d-%d ms   %d ROIs x %d samples = %d features   N = %d', ...
                w0, w1, numel(roilist), nsmp, numel(roilist)*nsmp, n)
        sprintf('covert : %.2f%% +/- %.2f%%   (range %.2f-%.2f)   vs 20%%: t(%d) = %.2f, p = %.3g', ...
                mean(acc_cov)*100, std(acc_cov)*100, min(acc_cov)*100, max(acc_cov)*100, n-1, ...
                (mean(acc_cov)-0.2)/(std(acc_cov)/sqrt(n)), p_cov)
        sprintf('overt  : %.2f%% +/- %.2f%%   (range %.2f-%.2f)   vs 20%%: t(%d) = %.2f, p = %.3g', ...
                mean(acc_ovt)*100, std(acc_ovt)*100, min(acc_ovt)*100, max(acc_ovt)*100, n-1, ...
                (mean(acc_ovt)-0.2)/(std(acc_ovt)/sqrt(n)), p_ovt)
        sprintf('overt - covert : %+.2f pp   paired t(%d) = %.3f, p = %.4g, dz = %.3f', ...
                mean(d)*100, st.df, st.tstat, p, dz)
        sprintf('overt higher in %d/%d subjects', sum(d > 0), n)
        };
    fid = fopen(sprintf('%ss30_classification_%d-%dms_summary.txt', outdir, w0, w1), 'w');
    for k = 1:numel(lines), fprintf(fid, '%s\n', lines{k}); fprintf('%s\n', lines{k}); end
    fclose(fid);

    fprintf('wrote %s\n', csv);
end

fprintf('\nDONE\n');

%% ---------------- helpers ----------------
function [acc, n_valid] = classify_one(matfile, s0, s1, pre_smp, post_smp, roilist)
%CLASSIFY_ONE  Per-subject decoding for one analysis window.
S = load(matfile, 'condition_data', 'condition_data_type');
cd_ = S.condition_data; ct = S.condition_data_type;
keep = ~cellfun(@isempty, cd_);
cd_ = cd_(keep); ct = ct(keep);
nt = numel(cd_);

% Labels are of the form 'O 2_u_1_b_1': phrase after the condition letter,
% block number in the '_b_<n>' suffix.
word_labels  = zeros(nt, 1);
block_labels = zeros(nt, 1);
for t = 1:nt
    word_labels(t) = str2double(ct{t}(3));
    bm = regexp(ct{t}, '_b_(\d+)', 'tokens', 'once');
    if isempty(bm)
        block_labels(t) = NaN;
    else
        block_labels(t) = str2double(bm{1});
    end
end

nroi = numel(roilist);
nsmp = s1 - s0 + 1;
X = zeros(nt, nroi * nsmp);
for t = 1:nt
    nd = cd_{t};
    base = mean(nd(roilist, 1:pre_smp), 2);                       % per-ROI baseline mean
    ev   = nd(roilist, (pre_smp+1):(pre_smp+post_smp)) - base;    % baseline-corrected
    seg  = ev(:, s0:s1);                                          % then cut the window
    X(t, :) = reshape(seg.', 1, []);                              % ROI-major
end

valid = word_labels >= 1 & word_labels <= 5 & ~isnan(word_labels) & ~isnan(block_labels);
X = zscore(X(valid, :));
y = word_labels(valid);
b = block_labels(valid);
n_valid = sum(valid);

pred = zeros(n_valid, 1);
ub = unique(b);
for k = 1:numel(ub)
    tr = find(b ~= ub(k));
    te = find(b == ub(k));
    if isempty(tr) || isempty(te), continue, end
    mdl = fitcecoc(X(tr, :), y(tr), ...
        'Learners', templateSVM('KernelFunction', 'linear'), ...
        'Coding', 'onevsall', 'verbose', 0);
    pred(te) = predict(mdl, X(te, :));
end
acc = sum(pred == y) / n_valid;
end
