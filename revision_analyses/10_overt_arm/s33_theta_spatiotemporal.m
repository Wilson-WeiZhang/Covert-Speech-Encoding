%% =========================================================================
%  s33_theta_spatiotemporal.m
%
%  Theta-band version of the spatiotemporal comparison in s31: covert versus
%  overt repeated measures ANOVA over 148 ROIs x 30 windows, BH-FDR within
%  condition, plus the correspondence statistics between the two maps
%  (shared significant pairs with a Fisher exact test, and the Spearman
%  correlation of the per-ROI peak F over the planning period).
%
%  The analysis is identical to s31; only the input trees change:
%    covert  <data_root>/sourcedata_theta        (part of the distribution)
%    overt   <cfg.out>/sourcedata_theta_overt    (written by s32)
%  Both store the filtered data in condition_data_save.
%
%  INPUT   <data_root>/sourcedata_theta
%          <cfg.out>/sourcedata_theta_overt
%          assets/EEG_ROI_LABELS.csv
%  OUTPUT  <cfg.out>/overt_analysis/
%            s33_rmanova_theta_<cond>.mat, s33_sig_pairs_theta_<cond>.csv,
%            s33_window_counts_theta.csv, s33_summary_theta.txt
% =========================================================================

clearvars; clc;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(here), 'config'));
cfg = set_paths();

fprintf('=== s33: theta spatiotemporal comparison ===\n%s\n\n', ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));

outdir = fullfile(cfg.out, 'overt_analysis', filesep);
if ~isfolder(outdir), mkdir(outdir); end

theta_covert = fullfile(cfg.data_root, 'sourcedata_theta', filesep);
theta_overt  = fullfile(cfg.out, 'sourcedata_theta_overt', filesep);
assert(isfolder(theta_covert), ...
    'covert theta tree not found at %s', theta_covert);
assert(isfolder(theta_overt), ...
    'overt theta tree not found at %s (run s32_bandpass_theta first)', theta_overt);

CONDS = {'covert', theta_covert; 'overt', theta_overt};

fs               = 250;
baseline_samples = 125;
num_rois         = 148;
num_words        = 5;
num_windows      = 30;
win_ms           = 50;
win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends   = baseline_samples + round((1:num_windows)   * win_ms * fs / 1000);

roi_table = readtable(fullfile(here, 'assets', 'EEG_ROI_LABELS.csv'));
roi_names = roi_table.eeg_name;

WithinDesign = table((1:num_words)', 'VariableNames', {'Word'});
WithinDesign.Word = categorical(WithinDesign.Word);

R = struct();

for c = 1:size(CONDS, 1)
    cond = CONDS{c, 1};
    src  = CONDS{c, 2};
    fprintf('\n================ %s (theta) ================\n', upper(cond));

    files = dir(fullfile(src, 'Subject*_sLORETA_raw.mat'));
    ns = numel(files);
    fprintf('%d subjects in %s\n', ns, src);

    activity_data = zeros(ns, num_words, num_windows, num_rois);
    trial_counts  = zeros(ns, num_words);

    tic;
    for s = 1:ns
        S = load(fullfile(src, files(s).name));
        if isfield(S, 'condition_data'), cd_all = S.condition_data;
        else,                            cd_all = S.condition_data_save; end   % band trees
        keep = ~cellfun(@isempty, cd_all);
        cd_ = cd_all(keep);
        ct  = S.condition_data_type(keep);

        word_accum = zeros(num_words, num_windows, num_rois);
        word_count = zeros(num_words, 1);

        for t = 1:numel(cd_)
            parts = strsplit(ct{t}, ' ');
            w = str2double(parts{2}(1));
            if isnan(w) || w < 1 || w > 5, continue; end

            td = cd_{t};
            td = td - mean(td(:, 1:baseline_samples), 2);

            for win = 1:num_windows
                if win_ends(win) <= size(td, 2)
                    a = mean(td(:, win_starts(win):win_ends(win)), 2);
                    word_accum(w, win, :) = squeeze(word_accum(w, win, :)) + a;
                end
            end
            word_count(w) = word_count(w) + 1;
        end

        for w = 1:num_words
            if word_count(w) > 0
                activity_data(s, w, :, :) = word_accum(w, :, :) / word_count(w);
            end
        end
        trial_counts(s, :) = word_count';
        if mod(s, 10) == 0, fprintf('  loaded %d/%d\n', s, ns); end
    end
    fprintf('load and average: %.0f s | mean trials per phrase %.1f\n', toc, mean(trial_counts(:)));

    F_values = zeros(num_rois, num_windows);
    p_values = zeros(num_rois, num_windows);
    tic;
    for roi = 1:num_rois
        for win = 1:num_windows
            Y = squeeze(activity_data(:, :, win, roi));
            if std(Y(:)) ~= 0
                T  = array2table(Y, 'VariableNames', {'W1','W2','W3','W4','W5'});
                rm = fitrm(T, 'W1-W5 ~ 1', 'WithinDesign', WithinDesign);
                tb = ranova(rm);
                F_values(roi, win) = tb.F(1);
                p_values(roi, win) = tb.pValue(1);
            else
                F_values(roi, win) = NaN;
                p_values(roi, win) = NaN;
            end
        end
        if mod(roi, 40) == 0, fprintf('  rmANOVA ROI %d/%d\n', roi, num_rois); end
    end
    fprintf('rmANOVA: %.0f s\n', toc);

    vi = ~isnan(p_values);
    q_values = nan(size(p_values));
    q_values(vi) = mafdr(p_values(vi), 'BHFDR', true);
    sig = q_values < 0.05;

    fprintf('%s theta: %d significant ROI-window pairs, %d unique ROIs\n', ...
        cond, sum(sig(:)), numel(unique(find(any(sig, 2)))));

    save([outdir 's33_rmanova_theta_' cond '.mat'], 'activity_data', 'trial_counts', ...
        'F_values', 'p_values', 'q_values', 'sig', '-v7.3');

    [sr, sw] = find(sig);
    Fs = arrayfun(@(i) F_values(sr(i), sw(i)), (1:numel(sr))');
    qs = arrayfun(@(i) q_values(sr(i), sw(i)), (1:numel(sr))');
    [~, ord] = sort(Fs, 'descend');
    fid = fopen([outdir 's33_sig_pairs_theta_' cond '.csv'], 'w');
    fprintf(fid, 'Rank,ROI,Hemisphere,Window_ms,F,q\n');
    for i = 1:numel(ord)
        k = ord(i);
        nm = roi_names{sr(k)};
        fprintf(fid, '%d,%s,%s,%d-%d,%.2f,%.3f\n', i, strtrim(nm(1:end-1)), nm(end), ...
            (sw(k)-1)*win_ms, sw(k)*win_ms, Fs(k), qs(k));
    end
    fclose(fid);

    R.(cond) = struct('F', F_values, 'p', p_values, 'q', q_values, 'sig', sig, ...
                      'ns', ns, 'trials', mean(trial_counts(:)));
end

%% ---------------- correspondence between the two maps ----------------
Fc = R.covert.F;  Fo = R.overt.F;
Sc = R.covert.sig; So = R.overt.sig;
ok = ~isnan(Fc) & ~isnan(Fo);

rho_all = corr(Fc(ok), Fo(ok), 'type', 'Spearman');

% per-ROI peak F over the planning period (0-600 ms = windows 1:12)
peak_c = max(Fc(:, 1:12), [], 2);
peak_o = max(Fo(:, 1:12), [], 2);
[rho_peak, p_peak] = corr(peak_c, peak_o, 'type', 'Spearman');

n_c = sum(Sc(:)); n_o = sum(So(:)); n_both = sum(Sc(:) & So(:));
n_tot = sum(ok(:));
tbl = [n_both, n_c - n_both; n_o - n_both, n_tot - n_c - n_o + n_both];
[~, p_fisher] = fishertest(tbl);
enrich = (n_both / n_c) / (n_o / n_tot);

cnt = [sum(Sc, 1); sum(So, 1)];
fid = fopen([outdir 's33_window_counts_theta.csv'], 'w');
fprintf(fid, 'Window_ms,covert_sig_ROIs,overt_sig_ROIs\n');
for win = 1:num_windows
    fprintf(fid, '%d-%d,%d,%d\n', (win-1)*win_ms, win*win_ms, cnt(1,win), cnt(2,win));
end
fclose(fid);

fid = fopen([outdir 's33_summary_theta.txt'], 'w');
fp = @(varargin) fprintf(fid, varargin{:});
fp('theta spatiotemporal rmANOVA -- 148 ROIs x 30 windows (50 ms, 0-1500 ms)\n');
fp('procedure identical to s31; band = theta 4-8 Hz; FDR within condition\n\n');
fp('COUNTS\n');
fp('  covert theta: %d pairs, %d unique ROIs, N = %d, mean %.1f trials/phrase\n', ...
    n_c, numel(unique(find(any(Sc,2)))), R.covert.ns, R.covert.trials);
fp('  overt  theta: %d pairs, %d unique ROIs, N = %d, mean %.1f trials/phrase\n\n', ...
    n_o, numel(unique(find(any(So,2)))), R.overt.ns, R.overt.trials);
fp('CORRESPONDENCE (theta)\n');
fp('  pairs significant in both        : %d of %d covert pairs (enrichment %.1f-fold, Fisher p = %.3g)\n', ...
    n_both, n_c, enrich, p_fisher);
fp('  Spearman rho, full F maps        : %.3f\n', rho_all);
fp('  Spearman rho, per-ROI peak F 0-600 ms : %.3f (p = %.3g)\n\n', rho_peak, p_peak);
fp('SIGNIFICANT ROIs PER WINDOW (covert / overt)\n');
for win = 1:num_windows
    if cnt(1,win) > 0 || cnt(2,win) > 0
        fp('  %4d-%4d ms   %3d / %3d\n', (win-1)*win_ms, win*win_ms, cnt(1,win), cnt(2,win));
    end
end
fclose(fid);

type([outdir 's33_summary_theta.txt']);
fprintf('\nDONE -> %s\n', outdir);
