%% =========================================================================
%  s31_rmanova_spatiotemporal.m
%
%  Spatiotemporal map of the phrase effect for both conditions: a repeated
%  measures ANOVA on the phrase factor for each of 148 ROIs x 30 windows,
%  BH-FDR corrected, computed separately for the covert and the overt blocks.
%
%  The procedure is the one behind the main spatiotemporal figure: per
%  subject and phrase the trials are averaged, each trial is corrected by its
%  own 0.5 s pre-onset baseline, activity is averaged in 50 ms windows tiling
%  0-1500 ms, and each ROI-window cell is tested with fitrm/ranova. The
%  covert condition is recomputed here rather than reused, so that the
%  reproduction can be checked against the published count of significant
%  ROI-window pairs (53) and so that both F maps come from a single run.
%
%  FDR is applied within each condition. The two conditions are not pooled:
%  a joint correction would change the published covert numbers.
%
%  INPUT   cfg.source        covert source data
%          cfg.source_overt  overt source data
%          assets/EEG_ROI_LABELS.csv
%  OUTPUT  <cfg.out>/overt_analysis/
%            s31_rmanova_<cond>.mat     F, p, q matrices and activity_data
%            s31_sig_pairs_<cond>.csv   significant ROI-window pairs, ranked
%            s31_window_counts.csv      significant ROI count per window
%            s31_summary.txt            reproduction check and comparison
% =========================================================================

clearvars; clc;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(here), 'config'));
cfg = set_paths();

outdir = fullfile(cfg.out, 'overt_analysis', filesep);
if ~isfolder(outdir), mkdir(outdir); end

CONDS = { ...
    'covert', fullfile(cfg.source, filesep); ...
    'overt',  fullfile(cfg.source_overt, filesep)};

% --- parameters ---------------------------------------------------------
fs               = 250;
baseline_samples = 125;      % 0.5 s pre-onset
num_rois         = 148;
num_words        = 5;
num_windows      = 30;       % 0-1500 ms
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
    fprintf('\n================ %s ================\n', upper(cond));

    files = dir(fullfile(src, 'Subject*_sLORETA_raw.mat'));
    ns = numel(files);
    fprintf('%d subjects in %s\n', ns, src);

    activity_data = zeros(ns, num_words, num_windows, num_rois);
    trial_counts  = zeros(ns, num_words);

    % ---- Step 1: trial average per subject and phrase -------------------
    tic;
    for s = 1:ns
        S = load(fullfile(src, files(s).name), 'condition_data', 'condition_data_type');
        keep = ~cellfun(@isempty, S.condition_data);
        cd_ = S.condition_data(keep);
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

    % ---- Step 2: repeated measures ANOVA per ROI and window -------------
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

    % ---- Step 3: BH-FDR within condition --------------------------------
    vi = ~isnan(p_values);
    q_values = nan(size(p_values));
    q_values(vi) = mafdr(p_values(vi), 'BHFDR', true);
    sig = q_values < 0.05;

    fprintf('%s: %d significant ROI-window pairs, %d unique ROIs\n', ...
        cond, sum(sig(:)), numel(unique(find(any(sig, 2)))));

    % ---- save -----------------------------------------------------------
    save([outdir 's31_rmanova_' cond '.mat'], 'activity_data', 'trial_counts', ...
        'F_values', 'p_values', 'q_values', 'sig', '-v7.3');

    [sr, sw] = find(sig);
    Fs = arrayfun(@(i) F_values(sr(i), sw(i)), (1:numel(sr))');
    qs = arrayfun(@(i) q_values(sr(i), sw(i)), (1:numel(sr))');
    [~, ord] = sort(Fs, 'descend');
    fid = fopen([outdir 's31_sig_pairs_' cond '.csv'], 'w');
    fprintf(fid, 'Rank,ROI,Hemisphere,Window_ms,F,q\n');
    for i = 1:numel(ord)
        k = ord(i);
        nm = roi_names{sr(k)};
        hemi = nm(end);                       % names end in ' L' or ' R'
        fprintf(fid, '%d,%s,%s,%d-%d,%.2f,%.3f\n', i, strtrim(nm(1:end-1)), hemi, ...
            (sw(k)-1)*win_ms, sw(k)*win_ms, Fs(k), qs(k));
    end
    fclose(fid);

    R.(cond) = struct('F', F_values, 'p', p_values, 'q', q_values, 'sig', sig, ...
                      'ns', ns, 'trials', mean(trial_counts(:)));
end

%% ---------------- comparison ----------------
Fc = R.covert.F;  Fo = R.overt.F;
Sc = R.covert.sig; So = R.overt.sig;
ok = ~isnan(Fc) & ~isnan(Fo);

rho_F = corr(Fc(ok), Fo(ok), 'type', 'Spearman');

n_c    = sum(Sc(:));
n_o    = sum(So(:));
n_both = sum(Sc(:) & So(:));
% overlap expected by chance, given the two marginal counts over the same cells
exp_both = n_c * n_o / sum(ok(:));

cnt = [sum(Sc, 1); sum(So, 1)];
fid = fopen([outdir 's31_window_counts.csv'], 'w');
fprintf(fid, 'Window_ms,covert_sig_ROIs,overt_sig_ROIs\n');
for win = 1:num_windows
    fprintf(fid, '%d-%d,%d,%d\n', (win-1)*win_ms, win*win_ms, cnt(1,win), cnt(2,win));
end
fclose(fid);

fid = fopen([outdir 's31_summary.txt'], 'w');
fp = @(varargin) fprintf(fid, varargin{:});
fp('spatiotemporal rmANOVA -- 148 ROIs x 30 windows (50 ms, 0-1500 ms)\n');
fp('FDR applied within condition\n\n');
fp('REPRODUCTION CHECK (covert)\n');
fp('  significant pairs this run : %d\n', n_c);
fp('  published table rows       : 53\n');
fp('  match                      : %s\n\n', string(n_c == 53));
fp('COUNTS\n');
fp('  covert : %d pairs, %d unique ROIs, N = %d, mean %.1f trials/phrase\n', ...
    n_c, numel(unique(find(any(Sc,2)))), R.covert.ns, R.covert.trials);
fp('  overt  : %d pairs, %d unique ROIs, N = %d, mean %.1f trials/phrase\n\n', ...
    n_o, numel(unique(find(any(So,2)))), R.overt.ns, R.overt.trials);
fp('CORRESPONDENCE\n');
fp('  Spearman rho between the two F maps (%d cells) : %.3f\n', sum(ok(:)), rho_F);
fp('  pairs significant in both                      : %d (expected by chance %.1f)\n', ...
    n_both, exp_both);
fp('  significant in covert only                     : %d\n', n_c - n_both);
fp('  significant in overt only                      : %d\n\n', n_o - n_both);
fp('SIGNIFICANT ROIs PER WINDOW (covert / overt)\n');
for win = 1:num_windows
    if cnt(1,win) > 0 || cnt(2,win) > 0
        fp('  %4d-%4d ms   %3d / %3d\n', (win-1)*win_ms, win*win_ms, cnt(1,win), cnt(2,win));
    end
end
fclose(fid);

type([outdir 's31_summary.txt']);
fprintf('\nDONE -> %s\n', outdir);
