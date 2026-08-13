%% =========================================================================
%  s34_decoding_timecourse.m
%
%  Sliding-window phrase decoding for both conditions: instead of a single
%  analysis window, the classifier is retrained in each of 30 consecutive
%  50 ms windows tiling 0-1500 ms, giving an accuracy time course per
%  subject and condition. The windows match those of the spatiotemporal
%  rmANOVA in s31, so the two views of the same data are aligned in time.
%
%  The classifier is the one used throughout: per trial each ROI's 0.5 s
%  pre-onset baseline mean is subtracted BEFORE the window is cut, features
%  are 148 ROIs x window samples z-scored column-wise across trials,
%  cross-validation is leave-one-block-out over the five blocks of the
%  condition, and the model is fitcecoc with a linear SVM, one-vs-all.
%
%  INPUT   cfg.source        covert source data
%          cfg.source_overt  overt source data
%  OUTPUT  <cfg.out>/overt_analysis/s34_timecourse_<cond>.csv  subjects x 30 windows
%          <cfg.out>/overt_analysis/s34_summary.txt
% =========================================================================

clearvars; clc;

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

fprintf('=== s34: sliding-window decoding time course ===\n%s\n\n', ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));

outdir = fullfile(cfg.out, 'overt_analysis', filesep);
if ~isfolder(outdir), mkdir(outdir); end

CONDS = { ...
    'covert', fullfile(cfg.source, filesep); ...
    'overt',  fullfile(cfg.source_overt, filesep)};

fs               = 250;
baseline_samples = 125;
num_rois         = 148;
num_windows      = 30;
win_ms           = 50;
win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends   = baseline_samples + round((1:num_windows)   * win_ms * fs / 1000);

if isempty(gcp('nocreate')), parpool('Processes', cfg.n_workers); end

ACC = struct();

for c = 1:size(CONDS, 1)
    cond = CONDS{c, 1};
    src  = CONDS{c, 2};
    files = dir(fullfile(src, 'Subject*_sLORETA_raw.mat'));
    ns = numel(files);
    fprintf('\n================ %s: %d subjects ================\n', upper(cond), ns);

    acc = nan(ns, num_windows);
    subj_names = {files.name};

    tic;
    parfor s = 1:ns
        S = load(fullfile(src, files(s).name), 'condition_data', 'condition_data_type');
        keep = ~cellfun(@isempty, S.condition_data);
        cd_ = S.condition_data(keep);
        ct  = S.condition_data_type(keep);

        nt  = numel(cd_);
        y   = zeros(nt, 1);
        blk = zeros(nt, 1);
        ok  = true(nt, 1);
        for t = 1:nt
            parts = strsplit(ct{t}, ' ');
            w = str2double(parts{2}(1));
            bm = regexp(ct{t}, '_b_(\d+)', 'tokens', 'once');
            if isnan(w) || w < 1 || w > 5 || isempty(bm), ok(t) = false; continue; end
            y(t) = w; blk(t) = str2double(bm{1});
        end
        cd_ = cd_(ok); y = y(ok); blk = blk(ok); nt = numel(cd_);

        % baseline subtraction once per trial, before any window is cut
        for t = 1:nt
            cd_{t} = cd_{t} - mean(cd_{t}(:, 1:baseline_samples), 2);
        end
        blocks = unique(blk);

        acc_s = nan(1, num_windows);
        for win = 1:num_windows
            nsamp = win_ends(win) - win_starts(win) + 1;
            X = zeros(nt, num_rois * nsamp);
            for t = 1:nt
                seg = cd_{t}(:, win_starts(win):win_ends(win));
                X(t, :) = seg(:)';
            end
            X = zscore(X, 0, 1);                       % column-wise across trials

            correct = 0; total = 0;
            for b = 1:numel(blocks)
                te = blk == blocks(b);
                tr = ~te;
                if nnz(te) == 0 || numel(unique(y(tr))) < 5, continue; end
                mdl = fitcecoc(X(tr, :), y(tr), ...
                    'Learners', templateSVM('KernelFunction', 'linear'), ...
                    'Coding', 'onevsall');
                pred = predict(mdl, X(te, :));
                correct = correct + sum(pred == y(te));
                total   = total + nnz(te);
            end
            if total > 0, acc_s(win) = 100 * correct / total; end
        end
        acc(s, :) = acc_s;
        fprintf('[%2d/%d] %s mean %.1f%%\n', s, ns, files(s).name, mean(acc_s, 'omitnan'));
    end
    fprintf('%s: %.1f min\n', cond, toc/60);

    fid = fopen([outdir 's34_timecourse_' cond '.csv'], 'w');
    fprintf(fid, 'subject');
    for win = 1:num_windows, fprintf(fid, ',w%d_%d', (win-1)*win_ms, win*win_ms); end
    fprintf(fid, '\n');
    for s = 1:ns
        fprintf(fid, '%s', erase(subj_names{s}, '_sLORETA_raw.mat'));
        fprintf(fid, ',%.2f', acc(s, :));
        fprintf(fid, '\n');
    end
    fclose(fid);

    ACC.(cond) = acc;
end

%% ---------------- summary ----------------
fid = fopen([outdir 's34_summary.txt'], 'w');
fp = @(varargin) fprintf(fid, varargin{:});
fp('sliding-window decoding (30 x 50 ms windows, leave-one-block-out, linear SVM)\n\n');
for c = 1:size(CONDS, 1)
    cond = CONDS{c, 1};
    m = mean(ACC.(cond), 1, 'omitnan');
    [pk, pkw] = max(m);
    fp('%s: cohort-mean accuracy per window (chance 20%%)\n', upper(cond));
    for win = 1:num_windows
        fp('  %4d-%4d ms  %6.2f%%\n', (win-1)*win_ms, win*win_ms, m(win));
    end
    fp('  peak %.2f%% at %d-%d ms\n\n', pk, (pkw-1)*win_ms, pkw*win_ms);
end
fclose(fid);
type([outdir 's34_summary.txt']);
fprintf('\nDONE -> %s\n', outdir);
