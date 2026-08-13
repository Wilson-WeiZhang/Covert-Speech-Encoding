%% =========================================================================
%  s09_classify_component_variants.m
%
%  Run the phrase classification of the main analysis on the published source
%  data and on every component add-back variant, and compare each variant with
%  the published set.
%
%  INPUT
%    cfg.source                published source data (reference set)
%    cfg.addback(<variant>)    add-back datasets written by s04 and s05:
%                              lateye, blink, mus, lateye_union, blink_union
%
%  OUTPUT
%    cfg.out/accuracy_by_variant.csv   per-participant accuracy (%) for every
%                                      set, one column per set
%    printed summary and, for each variant, a paired t-test against the
%    published set
%
%  METHOD (identical to the main classification analysis)
%    Features   148 ROIs x 150 samples (0-600 ms after phrase onset), per-ROI
%               baseline (-0.5 to 0 s) subtracted, column-wise z-score
%    Labels     phrase identity 1-5, read from the trial label
%    Validation leave-one-block-out
%    Learner    linear SVM, one-vs-all
%
%    Speed-up: a linear SVM depends on the data only through the Gram matrix,
%    so the economy SVD X = U*S*V' can replace the feature matrix. Xr = U*S
%    satisfies Xr*Xr' = X*X' exactly, so the classification is unchanged while
%    the feature dimension drops from 22200 to the number of trials.
%
%  ENV
%    AB_SETS     comma-separated list of dataset directory names, resolved
%                under cfg.data_root, overriding the default six. The first
%                entry is used as the reference set.
%    AB_WORKERS  parallel pool size (default cfg.n_workers)
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

setsEnv = getenv('AB_SETS');
if isempty(setsEnv)
    setDirs = {cfg.source, ...
               cfg.addback('lateye'), ...
               cfg.addback('blink'), ...
               cfg.addback('mus'), ...
               cfg.addback('lateye_union'), ...
               cfg.addback('blink_union')};
else
    names = strtrim(strsplit(setsEnv, ','));
    setDirs = cellfun(@(nm) fullfile(cfg.data_root, nm), names, 'UniformOutput', false);
end
labels = cellfun(@dirName, setDirs, 'UniformOutput', false);

fs = 250;                    % sampling rate, Hz
pre_smp  = 0.5*fs;           % baseline, 0.5 s before phrase onset
post_smp = 1.5*fs;           % epoch after phrase onset
win_smp  = round(0.6*fs);    % classification window, 0-600 ms
nroi = 148;                  % Destrieux ROIs

nw = str2double(getenv('AB_WORKERS'));
if isnan(nw), nw = cfg.n_workers; end
if nw > 1 && isempty(gcp('nocreate')), parpool('Processes', nw); end

pub = dir(fullfile(cfg.source, 'Subject*_sLORETA_raw.mat'));
subjFiles = {pub.name};
n = numel(subjFiles);
fprintf('%d participants, %d sets\n\n', n, numel(setDirs));

ACC = nan(n, numel(setDirs));
for s = 1:numel(setDirs)
    dirPath = setDirs{s};
    if ~isfolder(dirPath)
        fprintf('MISSING %s\n', dirPath); continue
    end
    acc = nan(n,1);
    parfor i = 1:n
        f = fullfile(dirPath, subjFiles{i});
        if ~isfile(f), continue, end
        S = load(f, 'condition_data', 'condition_data_type');
        cd_ = S.condition_data; ct = S.condition_data_type;
        keep = ~cellfun(@isempty, cd_); cd_ = cd_(keep); ct = ct(keep);
        nt = numel(cd_);

        y = zeros(nt,1); b = zeros(nt,1);
        for t = 1:nt
            y(t) = str2double(ct{t}(3));      % phrase identity
            b(t) = str2double(ct{t}(end));    % block
        end
        X = zeros(nt, nroi*win_smp);
        for t = 1:nt
            nd = cd_{t};
            base = mean(nd(1:nroi, 1:pre_smp), 2);
            ev   = nd(1:nroi, (pre_smp+1):(pre_smp+post_smp)) - base;
            X(t,:) = reshape(ev(:, 1:win_smp).', 1, []);
        end
        ok = y >= 1 & y <= 5 & ~isnan(y);
        X = zscore(X(ok,:)); y = y(ok); b = b(ok);
        [U,Sv,~] = svd(X, 'econ');
        acc(i) = lobo_acc(U*Sv, y, b);
    end
    ACC(:,s) = acc;
    fprintf('%-46s %6.2f%% +/- %5.2f   (n=%d)\n', labels{s}, ...
        mean(acc, 'omitnan')*100, std(acc, 'omitnan')*100, sum(~isnan(acc)));
end

fprintf('\n=== paired comparison against the reference set ===\n');
base = ACC(:,1);
fprintf('%-46s %8s %8s %10s %10s %8s\n', 'set', 'mean%', 'delta', 't', 'p', 'better');
for s = 1:numel(setDirs)
    v = ACC(:,s);
    ok = ~isnan(v) & ~isnan(base);
    if s == 1
        fprintf('%-46s %8.2f %8s %10s %10s %8s\n', labels{s}, mean(base(ok))*100, '-', '-', '-', '-');
        continue
    end
    dlt = (v(ok) - base(ok)) * 100;
    [~, p, ~, st] = ttest(v(ok), base(ok));
    fprintf('%-46s %8.2f %+8.2f %10.3f %10.4g %5d/%d\n', labels{s}, mean(v(ok))*100, ...
        mean(dlt), st.tstat, p, sum(dlt > 0), numel(dlt));
end

T = array2table(ACC*100, 'VariableNames', matlab.lang.makeValidName(labels));
T.subject = string(erase(subjFiles(:), '_sLORETA_raw.mat'));
csvf = fullfile(cfg.out, 'accuracy_by_variant.csv');
writetable(T, csvf);
fprintf('\nwrote %s\n', csvf);


function a = lobo_acc(X, y, b)
% Leave-one-block-out accuracy of a linear one-vs-all SVM.
pred = zeros(numel(y),1);
ub = unique(b);
for k = 1:numel(ub)
    tr = b ~= ub(k); te = ~tr;
    if ~any(tr) || ~any(te), continue, end
    mdl = fitcecoc(X(tr,:), y(tr), ...
        'Learners', templateSVM('KernelFunction','linear'), ...
        'Coding', 'onevsall', 'verbose', 0);
    pred(te) = predict(mdl, X(te,:));
end
a = sum(pred == y) / numel(y);
end


function nm = dirName(p)
% Name of the last component of a directory path.
p = char(p);
while ~isempty(p) && (p(end) == '/' || p(end) == '\'), p(end) = []; end
[~, base, ext] = fileparts(p);
nm = [base ext];
end
