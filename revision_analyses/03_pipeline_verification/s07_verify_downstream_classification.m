%% =========================================================================
%  s07_verify_downstream_classification.m
%
%  Check whether the residual numerical difference between the published source
%  data and the control run of the add-back source export has any effect on the
%  analysis that uses it.
%
%  The published source file and the control-run file of the same participant
%  are put through the classification procedure of the main analysis, and the
%  two accuracies are compared. Because the source values are compared through
%  the quantity the paper reports, this is the decisive equivalence test; the
%  value-by-value comparison in s06 is its upstream counterpart.
%
%  METHOD (identical to the main classification analysis)
%    Features   148 ROIs x 150 samples (0-600 ms after phrase onset), per-ROI
%               baseline (-0.5 to 0 s) subtracted, column-wise z-score
%    Labels     phrase identity 1-5, read from the trial label
%    Validation leave-one-block-out
%    Learner    linear SVM, one-vs-all
%
%  INPUT
%    cfg.source                     published source data
%    cfg.out/verify_standard_dest   control-run output (override with AB_VERIFY_DIR)
%
%  OUTPUT
%    printed accuracies for both files and their difference
%
%  ENV
%    AB_VERIFY_SUBJ  participant to check (default Subject09)
%    AB_VERIFY_DIR   absolute path of the control-run directory
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

subj = getenv('AB_VERIFY_SUBJ');
if isempty(subj), subj = 'Subject09'; end

vdir = getenv('AB_VERIFY_DIR');
if isempty(vdir), vdir = fullfile(cfg.out, 'verify_standard_dest'); end
files = { ...
    'published', fullfile(cfg.source, [subj '_sLORETA_raw.mat']); ...
    'control',   fullfile(vdir,       [subj '_sLORETA_raw.mat'])};

fs = 250;                    % sampling rate, Hz
pre_smp  = 0.5*fs;           % baseline, 0.5 s before phrase onset
post_smp = 1.5*fs;           % epoch after phrase onset
win_smp  = round(0.6*fs);    % classification window, 0-600 ms
nroi = 148;                  % Destrieux ROIs

acc = nan(1,2);
for r = 1:2
    S = load(files{r,2}, 'condition_data', 'condition_data_type');
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

    acc(r) = lobo_acc(X, y, b);
    fprintf('%-10s  n=%3d trials  accuracy %6.2f%%\n', files{r,1}, numel(y), acc(r)*100);
end

fprintf('\ndifference: %.4f percentage points\n', abs(acc(1)-acc(2))*100);
if acc(1) == acc(2)
    fprintf('VERDICT: identical classifier output; the residual source difference is inert.\n');
else
    fprintf('VERDICT: *** classifier output differs *** investigate before proceeding.\n');
end

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
