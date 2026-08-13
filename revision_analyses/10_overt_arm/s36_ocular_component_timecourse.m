%% =========================================================================
%  s36_ocular_component_timecourse.m -- ocular activity, covert vs overt
%
%  WHY
%    Phrase discriminability during reading could in principle be carried by
%    eye movements rather than by speech-related activity. Comparing the two
%    reading conditions settles that only if the conditions do not themselves
%    differ in ocular behaviour, and they might: the instruction differs
%    (read silently versus read aloud), which could change fixation duration
%    or scan pattern even though the visual input is identical. Both
%    conditions have their own ICA and their own confirmed ocular components,
%    so the same measurement can be taken twice and compared within
%    participant.
%
%  WHAT COUNTS AS OCULAR
%    ICLabel Eye > 0.9, the automatic rule of the cleaning pipeline, in union
%    with the components a rater confirmed:
%      covert -> the per-subject table of blink and lateral components used
%                by the covert cleaning pipeline, reproduced below
%      overt  -> the entries in s26_remove_artefact_components.m
%    The union is used in both conditions so that the criterion stays
%    symmetric. ICLabel alone under-counts the covert condition, where about
%    a quarter of the confirmed lateral components fall below 0.9.
%
%  MEASURE
%    10 Hz low-pass envelope (blinks are slow monophasic deflections and
%    lateral saccades are step-like, both low-frequency, so the 20-95 Hz band
%    used for muscle activity would miss them), filtered per epoch and
%    expressed as percent change from the -450..-200 ms baseline.
%
%  Two comparisons are reported: the paired amplitude difference between
%  conditions point by point, FDR corrected, and a one-way repeated measures
%  ANOVA of the envelope over the five phrases within each condition. Only
%  phrase-dependent ocular activity could produce phrase discriminability,
%  so the second comparison is the one that bears on the question.
%
%  INPUT   covert  cfg.eeg/*_Filters_processed_trials_precut_ICA.set
%          overt   <cfg.out>/overt_ica/*_Filters_overt_processed_trials_precut_ICA.set
%  OUTPUT  <cfg.out>/ocular_ic_timecourse/
%            s36_ocular_ic_timecourse.mat, s36_ocular_ic_timecourse_summary.txt
% =========================================================================

clearvars; clc;

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

dirs = struct( ...
  'covert', cfg.eeg, ...
  'overt',  fullfile(cfg.out, 'overt_ica'));
pats = struct( ...
  'covert', '*_Filters_processed_trials_precut_ICA.set', ...
  'overt',  '*_Filters_overt_processed_trials_precut_ICA.set');
outdir = fullfile(cfg.out, 'ocular_ic_timecourse');
if ~isfolder(outdir); mkdir(outdir); end

EYE_TH  = 0.9;
LP      = 10;                 % Hz low-pass; the data are already 1-100 Hz
BASE_MS = [-450 -200];
EDGE_MS = [-450 1400];
SMOKE   = ~isempty(getenv('SMOKE'));

% --- rater-confirmed ocular components, {subject id, blink, lateral} -------
cov_manual = { ...
     9 2 12; 11 2 11; 12 2 5; 14 1 5; 15 2 3; 16 2 9; 17 2 20; 18 3 14; 20 1 5; ...
    21 2 3; 22 2 6; 23 2 7; 24 2 6; 25 2 13; 26 2 7; 27 4 11; 28 2 7; 29 2 [4,5]; ...
    30 2 5; 31 1 7; 32 3 19; 33 3 4; 34 2 5; 35 1 4; 36 4 15; 37 [1,2] [4,6]; ...
    38 2 3; 39 2 12; 40 [3,5] 2; 41 4 5; 42 1 6; 43 2 13; 44 2 [4,8]; 45 1 5; ...
    46 1 3; 47 1 6; 48 2 4; 49 2 5; 50 2 6; 51 4 2; 52 2 10; 53 5 13; 54 2 10; ...
    55 1 4; 56 1 3; 57 1 6; 58 1 8; 59 1 9; 60 2 3; 61 2 4; 62 2 5; 63 1 7; ...
    64 1 5; 65 1 3; 66 2 [11,14]; 67 1 4; 68 2 7};
ovt_manual = {34 2 [4 7]; 36 2 5; 39 2 15; 43 2 [3 4]; 44 2 [4 7]; 47 1 8; 54 [1 4] 5};

conds = {'covert', 'overt'};
TC = struct(); BW = struct(); INFO = struct();

for ci = 1:2
    cond = conds{ci};
    d = dir(fullfile(dirs.(cond), pats.(cond)));
    assert(~isempty(d), 'no datasets matching %s in %s', pats.(cond), dirs.(cond));
    sids = arrayfun(@(x) x.name(1:5), d, 'UniformOutput', false);
    drop = strcmp(sids, 'S0013');            % no structural MRI, excluded throughout
    if numel(d) == 58, d(drop) = []; end
    nSub = numel(d); if SMOKE, nSub = 3; end

    S0 = load(fullfile(dirs.(cond), d(1).name), '-mat');
    srate = double(S0.srate); pnts = double(S0.pnts);
    times = double(S0.times(:))';
    baseIdx = times >= BASE_MS(1) & times <= BASE_MS(2);
    [b, a] = butter(4, LP/(srate/2), 'low');

    tc = nan(nSub, pnts); ids = cell(nSub,1); nEye = nan(nSub,1);
    byword = nan(nSub, 5, pnts);
    if strcmp(cond, 'covert'), man = cov_manual; else, man = ovt_manual; end
    man_ids = cell2mat(man(:,1));

    for s = 1:nSub
        name = d(s).name(1:5);
        S = load(fullfile(dirs.(cond), d(s).name), '-mat');
        cls = S.etc.ic_classification.ICLabel.classifications;

        eye_ics = find(cls(:,3) > EYE_TH)';
        r = find(man_ids == str2double(name(2:end)), 1);
        if ~isempty(r)
            extra = [tolist(man{r,2}), tolist(man{r,3})];
            extra = extra(extra >= 1 & extra <= size(cls,1));
            eye_ics = unique([eye_ics, extra]);
        end

        if isempty(eye_ics)
            fprintf('%-6s %-7s no ocular component -- skipped\n', cond, name);
            ids{s} = name; continue
        end

        fdt = fullfile(dirs.(cond), [d(s).name(1:end-4) '.fdt']);
        fid = fopen(fdt, 'r');
        raw = fread(fid, [double(S.nbchan), pnts*double(S.trials)], 'float32');
        fclose(fid);
        ici = double(S.icachansind(:))';
        act = (S.icaweights * S.icasphere) * raw(ici, :);
        nTr = double(S.trials);

        tc(s,:) = env_of(act, eye_ics, b, a, pnts, nTr, baseIdx);

        % per phrase, because the question is not how much ocular activity
        % there is but whether it DIFFERS BETWEEN PHRASES
        wl = epoch_phrases(S, nTr);
        for w = 1:5
            sel = find(wl == w);
            if numel(sel) >= 5
                cols = reshape(((sel(:)-1)*pnts) + (1:pnts), 1, []);
                byword(s,w,:) = env_of(act(:,cols), eye_ics, b, a, pnts, numel(sel), baseIdx);
            end
        end

        ids{s} = name; nEye(s) = numel(eye_ics);
        fprintf('%-6s %-7s eyeIC=%2d  trials=%d\n', cond, name, numel(eye_ics), nTr);
    end
    TC.(cond) = tc; BW.(cond) = byword;
    INFO.(cond) = struct('ids', {ids}, 'nEye', nEye, 'times', times);
end

times = INFO.covert.times;
ok = times >= EDGE_MS(1) & times <= EDGE_MS(2);

% ---- pair the participants present in both conditions ----
[shared, ia, ib] = intersect(INFO.covert.ids, INFO.overt.ids, 'stable');
C = TC.covert(ia,:); O = TC.overt(ib,:);
good = ~all(isnan(C),2) & ~all(isnan(O),2);
C = C(good,:); O = O(good,:); shared = shared(good);
n = numel(shared);
fprintf('\npaired participants: %d\n', n);

[~, p] = ttest(O, C);                       % paired, per time point
q = fdrq(p);
D = O - C;
post = times > 0 & ok;
sig = (q < 0.05) & ok;

w = times >= 200 & times <= 400;
[~, pw, ~, sw] = ttest(mean(O(:,w),2), mean(C(:,w),2));
w2 = times >= 600 & times <= 1200;
[~, pw2, ~, sw2] = ttest(mean(O(:,w2),2), mean(C(:,w2),2));

% ---- is ocular activity phrase-specific? ----
L2 = {};
for ci = 1:2
    cond = conds{ci};
    idx = {ia(good), ib(good)};
    B = BW.(cond)(idx{ci}, :, :);
    for wi = 1:2
        if wi == 1, ww = w;  wn = '200-400 ms';
        else,       ww = w2; wn = '600-1200 ms'; end
        M = squeeze(mean(B(:,:,ww), 3));
        [F, pv, d1, d2, nn] = rmanova1(M);
        L2{end+1} = sprintf('   %-6s %-12s F(%d,%d) = %.2f, p = %.3f  (n = %d)', ...
                            cond, wn, d1, d2, F, pv, nn); %#ok<SAGROW>
    end
end

L = {
 sprintf('ocular ICs, %d Hz low-pass envelope, %% change from baseline, N = %d paired', LP, n)
 sprintf('ocular IC count : covert %.1f +/- %.1f   overt %.1f +/- %.1f', ...
         mean(INFO.covert.nEye(ia(good)),'omitnan'), std(INFO.covert.nEye(ia(good)),'omitnan'), ...
         mean(INFO.overt.nEye(ib(good)),'omitnan'),  std(INFO.overt.nEye(ib(good)),'omitnan'))
 '   (the component COUNT is not a clean comparison -- it depends on the ICA'
 '    decomposition and on ICLabel, which sees overt speech artifacts it was'
 '    not trained on. The amplitude comparison below is the reliable one.)'
 ''
 'AMPLITUDE, overt vs covert (paired):'
 sprintf('  200-400 ms  (decoding window) : covert %+.1f%%, overt %+.1f%%, diff %+.1f%%', ...
         mean(mean(C(:,w),2)), mean(mean(O(:,w),2)), mean(mean(D(:,w),2)))
 sprintf('      t(%d) = %.3f, p = %.4g', sw.df, sw.tstat, pw)
 sprintf('  600-1200 ms (articulation)    : covert %+.1f%%, overt %+.1f%%, diff %+.1f%%', ...
         mean(mean(C(:,w2),2)), mean(mean(O(:,w2),2)), mean(mean(D(:,w2),2)))
 sprintf('      t(%d) = %.3f, p = %.4g', sw2.df, sw2.tstat, pw2)
 sprintf('  timepoints with FDR q < .05 (post-stimulus) : %d / %d', sum(sig & post), sum(post))
 };
if any(sig & post)
    f = find(sig & post, 1, 'first'); l = find(sig & post, 1, 'last');
    L{end+1} = sprintf('      spanning %.0f - %.0f ms', times(f), times(l));
end
L{end+1} = '';
L{end+1} = 'PHRASE-SPECIFICITY of ocular activity (one-way rmANOVA over 5 phrases):';
L{end+1} = '  only a phrase-DEPENDENT ocular signal could produce phrase decoding';
L = [L; L2(:)];
fid = fopen(fullfile(outdir, 's36_ocular_ic_timecourse_summary.txt'), 'w');
for k = 1:numel(L), fprintf(fid, '%s\n', L{k}); fprintf('%s\n', L{k}); end
fclose(fid);

save(fullfile(outdir, 's36_ocular_ic_timecourse.mat'), 'TC', 'BW', 'INFO', 'times', ...
     'C', 'O', 'shared', 'p', 'q', 'LP', 'BASE_MS', 'EYE_TH');
fprintf('\nwrote %s\n', outdir);

%% ---------- helpers ----------
function v = tolist(x)
if iscell(x), v = [x{:}]; else, v = x(:)'; end
end

function e = env_of(act, ics, b, a, pnts, nTr, baseIdx)
% Percent change from baseline rather than baseline z. Ocular components are
% almost flat between trials and then produce a blink an order of magnitude
% larger, so the baseline standard deviation is tiny and the z score runs to
% +30. Percent change carries the same information on a readable scale.
% Filtering is per epoch: the discontinuity at each epoch boundary would
% otherwise produce a transient inside the baseline window.
A3 = reshape(act(ics,:), numel(ics), pnts, nTr);
E = zeros(numel(ics), pnts, nTr);
for t = 1:nTr
    x = filtfilt(b, a, double(A3(:,:,t))')';
    E(:,:,t) = abs(hilbert(x'))';
end
E = reshape(mean(E,3), numel(ics), pnts);
mu = mean(E(:,baseIdx),2); mu(mu == 0) = eps;
e = mean((E - mu) ./ mu, 1) * 100;
end

function lab = epoch_phrases(S, nTr)
% Phrase of each epoch, taken from the event at latency zero. Event types
% are of the form 'C 1_u_1_b_2'.
lab = zeros(1, nTr);
try
    for t = 1:nTr
        ev = S.epoch(t).eventtype;    if ~iscell(ev), ev = {ev}; end
        lt = S.epoch(t).eventlatency; if ~iscell(lt), lt = {lt}; end
        for k = 1:numel(ev)
            s = ev{k}; if ~ischar(s), continue, end
            tok = regexp(s, '^[CO]\s*(\d+)', 'tokens', 'once');
            if ~isempty(tok) && abs(lt{k}) < 1e-6
                lab(t) = str2double(tok{1}); break
            end
        end
    end
catch
end
end

function [F, p, df1, df2, nn] = rmanova1(M)
% one-way repeated measures ANOVA; M is nSubjects x nConditions
M = M(all(isfinite(M), 2), :);
[nn, k] = size(M);
g = mean(M(:));
SSc = nn * sum((mean(M,1) - g).^2);
SSs = k * sum((mean(M,2) - g).^2);
SSe = sum((M(:) - g).^2) - SSc - SSs;
df1 = k - 1; df2 = (nn-1)*(k-1);
F = (SSc/df1) / (SSe/df2);
p = 1 - fcdf(F, df1, df2);
end

function q = fdrq(p)
[ps, ix] = sort(p(:)); m = numel(ps);
qs = ps .* m ./ (1:m)';
for i = m-1:-1:1, qs(i) = min(qs(i), qs(i+1)); end
q = nan(size(p)); q(ix) = qs;
end
