%% =========================================================================
%  s35_muscle_component_timecourse.m
%
%  High-frequency envelope of the muscle components over the trial, for the
%  covert and the overt blocks.
%
%  The covert task carries no behavioural readout of speech timing: the only
%  recorded markers are the cue and the phrase label, with no response
%  event. Whether covert trials contain articulatory muscle activity, and
%  when, therefore has to be read from the EEG itself. This script measures
%  the envelope of the muscle components and tests it against the pre-trial
%  baseline point by point. The overt condition is measured identically and
%  serves as a positive control: participants there do speak, so a
%  time-locked rise must be detectable if the measurement is sensitive
%  enough for the covert result to mean anything.
%
%  COMPONENT SELECTION
%    ICLabel Muscle > 0.9, the same rule the cleaning pipeline uses. Both
%    conditions are read from the datasets after ICA but before component
%    removal, because the muscle components are precisely the ones the
%    cleaning discards. Brain components are measured alongside as a
%    reference, so that a global drift can be told from a muscle-specific one.
%
%  MEASURE
%    20-95 Hz band-pass (49-51 Hz is already notched out), Hilbert envelope,
%    filtered epoch by epoch, averaged over trials, then expressed relative
%    to the -450..-200 ms baseline both as a z score and as a percent change.
%
%  The datasets are read with load('-mat') plus a direct read of the .fdt, so
%  EEGLAB is not required.
%
%  ENVIRONMENT
%    CONDITION  'covert' (default) or 'overt'
%    SMOKE      set to any value to process only the first three subjects
%
%  INPUT   covert  cfg.eeg/*_Filters_processed_trials_precut_ICA.set
%          overt   <cfg.out>/overt_ica/*_Filters_overt_processed_trials_precut_ICA.set
%  OUTPUT  <cfg.out>/muscle_ic_timecourse/<condition>/
%            s35_muscle_ic_timecourse.mat, s35_muscle_ic_timecourse.png
% =========================================================================

clearvars; clc;

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

COND = getenv('CONDITION');
if isempty(COND); COND = 'covert'; end
assert(any(strcmp(COND, {'covert','overt'})), 'CONDITION must be covert or overt');

if strcmp(COND, 'covert')
    address = cfg.eeg;
    pattern = '*_Filters_processed_trials_precut_ICA.set';
else
    address = fullfile(cfg.out, 'overt_ica');
    pattern = '*_Filters_overt_processed_trials_precut_ICA.set';
end
outdir = fullfile(cfg.out, 'muscle_ic_timecourse', COND);
if ~isfolder(outdir); mkdir(outdir); end

IC_TH    = 0.9;            % ICLabel threshold, as in the cleaning pipeline
BP       = [20 95];        % muscle band; 49-51 Hz is notched out
BASE_MS  = [-450 -200];    % baseline window, clear of the epoch edges
SMOKE    = ~isempty(getenv('SMOKE'));

fprintf('=== CONDITION = %s ===\n%s\n', COND, address);

d = dir(fullfile(address, pattern));
assert(~isempty(d), 'no datasets matching %s in %s', pattern, address);

% S0013 is excluded, as in the main analysis: it has no structural MRI and
% therefore no source reconstruction. Selected by identifier rather than by
% position, so the exclusion cannot shift onto another subject.
sids_all = arrayfun(@(x) x.name(1:5), d, 'UniformOutput', false);
drop = strcmp(sids_all, 'S0013');
assert(sum(drop) <= 1, 'S0013 matched %d files', sum(drop));
if numel(d) == 58
    assert(any(drop), 'expected S0013 among the 58 files, found none');
    d(drop) = [];
end
nSub = numel(d);
if SMOKE; nSub = 3; end

% read one header for the time axis
S0    = load(fullfile(address, d(1).name), '-mat');
srate = double(S0.srate);
pnts  = double(S0.pnts);
times = double(S0.times(:))';               % ms
baseIdx = times >= BASE_MS(1) & times <= BASE_MS(2);

[b, a] = butter(4, BP / (srate/2), 'bandpass');

tc_mus        = nan(nSub, pnts);  % muscle components, baseline z
tc_mus_pct    = nan(nSub, pnts);  % same, as percent change from baseline
tc_brain      = nan(nSub, pnts);  % brain components, reference
tc_mus_byword = nan(nSub, 5, pnts);
info = struct('sid', {}, 'nMusIC', {}, 'nBrainIC', {}, 'nTrials', {});

for s = 1:nSub
    name = d(s).name(1:5);
    S    = load(fullfile(address, d(s).name), '-mat');
    cls  = S.etc.ic_classification.ICLabel.classifications;
    nIC  = size(S.icawinv, 2);

    % --- component classes, as in the cleaning pipeline ---
    t_mus   = find(cls(:,2) > IC_TH)';
    t_eye   = find(cls(:,3) > IC_TH)';
    t_heart = find(cls(:,4) > IC_TH)';
    t_line  = find(cls(:,5) > IC_TH)';
    t_chan  = find(cls(:,6) > IC_TH)';
    if isempty(t_heart); [~, t_heart] = max(cls(1:5,4)); end
    all_art = unique([t_mus, t_eye, t_heart, t_line, t_chan]);
    speak   = setdiff(find(cls(:,1) < 0.1)', all_art);
    t_brain = setdiff(1:nIC, [all_art, speak]);

    % --- component activations from the raw .fdt ---
    fdt = fullfile(address, [d(s).name(1:end-4) '.fdt']);
    fid = fopen(fdt, 'r');
    raw = fread(fid, [double(S.nbchan), pnts*double(S.trials)], 'float32');
    fclose(fid);

    icachansind = double(S.icachansind(:))';
    act = (S.icaweights * S.icasphere) * raw(icachansind, :);   % nIC x (pnts*trials)
    nTr = double(S.trials);

    wordlab = local_epoch_labels(S, nTr);

    [tc_mus(s,:), tc_mus_pct(s,:)] = local_env(act, t_mus,   b, a, pnts, nTr, baseIdx);
    tc_brain(s,:)                  = local_env(act, t_brain, b, a, pnts, nTr, baseIdx);
    for w = 1:5
        sel = find(wordlab == w);
        if numel(sel) >= 5
            tc_mus_byword(s,w,:) = local_env(act(:, local_expand(sel, pnts)), ...
                                             t_mus, b, a, pnts, numel(sel), baseIdx);
        end
    end

    info(s).sid = name; info(s).nMusIC = numel(t_mus);
    info(s).nBrainIC = numel(t_brain); info(s).nTrials = nTr;
    fprintf('%s  musIC=%2d  brainIC=%2d  trials=%d\n', name, numel(t_mus), numel(t_brain), nTr);
end

save(fullfile(outdir, 's35_muscle_ic_timecourse.mat'), ...
     'tc_mus', 'tc_mus_pct', 'tc_brain', 'tc_mus_byword', 'times', 'info', ...
     'BP', 'BASE_MS', 'IC_TH');

% --- group level: time-point-wise t test against the baseline, FDR corrected ---
valid = ~all(isnan(tc_mus), 2);
X = tc_mus(valid, :);
[~, p] = ttest(X);
q = local_fdr(p);
gm   = mean(X, 1, 'omitnan');
sem  = std(X, 0, 1, 'omitnan') ./ sqrt(sum(valid));
gpct = mean(tc_mus_pct(valid,:), 1, 'omitnan');

% filtfilt and hilbert leave transients at both epoch edges, so the analysis
% is restricted to the interior of the epoch
EDGE_MS = [-450 1400];
ok   = times >= EDGE_MS(1) & times <= EDGE_MS(2);
sig  = (q < 0.05) & ok;
post = (times > 0) & ok;
fprintf('\n=== group level (N=%d, restricted to %d..%d ms) ===\n', ...
        sum(valid), EDGE_MS(1), EDGE_MS(2));

% the question is whether the envelope RISES after the stimulus, not whether
% it changes in either direction
up = sig & post & (gm > 0);
if any(up)
    fprintf('significant increase from %.0f to %.0f ms (%d time points)\n', ...
            times(find(up,1,'first')), times(find(up,1,'last')), sum(up));
else
    fprintf('no significant post-stimulus increase (FDR q < 0.05)\n');
end
dn = sig & post & (gm < 0);
if any(dn)
    fprintf('significant decrease from %.0f to %.0f ms (%d time points)\n', ...
            times(find(dn,1,'first')), times(find(dn,1,'last')), sum(dn));
end

gpost = gm; gpost(~post) = -Inf;
[pk, pkIdx] = max(gpost);
fprintf('post-stimulus maximum: z=%+.3f (%+.1f%%) at %.0f ms\n', pk, gpct(pkIdx), times(pkIdx));
gmin = gm; gmin(~post) = Inf;
[tr, trIdx] = min(gmin);
fprintf('post-stimulus minimum: z=%+.3f (%+.1f%%) at %.0f ms\n', tr, gpct(trIdx), times(trIdx));

% 594 ms is the mean speech onset measured from the overt audio recordings
i594 = local_nearest(times, 594);
fprintf('at 594 ms            : z=%+.3f (%+.1f%%), p=%.3g, q=%.3g\n', ...
        gm(i594), gpct(i594), p(i594), q(i594));

% means over the decoding window and over the articulation period
w1 = times >= 200 & times <= 400;  w2 = times >= 600 & times <= 1200;
fprintf('200-400 ms           : z=%+.3f (%+.1f%%)\n', mean(gm(w1)), mean(gpct(w1)));
fprintf('600-1200 ms          : z=%+.3f (%+.1f%%)\n', mean(gm(w2)), mean(gpct(w2)));

% brain components, to show how much of any change is global
gb_ = mean(tc_brain(valid,:), 1, 'omitnan');
fprintf('brain components     : 200-400 z=%+.3f, 600-1200 z=%+.3f\n', ...
        mean(gb_(w1)), mean(gb_(w2)));

% --- does the muscle envelope differ between phrases? A phrase-dependent
% muscle signal would itself be able to carry phrase information, so it is
% tested explicitly and reported whichever way it comes out.
fprintf('\n=== phrase dependence of the muscle envelope (one-way rmANOVA) ===\n');
for wi = 1:2
    if wi == 1; win = w1; wn = '200-400 ms'; else; win = w2; wn = '600-1200 ms'; end
    M = squeeze(mean(tc_mus_byword(:, :, win), 3));      % nSub x 5
    M = M(all(isfinite(M), 2), :);
    [F, pv, df1, df2] = local_rmanova(M);
    fprintf('%s : F(%d,%d)=%.2f, p=%.3f  (n=%d)\n', wn, df1, df2, F, pv, size(M,1));
end

% --- figure ---
f = figure('Position', [100 100 760 420], 'Color', 'w');
hold on;
tt = times(ok); mm = gm(ok); ss = sem(ok); bb = gb_(ok);
fill([tt fliplr(tt)], [mm+ss fliplr(mm-ss)], [0.85 0.3 0.3], ...
     'FaceAlpha', 0.25, 'EdgeColor', 'none');
plot(tt, mm, 'Color', [0.8 0.15 0.15], 'LineWidth', 1.8);
plot(tt, bb, 'Color', [0.35 0.35 0.35], 'LineWidth', 1.2);
ylim([min(-1.2, min(mm)-0.3) max(0.6, max(mm)+0.3)]);
yl = ylim;
plot([594 594], yl, 'k--', 'LineWidth', 1);
plot([0 0], yl, 'k-', 'LineWidth', 0.5);
text(600, yl(2)*0.92, ' overt onset 594 ms', 'FontSize', 9);
if any(sig & post)
    plot(times(sig & post), repmat(yl(1)+0.02*range(yl), 1, sum(sig & post)), ...
         's', 'MarkerSize', 3, 'MarkerFaceColor', [0.8 0.15 0.15], 'MarkerEdgeColor', 'none');
end
xlabel('Time from stimulus onset (ms)'); ylabel('Envelope (baseline z)');
legend({'s.e.m.', 'Muscle ICs', 'Brain ICs'}, 'Box', 'off', 'Location', 'southwest');
title(sprintf('%s: muscle-IC %d-%d Hz envelope (N=%d)', ...
      [upper(COND(1)) COND(2:end)], BP(1), BP(2), sum(valid)));
box off; xlim(EDGE_MS);
print(f, fullfile(outdir, 's35_muscle_ic_timecourse.png'), '-dpng', '-r200');
fprintf('\nwrote %s\n', outdir);

%% ---------- helpers ----------
function [e, epct] = local_env(act, ics, b, a, pnts, nTr, baseIdx)
    % Filter and take the envelope epoch by epoch. Concatenating the epochs
    % and filtering the result as one continuous signal is not equivalent:
    % the discontinuity at each epoch boundary produces a transient at the
    % start of the segment, which is exactly where the baseline window sits.
    if isempty(ics); e = nan(1, pnts); epct = nan(1, pnts); return; end
    A3 = reshape(act(ics, :), numel(ics), pnts, nTr);
    E  = zeros(numel(ics), pnts, nTr);
    for t = 1:nTr
        x = filtfilt(b, a, double(A3(:,:,t))')';
        E(:,:,t) = abs(hilbert(x'))';
    end
    E = reshape(mean(E, 3), numel(ics), pnts);    % average across trials
    mu = mean(E(:, baseIdx), 2);
    sd = std(E(:, baseIdx), 0, 2);
    sd(sd == 0) = eps;
    e    = mean((E - mu) ./ sd, 1);               % baseline z, averaged over components
    epct = mean((E - mu) ./ mu, 1) * 100;         % percent change from baseline
end

function idx = local_expand(trials, pnts)
    idx = reshape(((trials(:)-1) * pnts) + (1:pnts), 1, []);
end

function lab = local_epoch_labels(S, nTr)
    % Phrase of each epoch, taken from the event at latency zero. Event types
    % are of the form 'C 1_u_1_b_2' (phrase 1, block 2); only the phrase
    % number following the condition letter is needed here.
    lab = zeros(1, nTr);
    try
        for t = 1:nTr
            ev = S.epoch(t).eventtype;
            if ~iscell(ev); ev = {ev}; end
            lt = S.epoch(t).eventlatency;
            if ~iscell(lt); lt = {lt}; end
            for k = 1:numel(ev)
                s = ev{k}; if ~ischar(s); continue; end
                tok = regexp(s, '^[CO]\s*(\d+)', 'tokens', 'once');
                if ~isempty(tok) && abs(lt{k}) < 1e-6
                    lab(t) = str2double(tok{1}); break;
                end
            end
        end
    catch
    end
end

function q = local_fdr(p)
    [ps, ix] = sort(p(:)); m = numel(ps);
    qs = ps .* m ./ (1:m)';
    for i = m-1:-1:1; qs(i) = min(qs(i), qs(i+1)); end
    q = nan(size(p)); q(ix) = qs;
end

function i = local_nearest(v, x)
    [~, i] = min(abs(v - x));
end

function [F, p, df1, df2] = local_rmanova(M)
    % one-way repeated measures ANOVA; M is nSubjects x nConditions
    [n, k] = size(M);
    gmn = mean(M(:));
    SS_cond = n * sum((mean(M,1) - gmn).^2);
    SS_subj = k * sum((mean(M,2) - gmn).^2);
    SS_err  = sum((M(:) - gmn).^2) - SS_cond - SS_subj;
    df1 = k - 1; df2 = (n-1)*(k-1);
    F = (SS_cond/df1) / (SS_err/df2);
    p = 1 - fcdf(F, df1, df2);
end
