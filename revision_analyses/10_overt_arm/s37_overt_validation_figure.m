%% =========================================================================
%  s37_overt_validation_figure.m
%
%  Supplementary figure for the overt-block analyses: the muscle-component
%  envelope across the trial, covert and overt on the same axis, with the
%  cohort mean and its standard error.
%
%  The panel is what makes the overt-versus-covert decoding comparison
%  interpretable, because it bounds articulation in time: the 200-400 ms
%  decoding window is shaded, and the mean speech onset measured from the
%  overt audio recordings (594 ms) is marked, so the reader can see where the
%  overt muscle signal sits relative to the window the classifier uses.
%
%  The ribbons below the traces mark time points at which the envelope is
%  ELEVATED above its own baseline: a one-sided test at each post-stimulus
%  time point, BH-FDR corrected. Both conditions are tested by exactly the
%  same rule, so an empty covert ribbon is a result rather than an omitted
%  comparison.
%
%  INPUT   <cfg.out>/muscle_ic_timecourse/{covert,overt}/s35_muscle_ic_timecourse.mat
%  OUTPUT  <cfg.out>/figures/FigS_overt_block_validation.png
% =========================================================================

clearvars; clc;

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

edir = fullfile(cfg.out, 'muscle_ic_timecourse');
fdir = fullfile(cfg.out, 'figures', filesep);
if ~isfolder(fdir), mkdir(fdir); end

C_COV = [0.35 0.45 0.72];      % covert -- blue
C_OVT = [0.84 0.37 0.23];      % overt  -- warm red
FS    = 8;                     % base font size, journal supplementary scale

Ecov = load(fullfile(edir, 'covert', 's35_muscle_ic_timecourse.mat'));
Eovt = load(fullfile(edir, 'overt',  's35_muscle_ic_timecourse.mat'));

fig = figure('Units','centimeters','Position',[2 2 18 7.0],'Color','w');
set(fig, 'DefaultAxesFontName','Arial', 'DefaultTextFontName','Arial', ...
         'DefaultAxesFontSize',FS, 'DefaultAxesBox','off', ...
         'DefaultAxesTickDir','out', 'DefaultAxesLineWidth',0.6);

%% ---------------- muscle-component envelope ----------------
ax = axes('Position', [0.075 0.165 0.895 0.775]); hold(ax,'on');
times = Ecov.times(:)';
EDGE = [-450 1400];
ok = times >= EDGE(1) & times <= EDGE(2);

[gc, sc, qc] = grp(Ecov.tc_mus);
[go, so, qo] = grp(Eovt.tc_mus);

% decoding window, drawn first so it sits behind everything
yl = [-1.6 4.6];
patch(ax, [200 400 400 200], [yl(1) yl(1) yl(2) yl(2)], [0.93 0.93 0.85], ...
      'EdgeColor','none');
text(ax, 300, yl(2)-0.12, 'decoding window', 'HorizontalAlignment','center', ...
     'VerticalAlignment','top', 'FontSize', FS-1.5, 'Color', [0.45 0.45 0.3]);

shade(ax, times(ok), gc(ok), sc(ok), C_COV);
shade(ax, times(ok), go(ok), so(ok), C_OVT);
hc = plot(ax, times(ok), gc(ok), 'Color', C_COV, 'LineWidth', 1.3);
ho = plot(ax, times(ok), go(ok), 'Color', C_OVT, 'LineWidth', 1.3);

yline(ax, 0, 'k-', 'LineWidth', 0.5);
xline(ax, 0, 'k-', 'LineWidth', 0.5);
xline(ax, 594, 'k--', 'LineWidth', 0.9);      % mean overt speech onset

% ribbons: elevation above baseline, BH-FDR q < .05, same rule in both conditions
post = times > 0 & ok;
up_o = (qo < 0.05) & post;
up_c = (qc < 0.05) & post;
mark(ax, times, up_o, yl(1)+0.30, C_OVT);
mark(ax, times, up_c, yl(1)+0.14, C_COV);
fprintf('above-baseline time points: overt %d, covert %d\n', sum(up_o), sum(up_c));

xlim(ax, EDGE); ylim(ax, yl);
xlabel(ax, 'Time from phrase onset (ms)');
ylabel(ax, 'Muscle-IC envelope (baseline {\itz})');
legend(ax, [ho hc], {'overt','covert'}, 'Box','off', 'Location','northwest', ...
       'FontSize', FS-1);

% Values quoted alongside the figure. The 594 ms sample is the nearest one to
% the mean speech onset, matching s35_muscle_component_timecourse.m.
[~, i594] = min(abs(times - 594));
fprintf('at 594 ms      : overt z=%+.3f q=%.4g | covert z=%+.3f q=%.4g\n', ...
        go(i594), qo(i594), gc(i594), qc(i594));
w = times >= 200 & times <= 400;
fprintf('200-400 ms mean: overt z=%+.3f  covert z=%+.3f\n', mean(go(w)), mean(gc(w)));
f = find((qo < 0.05) & post & (go > 0), 1);
fprintf('overt first significant rise at %.0f ms\n', times(f));

%% ---------------- export ----------------
set(fig, 'Renderer', 'painters');
exportgraphics(fig, [fdir 'FigS_overt_block_validation.png'], 'Resolution', 400);
fprintf('wrote %sFigS_overt_block_validation.png\n', fdir);

%% ---------------- helpers ----------------
function [g, s, q] = grp(X)
% Cohort mean, standard error and FDR-corrected q values for a one-sided
% test of elevation above baseline at each time point.
v = ~all(isnan(X), 2);
X = X(v, :);
g = mean(X, 1, 'omitnan');
s = std(X, 0, 1, 'omitnan') ./ sqrt(sum(v));
[~, p2, ~, st] = ttest(X);
p1 = p2 / 2;
p1(st.tstat < 0) = 1 - p1(st.tstat < 0);
q = fdrq(p1);
end

function q = fdrq(p)
[ps, ix] = sort(p(:)); m = numel(ps);
qs = ps .* m ./ (1:m)';
for i = m-1:-1:1, qs(i) = min(qs(i), qs(i+1)); end
q = nan(size(p)); q(ix) = qs;
end

function shade(ax, t, m, e, col)
patch(ax, [t fliplr(t)], [m+e fliplr(m-e)], col, ...
      'FaceAlpha', 0.20, 'EdgeColor', 'none');
end

function mark(ax, times, msk, y, col)
if ~any(msk), return, end
plot(ax, times(msk), repmat(y, 1, sum(msk)), 's', 'MarkerSize', 2.0, ...
     'MarkerFaceColor', col, 'MarkerEdgeColor', 'none');
end
