%% =========================================================================
%  s38_plot_spatiotemporal_maps.m
%
%  Supplementary panels for the covert and overt spatiotemporal maps of the
%  phrase effect, from the rmANOVA computed in s31. Four separate PNGs, to be
%  laid out together afterwards.
%
%  The styling matches the main spatiotemporal figure of the paper, so that
%  the supplementary panels read as the same figure family:
%
%    heatmap   -log10(p), UNCORRECTED, capped at 5, NaN mapped to 0
%              caxis [0 4], hemisphere divider at 74.5 with the left
%              hemisphere on top, x ticks every two windows labelled in
%              seconds, y ticks at [1 37 74 111 148]
%
%    HMP bar   harmonic mean p across the 148 parcels within each window,
%              from the same uncorrected p values; bar height -log10(HMP)
%              capped at 5; bar colour from the FDR-corrected HMP through the
%              same colormap; dashed line at the BH effective threshold;
%              asterisks by q
%
%  Two settings differ from the main figure, both so that the two conditions
%  can be compared against each other rather than read one at a time: the HMP
%  y limit is shared between the conditions instead of set per panel, and the
%  same colour axis is used for both heatmaps.
%
%  INPUT   <cfg.out>/overt_analysis/s31_rmanova_{covert,overt}.mat
%          assets/redmap.mat
%  OUTPUT  <cfg.out>/figures/
%            FigS_spt_a_covert_heatmap.png   FigS_spt_b_overt_heatmap.png
%            FigS_spt_c_covert_hmp.png       FigS_spt_d_overt_hmp.png
% =========================================================================

clearvars; clc;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(here), 'config'));
cfg = set_paths();

adir = fullfile(cfg.out, 'overt_analysis', filesep);
fdir = fullfile(cfg.out, 'figures', filesep);
if ~isfolder(fdir), mkdir(fdir); end

num_rois = 148; num_windows = 30; winsize = 50;
roi_reorder = [1:2:num_rois, 2:2:num_rois];     % left hemisphere first

cmapfile = fullfile(here, 'assets', 'redmap.mat');
assert(isfile(cmapfile), 'colormap not found at %s', cmapfile);
S = load(cmapfile, 'cmap'); cmap = S.cmap;
fprintf('loaded colormap (%d colours)\n', size(cmap, 1));

COND = {'covert', 'overt'};
D = struct();
for c = 1:2
    L = load([adir 's31_rmanova_' COND{c} '.mat'], 'p_values', 'q_values', 'sig');
    p = L.p_values(roi_reorder, :);

    heat = -log10(p);
    heat(heat > 5) = 5;
    heat(isnan(heat)) = 0;

    hmp = zeros(1, num_windows);
    for win = 1:num_windows
        v = p(:, win);
        v = v(~isnan(v) & v > 0 & v < 1);
        if ~isempty(v)
            hmp(win) = numel(v) / sum(1 ./ v);      % harmonic mean p
        end
    end
    hq = mafdr(hmp, 'BHFDR', true);

    D.(COND{c}) = struct('heat', heat, 'hmp', hmp, 'hq', hq, ...
                         'nsig', sum(L.sig(:)));
    fprintf('%-7s %d significant parcel-window pairs | HMP windows q<.05: %d\n', ...
        COND{c}, sum(L.sig(:)), sum(hq < 0.05));
end

% shared y limit, so the two HMP panels can be read against each other
hmp_all = [-log10(D.covert.hmp), -log10(D.overt.hmp)];
hmp_all(hmp_all > 5) = 5;
YMAX = max(hmp_all) + 1;

%% ---------------- heatmap panels ----------------
for c = 1:2
    cond = COND{c};
    figure('Position', [100, 100, 1400, 900], 'Color', 'white');
    imagesc(D.(cond).heat);
    colormap(cmap);
    caxis([0, 4]);

    cb = colorbar;
    cb.Label.String = '-log_{10}(p)';
    cb.Label.FontSize = 30;
    cb.LineWidth = 3;

    hold on;
    plot([0.5, num_windows + 0.5], [num_rois/2 + 0.5, num_rois/2 + 0.5], ...
        'k-', 'LineWidth', 3);
    hold off;

    xtick_pos = 0.5:2:num_windows + 0.5;
    xticks(xtick_pos);
    xticklabels(arrayfun(@(x) sprintf('%.1f', (x - 0.5) * winsize / 1000), ...
        xtick_pos, 'UniformOutput', false));
    xlabel('Event-locked time / s', 'FontSize', 38);

    yticks([1, 37, 74, 111, 148]);
    yticklabels({'1', '37', '74', '111', '148'});

    set(gca, 'FontSize', 24, 'LineWidth', 3, 'TickDir', 'out', 'Box', 'on');

    out = [fdir 'FigS_spt_' char('a' + c - 1) '_' cond '_heatmap.png'];
    print(gcf, out, '-dpng', '-r300');
    close(gcf);
    fprintf('wrote %s\n', out);
end

%% ---------------- HMP panels ----------------
for c = 1:2
    cond = COND{c};
    hmp = D.(cond).hmp; hq = D.(cond).hq;
    hd = -log10(hmp); hd(hd > 5) = 5;

    % taller canvas than the main figure's bar panel: the shared y limit
    % pushes the axis up and would otherwise clip the y label
    figure('Position', [100, 100, 1400, 460], 'Color', 'white');
    bh = bar(1:num_windows, hd, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);

    qd = -log10(hq); qd(qd > 4) = 4; qd(qd < 0) = 0;
    for win = 1:num_windows
        idx = round(qd(win) / 4 * (size(cmap, 1) - 1)) + 1;
        bh.CData(win, :) = cmap(max(1, min(size(cmap, 1), idx)), :);
    end

    hold on;
    hsig = hq < 0.05;
    if any(hsig)
        yline(-log10(0.05 * sum(hsig) / num_windows), 'k--', 'LineWidth', 2);
    end
    for win = 1:num_windows
        if hsig(win)
            if hq(win) < 0.005, s = '***';
            elseif hq(win) < 0.01, s = '**';
            else, s = '*';
            end
            text(win, hd(win) + 0.15, s, 'HorizontalAlignment', 'center', ...
                'FontSize', 32, 'FontWeight', 'bold');
        end
    end
    hold off;

    xtick_pos = 1:2:num_windows;
    xticks(xtick_pos);
    % one decimal place, as in the heatmap panels: two decimals make the
    % labels wide enough that MATLAB rotates them
    xticklabels(arrayfun(@(x) sprintf('%.1f', (x - 1) * winsize / 1000), ...
        xtick_pos, 'UniformOutput', false));
    xlabel('Event-locked time / s', 'FontSize', 24);
    ylabel('-log_{10}(HMP)', 'FontSize', 24);
    ylim([0, YMAX]);
    xlim([0.5, num_windows + 0.5]);
    set(gca, 'FontSize', 24, 'LineWidth', 3, 'TickDir', 'out', 'Box', 'off');

    out = [fdir 'FigS_spt_' char('c' + c - 1) '_' cond '_hmp.png'];
    print(gcf, out, '-dpng', '-r300');
    close(gcf);
    fprintf('wrote %s\n', out);
end

%% ---------------- report ----------------
fprintf('\nHMP windows with q < .05\n');
for c = 1:2
    cond = COND{c};
    idx = find(D.(cond).hq < 0.05);
    fprintf('  %-7s (%d): ', cond, numel(idx));
    for k = idx
        fprintf('%d-%d ', (k-1)*winsize, k*winsize);
    end
    fprintf('\n');
end
