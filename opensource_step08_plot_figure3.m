%% opensource_step08_plot_figure3.m
% Generate Figure 3: Statistical evidence for phrase discrimination
%
% This script corresponds to Figure 3 in the manuscript.
%
% Panels:
%   a) ROI x Time heatmap (-log10 p-values)
%   b) Time window significance summary
%   c) Brain surface projections (peak windows)
%
% Requirements:
%   - Results from Step 06 and Step 07
%   - ROI label file (EEG_ROI_LABELS.csv)
%
% Author: Wei Zhang
% Affiliation: Nanyang Technological University
% License: CC BY-NC 4.0
%
%==========================================================================

clear all
clc
close all

%% USER CONFIGURATION
data_path = '/path/to/data/';
results_folder = fullfile(data_path, 'results');
atlas_file = fullfile(data_path, 'atlas', 'EEG_ROI_LABELS.csv');
output_folder = fullfile(data_path, 'figures');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% LOAD RESULTS
load(fullfile(results_folder, 'rmanova_results.mat'));
roi_labels = readtable(atlas_file);

fprintf('=== Step 08: Plot Figure 3 ===\n');

%% PANEL A: ROI x Time Heatmap
figure('Position', [100 100 1200 600]);

subplot(2, 2, [1 3]);

% Convert p-values to -log10
neg_log_p = -log10(p_values);
neg_log_p(isinf(neg_log_p)) = 10;  % Cap at 10

% Plot heatmap
imagesc(neg_log_p);
colormap(hot);
colorbar;
caxis([0 5]);

xlabel('Time Window (50ms bins)');
ylabel('ROI Index');
title('Figure 3a: -log_{10}(p) Heatmap');

% Add significance threshold line
hold on;
contour(neg_log_p, [-log10(0.05)], 'w', 'LineWidth', 1);

% Time labels
xticks(1:5:30);
xticklabels({'0', '250', '500', '750', '1000', '1250'});

%% PANEL B: Time Window Summary
subplot(2, 2, 2);

% Count significant ROIs per window
sig_rois_per_window = sum(q_values < 0.05, 1);

bar(sig_rois_per_window, 'FaceColor', [0.8 0.2 0.2]);
xlabel('Time Window (50ms bins)');
ylabel('# Significant ROIs');
title('Figure 3b: Significant ROIs per Window');

xticks(1:5:30);
xticklabels({'0', '250', '500', '750', '1000', '1250'});

% Mark peak window
[~, peak_window] = max(sig_rois_per_window);
hold on;
plot(peak_window, sig_rois_per_window(peak_window), 'k*', 'MarkerSize', 15);
text(peak_window, sig_rois_per_window(peak_window) + 2, ...
     sprintf('Peak: %d-%dms', (peak_window-1)*50, peak_window*50), ...
     'HorizontalAlignment', 'center');

%% PANEL C: Summary Statistics
subplot(2, 2, 4);

% Bar plot of significant pairs
sig_count = sum(q_values(:) < 0.05);
total_count = numel(q_values);

bar([sig_count, total_count - sig_count], 'stacked');
set(gca, 'XTickLabel', {'Significant', 'Not Significant'});
ylabel('ROI-Window Pairs');
title(sprintf('Figure 3c: %d/%d pairs significant (%.1f%%)', ...
      sig_count, total_count, 100*sig_count/total_count));

%% SAVE FIGURE
saveas(gcf, fullfile(output_folder, 'figure3_statistical_evidence.png'));
saveas(gcf, fullfile(output_folder, 'figure3_statistical_evidence.fig'));

fprintf('Figure saved to: %s\n', output_folder);
fprintf('=== Step 08 Complete ===\n');
