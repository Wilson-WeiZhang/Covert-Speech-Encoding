%% opensource_step15_plot_figure4.m
% Generate Figure 4: Individual differences in neural encoding
%
% This script corresponds to Figure 4 in the manuscript.
%
% Panels:
%   a) Classification confusion matrix
%   b) Classification accuracy distribution
%   c) LME model formulations (text)
%   d) Brain R^2 maps
%   e) Variance decomposition bar chart
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
output_folder = fullfile(data_path, 'figures');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% LOAD RESULTS
load(fullfile(results_folder, 'classification_results.mat'));
load(fullfile(results_folder, 'lme_results.mat'));

fprintf('=== Step 15: Plot Figure 4 ===\n');

%% FIGURE SETUP
figure('Position', [50 50 1400 900]);

%% PANEL A: Confusion Matrix
subplot(2, 3, 1);

% Average confusion matrix across subjects
avg_confusion = mean(confusion_all, 3);
avg_confusion_norm = avg_confusion ./ sum(avg_confusion, 2);

imagesc(avg_confusion_norm);
colormap(gca, hot);
colorbar;
caxis([0 0.5]);

xlabel('Predicted Word');
ylabel('True Word');
title('Figure 4a: Confusion Matrix');
axis square;

% Add text labels
for i = 1:5
    for j = 1:5
        text(j, i, sprintf('%.0f%%', 100*avg_confusion_norm(i,j)), ...
             'HorizontalAlignment', 'center', 'Color', 'w', 'FontWeight', 'bold');
    end
end

%% PANEL B: Accuracy Distribution
subplot(2, 3, 2);

histogram(accuracies, 15, 'FaceColor', [0.3 0.6 0.9], 'EdgeColor', 'w');
hold on;

% Chance line
xline(20, 'r--', 'LineWidth', 2);

% Mean line
xline(mean(accuracies), 'k-', 'LineWidth', 2);

xlabel('Classification Accuracy (%)');
ylabel('Number of Subjects');
title(sprintf('Figure 4b: Accuracy Distribution\nMean: %.1f%% +/- %.1f%%', ...
      mean(accuracies), std(accuracies)));

legend({'Distribution', 'Chance (20%)', sprintf('Mean (%.1f%%)', mean(accuracies))}, ...
       'Location', 'northeast');

%% PANEL C: LME Model Formulations
subplot(2, 3, 3);
axis off;

text(0.1, 0.9, 'Figure 4c: LME Model Formulations', 'FontWeight', 'bold', 'FontSize', 12);
text(0.1, 0.7, 'Fixed: Y ~ Word', 'FontSize', 10, 'FontName', 'Courier');
text(0.1, 0.5, 'RI:    Y ~ Word + (1|Subject)', 'FontSize', 10, 'FontName', 'Courier');
text(0.1, 0.3, 'RS:    Y ~ Word + (Word|Subject)', 'FontSize', 10, 'FontName', 'Courier');

%% PANEL D: R^2 Brain Maps (simplified - show as heatmaps)
subplot(2, 3, 4);

% Average R^2 across windows for each ROI
R2_fixed_per_roi = mean(results.R2m_fixed, 2) * 100;
R2_rs_per_roi = mean(results.R2c_rs, 2) * 100;

% Plot as bar comparison
bar_data = [R2_fixed_per_roi, R2_rs_per_roi];
imagesc(bar_data');
colormap(gca, parula);
colorbar;

xlabel('ROI Index');
yticks([1 2]);
yticklabels({'R^2_m (Fixed)', 'R^2_c (RS)'});
title('Figure 4d: R^2 per ROI');

%% PANEL E: Variance Decomposition Bar Chart
subplot(2, 3, 5);

% Extract values
R2m = results.mean_R2m_fixed;
delta_ri = results.delta_R2_ri;
delta_rs = results.delta_R2_rs;

% Stacked bar
bar_heights = [R2m, delta_ri; 0, delta_rs - delta_ri];
b = bar([1], [R2m, delta_ri, delta_rs - delta_ri], 'stacked');

b(1).FaceColor = [0.2 0.4 0.8];  % Fixed (blue)
b(2).FaceColor = [0.8 0.4 0.2];  % RI (orange)
b(3).FaceColor = [0.4 0.8 0.2];  % RS (green)

ylabel('Variance Explained (%)');
title('Figure 4e: Variance Decomposition');
legend({'Fixed (Word)', 'Random Intercept', 'Random Slope'}, 'Location', 'northwest');

% Add text labels
total_R2 = R2m + delta_ri + (delta_rs - delta_ri);
text(1, total_R2 + 0.5, sprintf('Total: %.1f%%', total_R2), ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold');

%% PANEL F: Summary Statistics
subplot(2, 3, 6);
axis off;

text(0.1, 0.9, 'Key Results:', 'FontWeight', 'bold', 'FontSize', 12);
text(0.1, 0.75, sprintf('R^2_{marginal} (Fixed): %.2f%%', R2m), 'FontSize', 10);
text(0.1, 0.60, sprintf('Delta R^2 (RI): +%.2f%%', delta_ri), 'FontSize', 10);
text(0.1, 0.45, sprintf('Delta R^2 (RS): +%.2f%%', delta_rs), 'FontSize', 10);

if R2m > 0
    ratio = delta_rs / R2m;
    text(0.1, 0.25, sprintf('Individual/Group Ratio: %.0fx', ratio), ...
         'FontSize', 12, 'FontWeight', 'bold', 'Color', [0.8 0.2 0.2]);
end

%% SAVE
saveas(gcf, fullfile(output_folder, 'figure4_individual_differences.png'));
saveas(gcf, fullfile(output_folder, 'figure4_individual_differences.fig'));

fprintf('Figure saved to: %s\n', output_folder);
fprintf('=== Step 15 Complete ===\n');
