%% opensource_step25_plot_figure6.m
% Generate Figure 6: Connectivity analysis summary
%
% This script corresponds to Figure 6 in the manuscript.
%
% Panels:
%   a) Method demo - wPLI/dPLI computation
%   b) ROI 55 node strength across periods
%   c) F-value ranking across ROIs
%   d) Connectivity network visualization
%   e) dPLI direction pathway
%   f) Duration-connectivity scatter
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
load(fullfile(results_folder, 'wpli_results.mat'));
load(fullfile(results_folder, 'hub_analysis_results.mat'));
load(fullfile(results_folder, 'information_flow_results.mat'));
load(fullfile(results_folder, 'duration_correlation_results.mat'));

fprintf('=== Step 25: Plot Figure 6 ===\n');

%% FIGURE SETUP
figure('Position', [50 50 1600 1000]);

%% PANEL A: Method Demo
subplot(2, 3, 1);
% Schematic of wPLI/dPLI computation

% Generate example signals
t = 0:0.001:1;
f = 5;  % 5 Hz
signal1 = sin(2*pi*f*t);
signal2 = sin(2*pi*f*t + pi/4);  % 45 degree phase lag

plot(t(1:250), signal1(1:250), 'b-', 'LineWidth', 1.5);
hold on;
plot(t(1:250), signal2(1:250), 'r-', 'LineWidth', 1.5);

xlabel('Time (s)');
ylabel('Amplitude');
title('Figure 6a: Phase Lag Index Demo');
legend({'ROI 1', 'ROI 2 (lagging)'}, 'Location', 'northeast');

%% PANEL B: ROI 55 Node Strength
subplot(2, 3, 2);

roi_idx = 55;
band_names = {'Delta', 'Theta', 'Alpha', 'Beta'};
period_names = {'Plan', 'Exec'};

% Extract ROI 55 node strength
ns_roi55 = squeeze(hub_results.node_strength(:, roi_idx, :, :));  % subjects x bands x periods

% Mean and SEM
mean_ns = squeeze(mean(ns_roi55, 1));  % bands x periods
sem_ns = squeeze(std(ns_roi55, 0, 1) / sqrt(size(ns_roi55, 1)));

bar_data = mean_ns';  % periods x bands
errorbar_data = sem_ns';

b = bar(bar_data);
hold on;

% Add error bars
ngroups = size(bar_data, 1);
nbars = size(bar_data, 2);
groupwidth = min(0.8, nbars/(nbars + 1.5));

for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, bar_data(:,i), errorbar_data(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1);
end

set(gca, 'XTickLabel', period_names);
ylabel('Node Strength');
title('Figure 6b: ROI 55 Node Strength');
legend(band_names, 'Location', 'northeast');

% Mark significant result
text(1, max(bar_data(1,:)) + 0.02, '*', 'FontSize', 20, 'HorizontalAlignment', 'center');

%% PANEL C: F-value Ranking
subplot(2, 3, 3);

F_values = hub_results.F_values;
[~, sort_idx] = sort(F_values, 'descend');

% Top 20 ROIs
top_n = 20;
bar(F_values(sort_idx(1:top_n)), 'FaceColor', [0.3 0.6 0.9]);

xlabel('ROI Rank');
ylabel('F-value');
title('Figure 6c: F-value Ranking (Top 20 ROIs)');

% Highlight ROI 55
roi55_rank = find(sort_idx == 55);
if roi55_rank <= top_n
    hold on;
    bar(roi55_rank, F_values(55), 'FaceColor', [0.9 0.3 0.3]);
    text(roi55_rank, F_values(55) + 0.5, 'ROI 55', ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

%% PANEL D: Connectivity Network
subplot(2, 3, 4);

% Simplified network visualization
% Show top connections from ROI 55

mean_wpli = squeeze(mean(wpli_all(:, :, :, 1, 1), 1));  % Delta, Plan
roi55_connections = mean_wpli(55, :);
[~, top_connected] = sort(roi55_connections, 'descend');
top_connected = top_connected(1:10);

% Circular layout
theta = linspace(0, 2*pi, 11);
theta = theta(1:end-1);
x = cos(theta);
y = sin(theta);

scatter(x, y, 200, 'filled', 'MarkerFaceColor', [0.5 0.5 0.9]);
hold on;

% Connect to center (ROI 55)
scatter(0, 0, 300, 'filled', 'MarkerFaceColor', [0.9 0.3 0.3]);

for i = 1:10
    line_alpha = roi55_connections(top_connected(i)) / max(roi55_connections);
    plot([0 x(i)], [0 y(i)], 'k-', 'LineWidth', 2*line_alpha + 0.5);
end

text(0, 0.15, 'ROI 55', 'HorizontalAlignment', 'center', 'FontWeight', 'bold');
axis equal off;
title('Figure 6d: ROI 55 Connectivity Network');

%% PANEL E: Information Flow Pathway
subplot(2, 3, 5);

% Arrow diagram: Angular -> Postcentral -> Fusiform
positions = [0.2, 0.5; 0.5, 0.5; 0.8, 0.5];
roi_names = {'Angular', 'Postcentral', 'Fusiform'};

for i = 1:3
    scatter(positions(i,1), positions(i,2), 500, 'filled', ...
            'MarkerFaceColor', [0.3 0.6 0.9]);
    text(positions(i,1), positions(i,2)-0.15, roi_names{i}, ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

% Draw arrows
for i = 1:2
    annotation('arrow', [0.15 + 0.3*(i-1), 0.35 + 0.3*(i-1)], ...
               [0.35, 0.35], 'LineWidth', 2, 'HeadWidth', 15);
end

axis([0 1 0 1]);
axis off;
title('Figure 6e: Information Flow Direction');

% Add note
text(0.5, 0.2, 'Reversed vs DIVA model', 'HorizontalAlignment', 'center', ...
     'FontStyle', 'italic', 'Color', [0.5 0.5 0.5]);

%% PANEL F: Duration Correlation
subplot(2, 3, 6);

% Load duration data
word_durations = duration_results.word_durations;
ns_per_word = duration_results.ns_per_word;

scatter(duration_results.word_durations, mean(ns_per_word, 1), 150, 'filled', ...
        'MarkerFaceColor', [0.3 0.6 0.9]);
hold on;

% Fit line
coeffs = polyfit(word_durations, mean(ns_per_word, 1), 1);
x_line = linspace(min(word_durations), max(word_durations), 100);
plot(x_line, polyval(coeffs, x_line), 'r-', 'LineWidth', 2);

xlabel('Duration (ms)');
ylabel('Node Strength');
title(sprintf('Figure 6f: Duration Correlation\n\\rho = %.3f, p = %.4f', ...
      duration_results.rho, duration_results.p_value));

%% SAVE
saveas(gcf, fullfile(output_folder, 'figure6_connectivity_analysis.png'));
saveas(gcf, fullfile(output_folder, 'figure6_connectivity_analysis.fig'));

fprintf('Figure saved to: %s\n', output_folder);
fprintf('=== Step 25 Complete ===\n');
