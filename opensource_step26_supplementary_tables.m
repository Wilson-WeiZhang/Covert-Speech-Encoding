%% opensource_step26_supplementary_tables.m
% Generate Supplementary Table S1: Significant ROI-window pairs
%
% This script corresponds to Supplementary Table S1 in the manuscript.
%
% Content:
%   - List of significant ROI-window pairs (FDR q < 0.05)
%   - ROI name, hemisphere, time window, F-value, p-value, q-value
%
% Author: Wei Zhang
% Affiliation: Nanyang Technological University
% License: CC BY-NC 4.0
%
%==========================================================================

clear all
clc

%% USER CONFIGURATION
data_path = '/path/to/data/';
results_folder = fullfile(data_path, 'results');
atlas_file = fullfile(data_path, 'atlas', 'EEG_ROI_LABELS.csv');
output_folder = fullfile(data_path, 'tables');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% LOAD DATA
load(fullfile(results_folder, 'rmanova_results.mat'));
roi_labels = readtable(atlas_file);

fprintf('=== Step 26: Generate Supplementary Tables ===\n');

%% PARAMETERS
win_ms = 50;
alpha_level = 0.05;

%% FIND SIGNIFICANT PAIRS
[sig_rois, sig_windows] = find(q_values < alpha_level);
num_sig = length(sig_rois);

fprintf('Significant ROI-window pairs: %d\n\n', num_sig);

%% BUILD TABLE
TableS1 = table();

for i = 1:num_sig
    r = sig_rois(i);
    w = sig_windows(i);

    % ROI info
    roi_name = roi_labels.eeg_name{r};
    if contains(roi_name, ' L')
        hemisphere = 'Left';
    elseif contains(roi_name, ' R')
        hemisphere = 'Right';
    else
        hemisphere = 'Bilateral';
    end

    % Time window
    win_start_ms = (w - 1) * win_ms;
    win_end_ms = w * win_ms;
    time_window = sprintf('%d-%d ms', win_start_ms, win_end_ms);

    % Statistics
    F_val = F_values(r, w);
    p_val = p_values(r, w);
    q_val = q_values(r, w);

    % Add row
    new_row = table({roi_name}, {hemisphere}, {time_window}, F_val, p_val, q_val, ...
                    'VariableNames', {'ROI_Name', 'Hemisphere', 'Time_Window', ...
                                      'F_value', 'p_value', 'q_value'});
    TableS1 = [TableS1; new_row];
end

%% SORT BY Q-VALUE
TableS1 = sortrows(TableS1, 'q_value');

%% DISPLAY TOP 20
fprintf('=== Top 20 Significant ROI-Window Pairs ===\n');
disp(head(TableS1, 20));

%% SAVE
% CSV format
writetable(TableS1, fullfile(output_folder, 'TableS1_significant_pairs.csv'));

% Excel format (if available)
try
    writetable(TableS1, fullfile(output_folder, 'TableS1_significant_pairs.xlsx'));
catch
    warning('Excel export not available. CSV saved.');
end

fprintf('\nTable saved to: %s\n', output_folder);
fprintf('=== Step 26 Complete ===\n');
