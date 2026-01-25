%% opensource_step10_band_comparison.m
% Compare phrase discrimination across frequency bands
%
% This script corresponds to Supplementary Section S2 in the manuscript.
%
% Analysis:
%   - Run rmANOVA on each frequency band
%   - Compare number of significant ROI-window pairs
%   - Test whether broadband outperforms individual bands
%
% Key Results (from manuscript):
%   - Broadband (1-100Hz) > any single band
%   - Theta band shows strongest individual contribution
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
filtered_base = fullfile(data_path, 'sourcedata_filtered');
results_folder = fullfile(data_path, 'results');
output_folder = fullfile(data_path, 'figures');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
bands = {'raw', 'delta', 'theta', 'alpha', 'beta'};
band_labels = {'Broadband', 'Delta (1-4Hz)', 'Theta (4-8Hz)', ...
               'Alpha (8-13Hz)', 'Beta (13-30Hz)'};

fs = 250;
baseline_samples = 125;
num_rois = 148;
num_words = 5;
num_windows = 30;
win_ms = 50;
alpha_level = 0.05;

win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends = baseline_samples + round((1:num_windows) * win_ms * fs / 1000);

fprintf('=== Step 10: Band Comparison ===\n');

%% ANALYZE EACH BAND
sig_counts = zeros(length(bands), 1);

for b = 1:length(bands)
    band_name = bands{b};
    fprintf('\n--- Analyzing %s ---\n', band_labels{b});

    % Set source folder
    if strcmp(band_name, 'raw')
        source_folder = fullfile(data_path, 'sourcedata');
        file_pattern = 'Subject*_sLORETA_raw.mat';
    else
        source_folder = fullfile(filtered_base, band_name);
        file_pattern = ['Subject*_sLORETA_' band_name '.mat'];
    end

    source_files = dir(fullfile(source_folder, file_pattern));
    num_subjects = length(source_files);

    if num_subjects == 0
        warning('No files found for %s', band_name);
        continue;
    end

    % Load and average data
    activity_data = zeros(num_subjects, num_words, num_windows, num_rois);

    for s = 1:num_subjects
        data = load(fullfile(source_folder, source_files(s).name));

        valid_idx = ~cellfun(@isempty, data.condition_data);
        condition_data = data.condition_data(valid_idx);
        condition_data_type = data.condition_data_type(valid_idx);

        word_accum = zeros(num_words, num_windows, num_rois);
        word_count = zeros(num_words, 1);

        for t = 1:length(condition_data)
            label = condition_data_type{t};
            word_id = str2double(label(3));

            if word_id < 1 || word_id > 5, continue; end

            trial_data = condition_data{t};
            baseline = mean(trial_data(:, 1:baseline_samples), 2);
            trial_data = trial_data - baseline;

            for w = 1:num_windows
                win_mean = mean(trial_data(:, win_starts(w):win_ends(w)), 2);
                word_accum(word_id, w, :) = squeeze(word_accum(word_id, w, :)) + win_mean;
            end
            word_count(word_id) = word_count(word_id) + 1;
        end

        for w_id = 1:num_words
            if word_count(w_id) > 0
                activity_data(s, w_id, :, :) = word_accum(w_id, :, :) / word_count(w_id);
            end
        end
    end

    % Compute rmANOVA F-values and p-values
    p_values = zeros(num_rois, num_windows);

    for r = 1:num_rois
        for w = 1:num_windows
            Y = squeeze(activity_data(:, :, w, r));
            grand_mean = mean(Y(:));
            word_means = mean(Y, 1);
            subj_means = mean(Y, 2);

            SS_word = num_subjects * sum((word_means - grand_mean).^2);
            SS_subj = num_words * sum((subj_means - grand_mean).^2);
            SS_total = sum((Y(:) - grand_mean).^2);
            SS_error = SS_total - SS_word - SS_subj;

            df_word = num_words - 1;
            df_error = (num_subjects - 1) * (num_words - 1);

            if SS_error > 0
                F = (SS_word / df_word) / (SS_error / df_error);
                p_values(r, w) = 1 - fcdf(F, df_word, df_error);
            else
                p_values(r, w) = 1;
            end
        end
    end

    % FDR correction
    p_flat = p_values(:);
    [p_sorted, sort_idx] = sort(p_flat);
    m = length(p_flat);
    q_values = zeros(m, 1);
    for i = 1:m
        q_values(sort_idx(i)) = min(p_sorted(i) * m / i, 1);
    end
    q_values = reshape(q_values, num_rois, num_windows);

    sig_counts(b) = sum(q_values(:) < alpha_level);
    fprintf('  Significant pairs: %d/%d\n', sig_counts(b), numel(q_values));
end

%% PLOT COMPARISON
figure('Position', [100 100 800 400]);

bar(sig_counts, 'FaceColor', [0.3 0.6 0.9]);
set(gca, 'XTickLabel', band_labels);
ylabel('Significant ROI-Window Pairs (FDR q<0.05)');
title('Supplementary Figure S2: Frequency Band Comparison');

% Add value labels
for b = 1:length(bands)
    text(b, sig_counts(b) + 10, num2str(sig_counts(b)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold');
end

saveas(gcf, fullfile(output_folder, 'sup_figure_s2_band_comparison.png'));

fprintf('\n=== Step 10 Complete ===\n');
