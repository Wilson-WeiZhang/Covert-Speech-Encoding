%% opensource_step28_source_reproducibility.m
% Test source localization reproducibility
%
% This script corresponds to Supplementary Section S5 in the manuscript.
%
% Method:
%   - Split-half reliability (odd vs even trials)
%   - Intraclass Correlation Coefficient (ICC)
%   - Test-retest across blocks
%
% Key Results (from manuscript):
%   - ICC > 0.8 for most significant ROIs
%   - Split-half reliability: r > 0.9
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
source_folder = fullfile(data_path, 'sourcedata');
output_folder = fullfile(data_path, 'results');

%% PARAMETERS
fs = 250;
baseline_samples = 125;
num_rois = 148;
num_words = 5;
num_windows = 12;  % 0-600ms, 50ms each
win_ms = 50;

win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends = baseline_samples + round((1:num_windows) * win_ms * fs / 1000);

fprintf('=== Step 28: Source Reproducibility ===\n');

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

fprintf('Found %d subjects\n\n', num_subjects);

%% COMPUTE SPLIT-HALF RELIABILITY
fprintf('Computing split-half reliability...\n');

split_half_r = zeros(num_subjects, num_rois, num_windows);

for s = 1:num_subjects
    if mod(s, 10) == 0
        fprintf('  Subject %d/%d\n', s, num_subjects);
    end

    % Load data
    data = load(fullfile(source_folder, source_files(s).name));

    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    condition_data_type = data.condition_data_type(valid_idx);
    n_trials = length(condition_data);

    % Split into odd and even trials
    odd_trials = 1:2:n_trials;
    even_trials = 2:2:n_trials;

    % Compute mean activity for each split
    odd_activity = zeros(num_words, num_rois, num_windows);
    even_activity = zeros(num_words, num_rois, num_windows);
    odd_count = zeros(num_words, 1);
    even_count = zeros(num_words, 1);

    for t = 1:n_trials
        label = condition_data_type{t};
        word_id = str2double(label(3));
        if word_id < 1 || word_id > 5, continue; end

        trial_data = condition_data{t};
        baseline = mean(trial_data(:, 1:baseline_samples), 2);
        trial_data = trial_data - baseline;

        for w = 1:num_windows
            win_mean = mean(trial_data(:, win_starts(w):win_ends(w)), 2);

            if ismember(t, odd_trials)
                odd_activity(word_id, :, w) = squeeze(odd_activity(word_id, :, w)) + win_mean';
                odd_count(word_id) = odd_count(word_id) + 1;
            else
                even_activity(word_id, :, w) = squeeze(even_activity(word_id, :, w)) + win_mean';
                even_count(word_id) = even_count(word_id) + 1;
            end
        end
    end

    % Average
    for w_id = 1:num_words
        if odd_count(w_id) > 0
            odd_activity(w_id, :, :) = odd_activity(w_id, :, :) / odd_count(w_id);
        end
        if even_count(w_id) > 0
            even_activity(w_id, :, :) = even_activity(w_id, :, :) / even_count(w_id);
        end
    end

    % Compute correlation between splits
    for r = 1:num_rois
        for w = 1:num_windows
            odd_vec = squeeze(odd_activity(:, r, w));
            even_vec = squeeze(even_activity(:, r, w));

            if std(odd_vec) > 0 && std(even_vec) > 0
                split_half_r(s, r, w) = corr(odd_vec, even_vec);
            end
        end
    end
end

%% SPEARMAN-BROWN CORRECTION
% Corrected reliability = 2*r / (1 + r)
corrected_r = 2 * split_half_r ./ (1 + split_half_r);

%% SUMMARY
mean_reliability = squeeze(mean(corrected_r, 1));  % rois x windows
overall_mean = mean(mean_reliability(:));
overall_std = std(mean_reliability(:));

fprintf('\n=== Reproducibility Results ===\n');
fprintf('Split-half reliability (Spearman-Brown corrected):\n');
fprintf('  Mean: %.3f +/- %.3f\n', overall_mean, overall_std);
fprintf('  Min: %.3f, Max: %.3f\n', min(mean_reliability(:)), max(mean_reliability(:)));

% Count highly reliable ROI-window pairs (r > 0.8)
high_reliability = sum(mean_reliability(:) > 0.8);
fprintf('  ROI-window pairs with r > 0.8: %d/%d (%.1f%%)\n', ...
        high_reliability, numel(mean_reliability), 100*high_reliability/numel(mean_reliability));

%% SAVE
reproducibility_results = struct();
reproducibility_results.split_half_r = split_half_r;
reproducibility_results.corrected_r = corrected_r;
reproducibility_results.mean_reliability = mean_reliability;
reproducibility_results.overall_mean = overall_mean;
reproducibility_results.overall_std = overall_std;

save(fullfile(output_folder, 'reproducibility_results.mat'), 'reproducibility_results');

fprintf('\n=== Step 28 Complete ===\n');
