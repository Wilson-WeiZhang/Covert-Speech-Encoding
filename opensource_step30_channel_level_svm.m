%% opensource_step30_channel_level_svm.m
% Channel-level SVM classification for comparison
%
% This script corresponds to Supplementary Section S3 in the manuscript.
%
% Method:
%   - Same classification pipeline as Step 11
%   - Use channel data instead of source-localized data
%   - Compare accuracy: channel vs source
%
% Key Results (from manuscript):
%   - Source-level > Channel-level accuracy
%   - Demonstrates value of source localization
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
channel_folder = fullfile(data_path, 'channeldata');
results_folder = fullfile(data_path, 'results');

if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

%% PARAMETERS
fs = 250;
baseline_samples = 125;
num_words = 5;
num_blocks = 10;
num_windows = 12;
win_ms = 50;

win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends = baseline_samples + round((1:num_windows) * win_ms * fs / 1000);

fprintf('=== Step 30: Channel-Level SVM ===\n');

%% FIND SUBJECTS
channel_files = dir(fullfile(channel_folder, '*_channel_raw.mat'));
num_subjects = length(channel_files);

fprintf('Found %d subjects\n\n', num_subjects);

%% CLASSIFY EACH SUBJECT
accuracies_channel = zeros(num_subjects, 1);

for s = 1:num_subjects
    fprintf('Subject %d/%d: ', s, num_subjects);

    % Load channel data
    data = load(fullfile(channel_folder, channel_files(s).name));

    condition_data = data.condition_data;
    condition_data_type = data.condition_data_type;
    n_trials = length(condition_data);
    num_channels = size(condition_data{1}, 1);

    % Extract features
    features = zeros(n_trials, num_channels * num_windows);
    labels = zeros(n_trials, 1);
    blocks = zeros(n_trials, 1);

    for t = 1:n_trials
        if isempty(condition_data{t})
            continue;
        end

        trial_data = condition_data{t};  % channels x time

        % Baseline correction
        baseline = mean(trial_data(:, 1:baseline_samples), 2);
        trial_data = trial_data - baseline;

        % Window features
        feat_vec = zeros(num_channels, num_windows);
        for w = 1:num_windows
            feat_vec(:, w) = mean(trial_data(:, win_starts(w):win_ends(w)), 2);
        end
        features(t, :) = feat_vec(:)';

        % Labels
        label = condition_data_type{t};
        labels(t) = str2double(label(3));

        block_match = regexp(label, 'b_(\d+)', 'tokens');
        if ~isempty(block_match)
            blocks(t) = str2double(block_match{1}{1});
        end
    end

    % Remove invalid
    valid = labels >= 1 & labels <= num_words;
    features = features(valid, :);
    labels = labels(valid);
    blocks = blocks(valid);

    % Leave-one-block-out CV
    predictions = zeros(length(labels), 1);

    for fold = 1:num_blocks
        test_idx = (blocks == fold);
        train_idx = ~test_idx;

        if sum(test_idx) == 0 || sum(train_idx) == 0
            continue;
        end

        % Normalize
        train_mean = mean(features(train_idx, :), 1);
        train_std = std(features(train_idx, :), 0, 1);
        train_std(train_std == 0) = 1;

        X_train = (features(train_idx, :) - train_mean) ./ train_std;
        X_test = (features(test_idx, :) - train_mean) ./ train_std;

        % Train
        model = fitcecoc(X_train, labels(train_idx), 'Learners', 'linear');

        % Predict
        predictions(test_idx) = predict(model, X_test);
    end

    accuracies_channel(s) = 100 * mean(predictions == labels);
    fprintf('%.2f%%\n', accuracies_channel(s));
end

%% COMPARE WITH SOURCE-LEVEL
load(fullfile(results_folder, 'classification_results.mat'), 'accuracies');
accuracies_source = accuracies;

fprintf('\n=== Comparison Results ===\n');
fprintf('Channel-level: %.2f%% +/- %.2f%%\n', mean(accuracies_channel), std(accuracies_channel));
fprintf('Source-level:  %.2f%% +/- %.2f%%\n', mean(accuracies_source), std(accuracies_source));

% Paired t-test
[~, p_val] = ttest(accuracies_source, accuracies_channel);
fprintf('\nPaired t-test (Source vs Channel): p = %.4f\n', p_val);

if p_val < 0.05
    if mean(accuracies_source) > mean(accuracies_channel)
        fprintf('Result: Source > Channel (significant)\n');
    else
        fprintf('Result: Channel > Source (significant)\n');
    end
else
    fprintf('Result: No significant difference\n');
end

%% SAVE
channel_results = struct();
channel_results.accuracies = accuracies_channel;
channel_results.mean_accuracy = mean(accuracies_channel);
channel_results.std_accuracy = std(accuracies_channel);
channel_results.comparison_p = p_val;

save(fullfile(results_folder, 'channel_svm_results.mat'), 'channel_results');

fprintf('\n=== Step 30 Complete ===\n');
