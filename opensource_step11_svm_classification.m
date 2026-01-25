%% opensource_step11_svm_classification.m
% 5-class SVM classification for phrase discrimination
%
% This script corresponds to Results Section 3.2 and Figure 4a-b in the manuscript.
%
% Method:
%   - Feature: Mean ROI activity per 50ms window (148 ROIs x 12 windows = 1776 features)
%   - Classifier: Linear SVM (one-vs-all)
%   - Validation: Leave-one-block-out cross-validation (10 blocks)
%   - Window: 0-600ms post-stimulus
%
% Key Results (from manuscript):
%   - Overall accuracy: 27.26% +/- 6.97% (chance = 20%)
%   - Peak window: 300-400ms
%
% Output:
%   - classification_results.mat: accuracy per subject, confusion matrix
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

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
fs = 250;
baseline_samples = 125;
num_rois = 148;
num_words = 5;
num_blocks = 10;

% Analysis window: 0-600ms (12 x 50ms windows)
win_ms = 50;
num_windows = 12;
win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends = baseline_samples + round((1:num_windows) * win_ms * fs / 1000);

fprintf('=== Step 11: SVM Classification ===\n');
fprintf('Features: %d ROIs x %d windows = %d\n', num_rois, num_windows, num_rois * num_windows);
fprintf('Validation: Leave-one-block-out (%d blocks)\n\n', num_blocks);

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

fprintf('Found %d subjects\n\n', num_subjects);

%% CLASSIFY EACH SUBJECT
accuracies = zeros(num_subjects, 1);
confusion_all = zeros(num_words, num_words, num_subjects);

for s = 1:num_subjects
    fprintf('Subject %d/%d: ', s, num_subjects);

    % Load data
    data = load(fullfile(source_folder, source_files(s).name));

    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    condition_data_type = data.condition_data_type(valid_idx);
    n_trials = length(condition_data);

    % Extract features and labels
    features = zeros(n_trials, num_rois * num_windows);
    labels = zeros(n_trials, 1);
    blocks = zeros(n_trials, 1);

    for t = 1:n_trials
        trial_data = condition_data{t};  % 148 x 500

        % Baseline correction
        baseline = mean(trial_data(:, 1:baseline_samples), 2);
        trial_data = trial_data - baseline;

        % Extract window features
        feat_vec = zeros(num_rois, num_windows);
        for w = 1:num_windows
            feat_vec(:, w) = mean(trial_data(:, win_starts(w):win_ends(w)), 2);
        end
        features(t, :) = feat_vec(:)';

        % Parse label: 'C 1_u_1_b_3'
        label = condition_data_type{t};
        labels(t) = str2double(label(3));  % Word ID

        % Extract block number
        block_match = regexp(label, 'b_(\d+)', 'tokens');
        if ~isempty(block_match)
            blocks(t) = str2double(block_match{1}{1});
        end
    end

    % Leave-one-block-out cross-validation
    predictions = zeros(n_trials, 1);

    for fold = 1:num_blocks
        test_idx = (blocks == fold);
        train_idx = ~test_idx;

        if sum(test_idx) == 0 || sum(train_idx) == 0
            continue;
        end

        % Z-score normalization
        train_mean = mean(features(train_idx, :), 1);
        train_std = std(features(train_idx, :), 0, 1);
        train_std(train_std == 0) = 1;

        X_train = (features(train_idx, :) - train_mean) ./ train_std;
        X_test = (features(test_idx, :) - train_mean) ./ train_std;
        y_train = labels(train_idx);
        y_test = labels(test_idx);

        % Train linear SVM
        model = fitcecoc(X_train, y_train, 'Learners', 'linear');

        % Predict
        predictions(test_idx) = predict(model, X_test);
    end

    % Calculate accuracy
    correct = (predictions == labels);
    accuracies(s) = 100 * sum(correct) / n_trials;

    % Confusion matrix
    for i = 1:n_trials
        if predictions(i) > 0 && labels(i) > 0
            confusion_all(labels(i), predictions(i), s) = ...
                confusion_all(labels(i), predictions(i), s) + 1;
        end
    end

    fprintf('%.2f%%\n', accuracies(s));
end

%% SUMMARY
fprintf('\n=== Classification Results ===\n');
fprintf('Mean accuracy: %.2f%% +/- %.2f%%\n', mean(accuracies), std(accuracies));
fprintf('Chance level: %.1f%%\n', 100/num_words);

% Statistical test against chance
[~, p_val] = ttest(accuracies, 100/num_words);
fprintf('t-test vs chance: p = %.4f\n', p_val);

%% SAVE RESULTS
save(fullfile(output_folder, 'classification_results.mat'), ...
     'accuracies', 'confusion_all', 'num_subjects', 'num_rois', 'num_windows');

fprintf('\n=== Step 11 Complete ===\n');
