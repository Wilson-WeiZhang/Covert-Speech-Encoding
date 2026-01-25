%% opensource_step29_lobo_validation.m
% Leave-one-block-out (LOBO) cross-validation analysis
%
% This script corresponds to Supplementary Section S3 in the manuscript.
%
% Method:
%   - Within each subject: train on 9 blocks, test on 1 held-out block
%   - 10-fold CV (10 blocks per subject)
%   - Report per-subject and aggregate accuracy
%
% Note: This is the same LOBO method as Step 11, but provides detailed
% per-block breakdown and statistical validation.
%
% Key Parameters:
%   - Blocks: 10 per subject (alternating O/C)
%   - Folds: 10 (leave-one-block-out)
%   - Features: 148 ROIs x 12 windows = 1776
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
num_windows = 12;
win_ms = 50;

win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends = baseline_samples + round((1:num_windows) * win_ms * fs / 1000);

fprintf('=== Step 29: LOBO Validation ===\n');
fprintf('Method: Leave-one-block-out (10-fold CV within subject)\n\n');

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

fprintf('Found %d subjects\n\n', num_subjects);

%% LOBO CROSS-VALIDATION
% Store results per subject and per block
accuracies_per_subject = zeros(num_subjects, 1);
accuracies_per_block = zeros(num_subjects, num_blocks);

for s = 1:num_subjects
    fprintf('Subject %d/%d: ', s, num_subjects);

    % Load data
    data = load(fullfile(source_folder, source_files(s).name));

    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    condition_data_type = data.condition_data_type(valid_idx);
    n_trials = length(condition_data);

    % Extract features, labels, and block numbers
    features = zeros(n_trials, num_rois * num_windows);
    labels = zeros(n_trials, 1);
    blocks = zeros(n_trials, 1);

    for t = 1:n_trials
        trial_data = condition_data{t};

        % Baseline correction
        baseline = mean(trial_data(:, 1:baseline_samples), 2);
        trial_data = trial_data - baseline;

        % Window features
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

    % LOBO: Leave-one-block-out cross-validation
    predictions = zeros(n_trials, 1);
    block_acc = zeros(num_blocks, 1);

    for fold = 1:num_blocks
        test_idx = (blocks == fold);
        train_idx = ~test_idx;

        if sum(test_idx) == 0 || sum(train_idx) == 0
            continue;
        end

        % Z-score normalization (fit on training, apply to test)
        train_mean = mean(features(train_idx, :), 1);
        train_std = std(features(train_idx, :), 0, 1);
        train_std(train_std == 0) = 1;

        X_train = (features(train_idx, :) - train_mean) ./ train_std;
        X_test = (features(test_idx, :) - train_mean) ./ train_std;
        y_train = labels(train_idx);
        y_test = labels(test_idx);

        % Train linear SVM (multi-class)
        model = fitcecoc(X_train, y_train, 'Learners', 'linear');

        % Predict
        y_pred = predict(model, X_test);
        predictions(test_idx) = y_pred;

        % Per-block accuracy
        block_acc(fold) = 100 * mean(y_pred == y_test);
    end

    % Overall accuracy for this subject
    valid_predictions = predictions > 0 & labels > 0;
    accuracies_per_subject(s) = 100 * mean(predictions(valid_predictions) == labels(valid_predictions));
    accuracies_per_block(s, :) = block_acc;

    fprintf('%.2f%%\n', accuracies_per_subject(s));
end

%% RESULTS SUMMARY
fprintf('\n=== LOBO Results ===\n');
fprintf('Mean accuracy: %.2f%% +/- %.2f%%\n', mean(accuracies_per_subject), std(accuracies_per_subject));
fprintf('Chance level: %.1f%%\n', 100/num_words);

% Statistical test against chance
[~, p_val] = ttest(accuracies_per_subject, 100/num_words);
fprintf('t-test vs chance: p = %.6f\n', p_val);

if p_val < 0.05
    fprintf('Result: SIGNIFICANT (above chance)\n');
else
    fprintf('Result: Not significant\n');
end

% Per-block analysis
fprintf('\n--- Per-Block Accuracy (averaged across subjects) ---\n');
mean_block_acc = mean(accuracies_per_block, 1);
for b = 1:num_blocks
    block_type = mod(b, 2);  % 1=Odd, 0=Even
    if block_type == 1
        type_str = 'Odd ';
    else
        type_str = 'Even';
    end
    fprintf('Block %2d (%s): %.2f%%\n', b, type_str, mean_block_acc(b));
end

% Odd vs Even comparison
odd_blocks = 1:2:10;
even_blocks = 2:2:10;
odd_acc = mean(accuracies_per_block(:, odd_blocks), 2);
even_acc = mean(accuracies_per_block(:, even_blocks), 2);

fprintf('\nOdd blocks mean:  %.2f%% +/- %.2f%%\n', mean(odd_acc), std(odd_acc));
fprintf('Even blocks mean: %.2f%% +/- %.2f%%\n', mean(even_acc), std(even_acc));

[~, p_oe] = ttest(odd_acc, even_acc);
fprintf('Odd vs Even t-test: p = %.4f\n', p_oe);

%% SAVE
lobo_results = struct();
lobo_results.accuracies_per_subject = accuracies_per_subject;
lobo_results.accuracies_per_block = accuracies_per_block;
lobo_results.mean_accuracy = mean(accuracies_per_subject);
lobo_results.std_accuracy = std(accuracies_per_subject);
lobo_results.p_value = p_val;
lobo_results.odd_vs_even_p = p_oe;

save(fullfile(output_folder, 'lobo_results.mat'), 'lobo_results');

fprintf('\n=== Step 29 Complete ===\n');
