%% opensource_step07_rmanova_statistics.m
% Repeated measures ANOVA for phrase discrimination
%
% This script corresponds to Results Section 3.1 and Figure 3 in the manuscript.
%
% Method:
%   1. Average trials per subject x word (reduce noise)
%   2. rmANOVA: word as within-subject factor (5 levels)
%   3. Test at each ROI x time window pair
%   4. FDR correction (Benjamini-Hochberg)
%
% Key Parameters (from manuscript):
%   - Subjects: N = 57
%   - Words: 5 command phrases
%   - ROIs: 148 (Destrieux atlas)
%   - Windows: 30 x 50ms (0-1500ms)
%   - Correction: BH-FDR (q < 0.05)
%
% Output:
%   - rmanova_results.mat: F-values, p-values, q-values
%
% Author: Wei Zhang
% Affiliation: Nanyang Technological University
% License: CC BY-NC 4.0
%
%==========================================================================

clear all
clc

%% USER CONFIGURATION
data_path = '/path/to/data/';  % USER: Set your data path
source_folder = fullfile(data_path, 'sourcedata');

if ~exist(source_folder, 'dir')
    error('Source folder not found: %s', source_folder);
end

%% PARAMETERS
fs = 250;                      % Sampling rate (Hz)
baseline_samples = 125;        % 0.5s baseline
num_rois = 148;                % Destrieux ROIs
num_words = 5;                 % Phrase conditions
num_windows = 30;              % 50ms windows (0-1500ms)
win_ms = 50;                   % Window size (ms)
alpha_level = 0.05;            % Significance level

% Pre-calculate window boundaries
win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends = baseline_samples + round((1:num_windows) * win_ms * fs / 1000);

fprintf('=== Step 07: rmANOVA Statistics ===\n');
fprintf('ROIs: %d, Windows: %d x %dms\n\n', num_rois, num_windows, win_ms);

%% LOAD DATA
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

fprintf('Found %d subjects\n\n', num_subjects);

% Pre-allocate: subjects x words x windows x rois
activity_data = zeros(num_subjects, num_words, num_windows, num_rois);

for s = 1:num_subjects
    fprintf('Loading subject %d/%d\n', s, num_subjects);

    data = load(fullfile(source_folder, source_files(s).name));

    % Filter valid trials
    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    condition_data_type = data.condition_data_type(valid_idx);

    % Accumulator for each word
    word_accum = zeros(num_words, num_windows, num_rois);
    word_count = zeros(num_words, 1);

    for t = 1:length(condition_data)
        % Extract word label
        label = condition_data_type{t};
        word_id = str2double(label(3));  % 'C 1_u...' -> 1

        if word_id < 1 || word_id > 5
            continue;
        end

        trial_data = condition_data{t};  % 148 x 500

        % Baseline correction
        baseline = mean(trial_data(:, 1:baseline_samples), 2);
        trial_data = trial_data - baseline;

        % Extract windows
        for w = 1:num_windows
            win_mean = mean(trial_data(:, win_starts(w):win_ends(w)), 2);
            word_accum(word_id, w, :) = squeeze(word_accum(word_id, w, :)) + win_mean;
        end
        word_count(word_id) = word_count(word_id) + 1;
    end

    % Average across trials
    for w_id = 1:num_words
        if word_count(w_id) > 0
            activity_data(s, w_id, :, :) = word_accum(w_id, :, :) / word_count(w_id);
        end
    end
end

%% COMPUTE rmANOVA
fprintf('\n--- Computing rmANOVA ---\n');

F_values = zeros(num_rois, num_windows);
p_values = zeros(num_rois, num_windows);

for r = 1:num_rois
    for w = 1:num_windows
        % Extract data: subjects x words
        Y = squeeze(activity_data(:, :, w, r));

        % Compute F-statistic using rmANOVA formula
        grand_mean = mean(Y(:));
        word_means = mean(Y, 1);          % 1 x 5
        subj_means = mean(Y, 2);          % 57 x 1

        % Sum of squares
        SS_word = num_subjects * sum((word_means - grand_mean).^2);
        SS_subj = num_words * sum((subj_means - grand_mean).^2);
        SS_total = sum((Y(:) - grand_mean).^2);
        SS_error = SS_total - SS_word - SS_subj;

        % Degrees of freedom
        df_word = num_words - 1;
        df_error = (num_subjects - 1) * (num_words - 1);

        % F-statistic
        MS_word = SS_word / df_word;
        MS_error = SS_error / df_error;

        if MS_error > 0
            F_values(r, w) = MS_word / MS_error;
            p_values(r, w) = 1 - fcdf(F_values(r, w), df_word, df_error);
        else
            F_values(r, w) = 0;
            p_values(r, w) = 1;
        end
    end
end

%% FDR CORRECTION (Benjamini-Hochberg)
fprintf('Applying FDR correction...\n');

p_flat = p_values(:);
[p_sorted, sort_idx] = sort(p_flat);
m = length(p_flat);
q_values = zeros(m, 1);

for i = 1:m
    q_values(sort_idx(i)) = min(p_sorted(i) * m / i, 1);
end
q_values = reshape(q_values, num_rois, num_windows);

% Count significant pairs
sig_pairs = sum(q_values(:) < alpha_level);
fprintf('\nSignificant ROI-window pairs (FDR q<0.05): %d/%d\n', ...
        sig_pairs, num_rois * num_windows);

%% SAVE RESULTS
output_file = fullfile(data_path, 'results', 'rmanova_results.mat');
if ~exist(fullfile(data_path, 'results'), 'dir')
    mkdir(fullfile(data_path, 'results'));
end

save(output_file, 'F_values', 'p_values', 'q_values', 'activity_data', ...
     'num_subjects', 'num_rois', 'num_windows', '-v7.3');

fprintf('\nResults saved to: %s\n', output_file);
fprintf('=== Step 07 Complete ===\n');
