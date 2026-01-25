%% opensource_step06_ftest_word_discrimination.m
% F-test for phrase discrimination in source-localized EEG
%
% This script corresponds to Results Section 3.1 and Figure 3 in the manuscript.
%
% NOTE: All datasets are publicly available through NTU's data repository:
%   - Source-localized EEG: https://doi.org/[pending]
%
% Method:
%   - F-test comparing 5 phrase conditions at each ROI-window pair
%   - 148 ROIs (Destrieux atlas) × 30 time windows (50ms, 0-1500ms)
%   - Permutation test (N=1000) for significance
%   - FDR correction (Benjamini-Hochberg)
%
% Key Parameters (from manuscript):
%   - Sampling rate: 250 Hz
%   - Window size: 50 ms (non-overlapping)
%   - Analysis window: 0-1500 ms post-stimulus
%   - Subjects: N = 57
%
% Output:
%   - ftest_results.mat: F-values, p-values, permutation distributions
%
% Related scripts:
%   - Step 07: rmANOVA statistics
%   - Step 08: Figure 3 visualization
%
% Author: Wei Zhang
% Affiliation: Nanyang Technological University
% License: CC BY-NC 4.0
%
%==========================================================================

clear all
clc

%% USER CONFIGURATION - Modify these paths
% -------------------------------------------------------------------------
% Set your data path here. The folder should contain:
%   sourcedata/Subject##_sLORETA_raw.mat (57 files)
% -------------------------------------------------------------------------
data_path = '/path/to/data/';  % USER: Set your data path

% Verify path exists
if ~exist(data_path, 'dir')
    error(['Data path not found: %s\n', ...
           'Please download data from DOI: [pending] and update data_path.'], ...
           data_path);
end

%% ANALYSIS PARAMETERS (fixed for reproducibility)
% -------------------------------------------------------------------------
% These parameters match the published results. Do not modify.
% -------------------------------------------------------------------------

% Data parameters
source_folder = fullfile(data_path, 'sourcedata');
fs = 250;                      % Sampling rate (Hz)
baseline_samples = 0.5 * fs;   % 500ms baseline
num_rois = 148;                % Destrieux atlas
num_phrases = 5;               % 5 command phrases

% Analysis window parameters
window_size_ms = 50;           % 50ms windows
analysis_start_ms = 0;         % Start at stimulus onset
analysis_end_ms = 1500;        % End at 1500ms
num_windows = (analysis_end_ms - analysis_start_ms) / window_size_ms;  % 30 windows

% Statistical parameters
num_permutations = 1000;       % Permutation iterations
alpha_level = 0.05;            % Significance level

fprintf('=== F-test Word Discrimination Analysis ===\n');
fprintf('ROIs: %d (Destrieux atlas)\n', num_rois);
fprintf('Windows: %d × %dms (%d-%dms)\n', num_windows, window_size_ms, ...
        analysis_start_ms, analysis_end_ms);
fprintf('Permutations: %d\n\n', num_permutations);

%% LOAD SUBJECT DATA
% -------------------------------------------------------------------------
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

if num_subjects == 0
    error(['No source files found in: %s\n', ...
           'Expected files: Subject##_sLORETA_raw.mat'], source_folder);
end

fprintf('Found %d subjects\n\n', num_subjects);

% Pre-compute window indices
window_starts = round(analysis_start_ms * fs / 1000) + baseline_samples;
window_samples = round(window_size_ms * fs / 1000);
window_indices = zeros(num_windows, 2);
for w = 1:num_windows
    window_indices(w, 1) = window_starts + (w-1) * window_samples + 1;
    window_indices(w, 2) = window_starts + w * window_samples;
end

%% COMPUTE F-VALUES
% -------------------------------------------------------------------------
F_real = zeros(num_rois, num_windows, num_subjects);
F_perm = zeros(num_rois, num_windows, num_subjects, num_permutations);

for subj = 1:num_subjects
    fprintf('Processing subject %d/%d: %s\n', subj, num_subjects, ...
            source_files(subj).name);

    % Load source-localized data
    data = load(fullfile(source_folder, source_files(subj).name));

    % Remove empty trials
    valid_trials = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_trials);
    condition_labels = data.condition_data_type(valid_trials);

    num_trials = length(condition_data);

    % Extract phrase labels (1-5)
    phrase_labels = zeros(num_trials, 1);
    for t = 1:num_trials
        % Label format: 'C 1_u_1_b_3' or 'O 3_u_1_b_7'
        % Phrase ID is the digit after 'C ' or 'O '
        phrase_labels(t) = str2double(condition_labels{t}(3));
    end

    % Stack trials into 3D matrix: trials × ROIs × time
    trial_data = zeros(num_trials, num_rois, size(condition_data{1}, 2));
    for t = 1:num_trials
        trial_data(t, :, :) = condition_data{t};
    end

    % Compute F-values for each ROI-window pair
    for w = 1:num_windows
        win_start = window_indices(w, 1);
        win_end = window_indices(w, 2);

        % Mean activity in window
        window_activity = mean(trial_data(:, :, win_start:win_end), 3);

        for r = 1:num_rois
            % Real F-value
            F_real(r, w, subj) = compute_fvalue(window_activity(:, r), phrase_labels);

            % Permutation F-values
            for p = 1:num_permutations
                perm_labels = phrase_labels(randperm(num_trials));
                F_perm(r, w, subj, p) = compute_fvalue(window_activity(:, r), perm_labels);
            end
        end
    end
end

%% AGGREGATE AND TEST SIGNIFICANCE
% -------------------------------------------------------------------------
fprintf('\n=== Computing Group-Level Statistics ===\n');

% Average F-values across subjects
F_group_real = mean(F_real, 3);
F_group_perm = squeeze(mean(F_perm, 3));  % num_rois × num_windows × num_perms

% Compute p-values from permutation distribution
p_values = zeros(num_rois, num_windows);
for r = 1:num_rois
    for w = 1:num_windows
        null_dist = squeeze(F_group_perm(r, w, :));
        p_values(r, w) = mean(null_dist >= F_group_real(r, w));
    end
end

% FDR correction (Benjamini-Hochberg)
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
fprintf('Significant ROI-window pairs (FDR q<0.05): %d/%d\n', ...
        sig_pairs, num_rois * num_windows);

%% SAVE RESULTS
% -------------------------------------------------------------------------
output_file = fullfile(data_path, 'results', 'ftest_results.mat');
if ~exist(fullfile(data_path, 'results'), 'dir')
    mkdir(fullfile(data_path, 'results'));
end

save(output_file, 'F_group_real', 'p_values', 'q_values', ...
     'F_real', 'F_perm', 'window_indices', 'num_subjects', ...
     'num_permutations', 'alpha_level', '-v7.3');

fprintf('\n✓ Results saved to: %s\n', output_file);
fprintf('  Variables: F_group_real [%d×%d], p_values, q_values\n', ...
        num_rois, num_windows);

%% HELPER FUNCTION
% -------------------------------------------------------------------------
function F = compute_fvalue(activity, labels)
    % One-way ANOVA F-statistic
    % activity: trials × 1
    % labels: trials × 1 (group labels 1-5)

    groups = unique(labels);
    k = length(groups);
    n = length(activity);

    % Group means
    group_means = zeros(k, 1);
    group_counts = zeros(k, 1);
    for g = 1:k
        idx = labels == groups(g);
        group_means(g) = mean(activity(idx));
        group_counts(g) = sum(idx);
    end
    grand_mean = mean(activity);

    % Between-group variance (SSB)
    SSB = sum(group_counts .* (group_means - grand_mean).^2);

    % Within-group variance (SSW)
    SSW = 0;
    for g = 1:k
        idx = labels == groups(g);
        SSW = SSW + sum((activity(idx) - group_means(g)).^2);
    end

    % F-statistic
    df_between = k - 1;
    df_within = n - k;

    if SSW == 0
        F = 0;
    else
        F = (SSB / df_between) / (SSW / df_within);
    end
end
