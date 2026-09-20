%% opensource_step12_lme_data_preparation.m
% Prepare data for Linear Mixed Effects (LME) variance decomposition
%
% This script corresponds to Results Section 3.2 and Figure 4c-e in the manuscript.
%
% Pair selection:
%   The LME analysis is restricted to the ROI-window pairs that reached
%   p < 0.05 in the repeated-measures ANOVA of step 07, limited to the first
%   12 windows (0-600 ms).
%
% Observation unit:
%   One row per single trial (no averaging within subject or phrase), so each
%   pair carries all covert trials of all subjects.
%
% Data structure for LME:
%   - Response: Activity, the baseline-corrected mean ROI amplitude in the window
%   - Fixed effect: WordType (categorical, 5 phrases)
%   - Random effects: Subject intercept, Subject-by-WordType slope
%
% Output:
%   - lme_data_all.mat
%       lme_data_all : cell array, one entry per significant ROI-window pair
%                      .roi .window .window_ms .p_value .num_observations
%                      .data_table (Subject, Trial, WordType, Activity)
%       sig_rois, sig_windows, num_sig_pairs, source_subjects
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
results_folder = fullfile(data_path, 'results');

if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

%% PARAMETERS
fs = 250;
baseline_samples = 125;             % 0.5 s pre-onset baseline
neural_duration_samples = 375;      % 0-1500 ms post onset
num_rois = 148;
num_words = 5;

win_ms = 50;
win_samples = round(win_ms / 1000 * fs);   % 13 samples
num_windows = 12;                          % 0-600 ms
win_starts = (0:num_windows-1) * win_samples + 1;
win_ends = (1:num_windows) * win_samples;

fprintf('=== Step 12: LME Data Preparation ===\n');

%% SELECT SIGNIFICANT ROI-WINDOW PAIRS (rmANOVA, step 07)
load(fullfile(results_folder, 'rmanova_results.mat'), 'p_values');
p_values_trimmed = p_values(:, 1:num_windows);

[sig_rois, sig_windows] = find(p_values_trimmed < 0.05);
num_sig_pairs = length(sig_rois);
sig_linear = sub2ind([num_rois, num_windows], sig_rois, sig_windows);

fprintf('ROI-window pairs tested: %d\n', num_rois * num_windows);
fprintf('Significant pairs (rmANOVA p < 0.05): %d\n', num_sig_pairs);

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);
source_subjects = {source_files.name};

fprintf('Subjects: %d\n\n', num_subjects);

%% EXTRACT SINGLE-TRIAL ACTIVITY FOR EACH SIGNIFICANT PAIR
max_obs = num_subjects * 100;
obs_subject  = zeros(max_obs, 1);
obs_trial    = zeros(max_obs, 1);
obs_word     = zeros(max_obs, 1);
obs_activity = zeros(max_obs, num_sig_pairs);
obs_count = 0;

for s = 1:num_subjects
    fprintf('Subject %d/%d\n', s, num_subjects);

    data = load(fullfile(source_folder, source_files(s).name));
    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    condition_data_type = data.condition_data_type(valid_idx);

    for t = 1:length(condition_data)
        label = condition_data_type{t};
        word_id = str2double(label(3));
        if isnan(word_id) || word_id < 1 || word_id > num_words, continue; end

        trial_data = condition_data{t};                 % 148 x 500
        if size(trial_data, 1) < num_rois, continue; end

        baseline = mean(trial_data(1:num_rois, 1:baseline_samples), 2);
        event_data = trial_data(1:num_rois, baseline_samples + (1:neural_duration_samples));

        win_means = zeros(num_rois, num_windows);
        for w = 1:num_windows
            win_means(:, w) = mean(event_data(:, win_starts(w):win_ends(w)), 2) - baseline;
        end

        obs_count = obs_count + 1;
        obs_subject(obs_count)  = s;
        obs_trial(obs_count)    = t;
        obs_word(obs_count)     = word_id;
        obs_activity(obs_count, :) = win_means(sig_linear)';
    end
end

obs_subject  = obs_subject(1:obs_count);
obs_trial    = obs_trial(1:obs_count);
obs_word     = obs_word(1:obs_count);
obs_activity = obs_activity(1:obs_count, :);

fprintf('\nSingle-trial observations per pair: %d\n', obs_count);

%% BUILD ONE TABLE PER PAIR
lme_data_all = cell(num_sig_pairs, 1);

Subject = categorical(obs_subject);
WordType = categorical(obs_word);

for pair_idx = 1:num_sig_pairs
    temp_data = struct();
    temp_data.roi = sig_rois(pair_idx);
    temp_data.window = sig_windows(pair_idx);
    temp_data.window_ms = [(sig_windows(pair_idx)-1)*win_ms, sig_windows(pair_idx)*win_ms];
    temp_data.p_value = p_values_trimmed(sig_rois(pair_idx), sig_windows(pair_idx));
    temp_data.num_observations = obs_count;
    temp_data.subject_ids = source_subjects;

    Trial = obs_trial;
    Activity = obs_activity(:, pair_idx);
    temp_data.data_table = table(Subject, Trial, WordType, Activity);

    lme_data_all{pair_idx} = temp_data;
end

%% SAVE
save(fullfile(results_folder, 'lme_data_all.mat'), 'lme_data_all', ...
     'sig_rois', 'sig_windows', 'num_sig_pairs', 'num_rois', 'num_windows', ...
     'num_subjects', 'source_subjects', 'win_ms', '-v7.3');

fprintf('Saved %d pairs to lme_data_all.mat\n', num_sig_pairs);
fprintf('=== Step 12 Complete ===\n');
