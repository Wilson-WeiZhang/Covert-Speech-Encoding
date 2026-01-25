%% opensource_step17_load_eeg_patterns.m
% Load EEG activity patterns for RSA analysis
%
% This script corresponds to Figure 5 in the manuscript.
%
% Method:
%   - Load source-localized EEG data
%   - Extract ROIs corresponding to fMRI clusters
%   - Average activity in analysis window (0-600ms)
%   - Output: 5-word pattern matrix per subject
%
% Output:
%   - eeg_patterns.mat: word patterns per subject for each cluster
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

%% PARAMETERS
fs = 250;
baseline_samples = 125;
analysis_window = [0 600];  % ms
win_start = baseline_samples + round(analysis_window(1) * fs / 1000) + 1;
win_end = baseline_samples + round(analysis_window(2) * fs / 1000);
num_words = 5;

fprintf('=== Step 17: Load EEG Patterns ===\n');
fprintf('Analysis window: %d-%dms\n\n', analysis_window(1), analysis_window(2));

%% LOAD CLUSTER DEFINITIONS
load(fullfile(results_folder, 'fmri_clusters.mat'), 'clusters');
num_clusters = length(clusters);

%% FIND SUBJECTS (exclude those without fMRI)
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));

% Subjects without fMRI data (from manuscript)
exclude_subjects = {'Subject11', 'Subject20', 'Subject23', 'Subject28', ...
                   'Subject48', 'Subject53', 'Subject55', 'Subject56'};

% Filter
valid_files = {};
for i = 1:length(source_files)
    subj_name = source_files(i).name(1:9);
    if ~any(strcmp(subj_name, exclude_subjects))
        valid_files{end+1} = source_files(i).name;
    end
end

num_subjects = length(valid_files);
fprintf('Subjects with both EEG and fMRI: %d\n\n', num_subjects);

%% EXTRACT PATTERNS
eeg_patterns = cell(num_clusters, 1);

for c = 1:num_clusters
    cluster_rois = clusters(c).rois;
    num_cluster_rois = length(cluster_rois);

    fprintf('Cluster %d: %s (%d ROIs)\n', c, clusters(c).name, num_cluster_rois);

    % Pre-allocate: subjects x words x ROIs
    patterns = zeros(num_subjects, num_words, num_cluster_rois);

    for s = 1:num_subjects
        % Load source data
        data = load(fullfile(source_folder, valid_files{s}));

        valid_idx = ~cellfun(@isempty, data.condition_data);
        condition_data = data.condition_data(valid_idx);
        condition_data_type = data.condition_data_type(valid_idx);

        % Accumulate per word
        word_accum = zeros(num_words, num_cluster_rois);
        word_count = zeros(num_words, 1);

        for t = 1:length(condition_data)
            label = condition_data_type{t};
            word_id = str2double(label(3));

            if word_id < 1 || word_id > 5, continue; end

            trial_data = condition_data{t};  % 148 x 500

            % Baseline correction
            baseline = mean(trial_data(:, 1:baseline_samples), 2);
            trial_data = trial_data - baseline;

            % Extract cluster ROIs and average in window
            cluster_activity = mean(trial_data(cluster_rois, win_start:win_end), 2);

            word_accum(word_id, :) = word_accum(word_id, :) + cluster_activity';
            word_count(word_id) = word_count(word_id) + 1;
        end

        % Average
        for w = 1:num_words
            if word_count(w) > 0
                patterns(s, w, :) = word_accum(w, :) / word_count(w);
            end
        end
    end

    eeg_patterns{c} = patterns;
end

%% SAVE
save(fullfile(results_folder, 'eeg_patterns.mat'), 'eeg_patterns', 'clusters', ...
     'valid_files', 'num_subjects', 'analysis_window');

fprintf('\n=== Step 17 Complete ===\n');
