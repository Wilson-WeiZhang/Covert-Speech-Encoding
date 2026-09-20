%% opensource_step17_load_eeg_patterns.m
% Load EEG activity patterns for RSA analysis
%
% This script corresponds to Figure 5 in the manuscript.
%
% Method:
%   - Load source-localized EEG data of the participants who also have fMRI
%   - Keep the covert trials of the first utterance of each phrase
%   - Extract the EEG ROIs of each cluster in the analysis window
%     (0-600 ms, 150 samples at 250 Hz)
%   - Average across trials and then across the ROIs of the cluster, so the
%     pattern of a phrase is its ROI-averaged time course
%   - Output: [5 phrases x 150 time points] per participant and cluster
%
% Output:
%   - eeg_patterns.mat: phrase patterns per participant for each cluster
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
num_time_samples = win_end - win_start + 1;
num_words = 5;

% Covert trials are labelled 'C <phrase>', overt trials 'O <phrase>';
% '_u_1_' selects the first utterance of the phrase within the trial
word_codes = {'C 1', 'C 2', 'C 3', 'C 4', 'C 5'};
first_utterance = '_u_1_';

% Participants excluded because they have no fMRI session
no_fmri_subjects = [11, 20, 23, 27, 46, 51, 53, 54];

fprintf('=== Step 17: Load EEG Patterns ===\n');
fprintf('Analysis window: %d-%dms (%d samples)\n\n', ...
        analysis_window(1), analysis_window(2), num_time_samples);

%% LOAD CLUSTER DEFINITIONS
load(fullfile(results_folder, 'fmri_clusters.mat'), 'clusters');
num_clusters = length(clusters);

%% FIND PARTICIPANTS WITH BOTH EEG AND fMRI
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));

valid_files = {};
for i = 1:length(source_files)
    % 'Subject11_sLORETA_raw.mat' -> 11
    subj_num = str2double(source_files(i).name(8:9));
    if ~ismember(subj_num, no_fmri_subjects)
        valid_files{end+1} = source_files(i).name;
    end
end

num_subjects = length(valid_files);
fprintf('Participants with both EEG and fMRI: %d\n\n', num_subjects);

%% EXTRACT PATTERNS
eeg_patterns = cell(num_clusters, 1);
for c = 1:num_clusters
    eeg_patterns{c} = zeros(num_subjects, num_words, num_time_samples);
end

for s = 1:num_subjects
    data = load(fullfile(source_folder, valid_files{s}), ...
                'condition_data', 'condition_data_type');

    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    condition_data_type = data.condition_data_type(valid_idx);

    if isempty(condition_data)
        continue;
    end

    for c = 1:num_clusters
        cluster_rois = clusters(c).rois;

        for w = 1:num_words
            trial_idx = find(contains(condition_data_type, word_codes{w}) & ...
                             contains(condition_data_type, first_utterance));

            if isempty(trial_idx)
                continue;
            end

            % [trials x ROIs x time]
            trial_series = zeros(length(trial_idx), length(cluster_rois), ...
                                 num_time_samples);

            for t = 1:length(trial_idx)
                trial_data = condition_data{trial_idx(t)};
                if size(trial_data, 1) < max(cluster_rois)
                    continue;
                end
                trial_series(t, :, :) = trial_data(cluster_rois, win_start:win_end);
            end

            % Average across trials, then across the ROIs of the cluster
            roi_series = reshape(mean(trial_series, 1), ...
                                 length(cluster_rois), num_time_samples);
            eeg_patterns{c}(s, w, :) = mean(roi_series, 1);
        end
    end

    if mod(s, 10) == 0
        fprintf('  Participant %d/%d\n', s, num_subjects);
    end
end

%% SAVE
save(fullfile(results_folder, 'eeg_patterns.mat'), 'eeg_patterns', 'clusters', ...
     'valid_files', 'num_subjects', 'analysis_window');

fprintf('\n=== Step 17 Complete ===\n');
