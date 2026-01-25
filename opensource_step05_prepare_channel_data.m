%% opensource_step05_prepare_channel_data.m
% Prepare channel-level EEG data for comparison analyses
%
% This script corresponds to Supplementary Section S3 in the manuscript.
%
% Processing steps:
%   1. Load preprocessed EEG from Step 03
%   2. Extract channel data in trial format
%   3. Organize by phrase condition (1-5)
%   4. Save in format matching source-localized data
%
% Key Parameters:
%   - Channels: 56 (after bad channel rejection)
%   - Epoch: -0.5s to +1.5s per word (500 samples @ 250Hz)
%   - Phrases: 5 command phrases
%
% Output:
%   - Subject##_channel_raw.mat
%     Variables: condition_data (56 x 500), condition_data_type
%
% Note: This data is used for channel-level vs source-level comparison
% in Supplementary Section S3.
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
data_path = '/path/to/data/';  % USER: Set your data path
eeg_folder = fullfile(data_path, 'preprocessed_eeg');
output_folder = fullfile(data_path, 'channeldata');

% Verify paths exist
if ~exist(eeg_folder, 'dir')
    error('Preprocessed EEG folder not found: %s', eeg_folder);
end
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
trial_start = 1;               % First sample
trial_end = 500;               % Last sample (2s @ 250Hz)

fprintf('=== Step 05: Prepare Channel Data ===\n');

%% FIND SUBJECTS
file_list = dir(fullfile(eeg_folder, '*_rejectchan56_u1.set'));
num_subjects = length(file_list);

fprintf('Found %d subjects\n\n', num_subjects);

%% PROCESS EACH SUBJECT
for subj = 1:num_subjects
    fprintf('Processing subject %d/%d\n', subj, num_subjects);

    filename = file_list(subj).name;
    subj_id = filename(1:5);

    % Load preprocessed EEG
    EEG = pop_loadset('filename', filename, 'filepath', eeg_folder);

    % Initialize output
    num_trials = EEG.trials;
    condition_data = cell(num_trials, 1);
    condition_data_type = cell(num_trials, 1);

    % Extract channel data for each trial
    for trial_idx = 1:num_trials
        % Extract trial data (channels x time)
        trial_data = EEG.data(:, trial_start:trial_end, trial_idx);

        % Store
        condition_data{trial_idx} = trial_data;
        condition_data_type{trial_idx} = EEG.event(trial_idx).type;
    end

    % Channel info
    chanlocs = EEG.chanlocs;

    % Save channel data
    output_file = fullfile(output_folder, [subj_id '_channel_raw.mat']);
    save(output_file, 'condition_data', 'condition_data_type', 'chanlocs', '-v7.3');

    fprintf('  Saved: %s (%d trials, %d channels)\n', ...
            output_file, num_trials, EEG.nbchan);
end

fprintf('\n=== Step 05 Complete ===\n');
fprintf('Channel data saved to: %s\n', output_folder);
