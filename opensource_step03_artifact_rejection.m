%% opensource_step03_artifact_rejection.m
% Stage 3: Remove ICA artifact components and bad channels
%
% This script corresponds to Methods Section 2.2 in the manuscript.
%
% Processing steps:
%   1. Load ICA-decomposed EEG from Step 02
%   2. Identify artifact components using ICLabel probability threshold
%   3. Remove artifact components (muscle, eye, heart, line noise, channel noise)
%   4. Keep only brain components
%   5. Reject bad channels and interpolate
%   6. Re-reference to average
%
% Key Parameters (from manuscript):
%   - ICLabel threshold: 0.9 (>90% probability for artifact rejection)
%   - Brain probability threshold: <0.1 (exclude ambiguous components)
%   - Final channels: 56 (after removing bad channels)
%
% Output:
%   - S00##_*_rejectchan56_u1.set/.fdt
%
% Requirements:
%   - EEGLAB 2024 with ICLabel plugin
%
% Author: Wei Zhang
% Affiliation: Nanyang Technological University
% License: CC BY-NC 4.0
%
%==========================================================================

clear all
clc
close all

%% USER CONFIGURATION - Modify these paths
% -------------------------------------------------------------------------
data_path = '/path/to/data/';  % USER: Set your data path
eeg_folder = fullfile(data_path, 'preprocessed_eeg');

% Verify path exists
if ~exist(eeg_folder, 'dir')
    error('Preprocessed EEG folder not found: %s', eeg_folder);
end

%% INITIALIZE EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

%% PARAMETERS
ic_threshold = 0.9;        % ICLabel threshold for artifact classification
brain_threshold = 0.1;     % Exclude components with brain probability < 0.1

% Standard 62-channel layout (BrainVision)
standard_channels = {
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', ...
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', ...
    'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', ...
    'TP9', 'TP10', 'POz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', ...
    'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', ...
    'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', ...
    'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', ...
    'Fpz', 'CPz'
};

%% FIND SUBJECTS
file_list = dir(fullfile(eeg_folder, '*_precut_ICA.set'));
num_subjects = length(file_list);

fprintf('=== Step 03: Artifact Rejection ===\n');
fprintf('Found %d subjects\n\n', num_subjects);

%% PROCESS EACH SUBJECT
for subj = 1:num_subjects
    fprintf('Processing subject %d/%d\n', subj, num_subjects);

    filename = file_list(subj).name;
    subj_id = filename(1:5);

    % Load ICA-decomposed EEG
    EEG = pop_loadset('filename', filename, 'filepath', eeg_folder);

    % Get ICLabel classifications
    ic_class = EEG.etc.ic_classification.ICLabel.classifications;
    % Columns: 1=Brain, 2=Muscle, 3=Eye, 4=Heart, 5=Line, 6=Channel, 7=Other

    % Identify artifact components (>90% probability)
    artifact_muscle = find(ic_class(:, 2) > ic_threshold)';
    artifact_eye = find(ic_class(:, 3) > ic_threshold)';
    artifact_heart = find(ic_class(:, 4) > ic_threshold)';
    artifact_line = find(ic_class(:, 5) > ic_threshold)';
    artifact_channel = find(ic_class(:, 6) > ic_threshold)';

    % If no heart component found with high confidence, take most likely
    if isempty(artifact_heart)
        [~, artifact_heart] = max(ic_class(1:min(5, size(ic_class,1)), 4));
    end

    % Combine all artifact components
    all_artifacts = unique([artifact_muscle, artifact_eye, artifact_heart, ...
                           artifact_line, artifact_channel]);

    % Identify brain components (exclude artifacts AND low brain probability)
    all_components = 1:size(EEG.icawinv, 2);
    brain_components = setdiff(all_components, all_artifacts);

    % Further exclude components with brain probability < 0.1
    low_brain_prob = find(ic_class(:, 1) < brain_threshold);
    low_brain_prob = setdiff(low_brain_prob, all_artifacts);
    brain_components = setdiff(brain_components, low_brain_prob');

    % Remove artifact components
    components_to_remove = setdiff(all_components, brain_components);
    EEG = pop_subcomp(EEG, components_to_remove, 0);

    fprintf('  Removed %d artifact components, kept %d brain components\n', ...
            length(components_to_remove), length(brain_components));

    % Detect and interpolate bad channels
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, 'ChannelCriterion', 0.8, ...
        'LineNoiseCriterion', 4, 'Highpass', 'off', 'BurstCriterion', 'off', ...
        'WindowCriterion', 'off', 'BurstRejection', 'off', 'Distance', 'Euclidian');

    % Re-reference to average
    EEG = pop_reref(EEG, []);

    % Save cleaned data
    output_filename = [subj_id '_Filters_processed_trials_rejectchan56_u1.set'];
    EEG = pop_saveset(EEG, 'filename', output_filename, 'filepath', eeg_folder);

    fprintf('  Saved: %s (%d channels)\n', output_filename, EEG.nbchan);
end

fprintf('\n=== Step 03 Complete ===\n');
fprintf('Artifact rejection complete. Proceed to Step 04 for source localization.\n');
