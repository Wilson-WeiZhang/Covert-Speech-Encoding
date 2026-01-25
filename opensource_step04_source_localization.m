%% opensource_step04_source_localization.m
% Source localization using sLORETA with Destrieux atlas
%
% This script corresponds to Methods Section 2.3 in the manuscript.
%
% NOTE: Source localization was performed in Brainstorm. This script
% extracts ROI time series from Brainstorm output files.
%
% Processing steps:
%   1. Load Brainstorm imaging kernel (sLORETA)
%   2. Load Destrieux atlas ROI definitions (148 ROIs)
%   3. Apply kernel to preprocessed EEG data
%   4. Average source activity within each ROI
%   5. Save ROI time series per trial
%
% Key Parameters (from manuscript):
%   - Source method: sLORETA (standardized low-resolution brain
%     electromagnetic tomography)
%   - Head model: OpenMEEG BEM (boundary element method)
%   - Atlas: Destrieux 2009 (148 cortical ROIs, 74 per hemisphere)
%   - Epoch: -0.5s to +1.5s per word (500 samples @ 250Hz)
%
% Output:
%   - Subject##_sLORETA_raw.mat (148 ROIs x 500 samples per trial)
%     Variables: condition_data, condition_data_type, roiindex
%
% Requirements:
%   - Brainstorm source localization completed
%   - EEGLAB for loading preprocessed EEG
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
brainstorm_folder = fullfile(data_path, 'brainstorm_output');
eeg_folder = fullfile(data_path, 'preprocessed_eeg');
output_folder = fullfile(data_path, 'sourcedata');

% Verify paths exist
if ~exist(brainstorm_folder, 'dir')
    error('Brainstorm folder not found: %s', brainstorm_folder);
end
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS (fixed for reproducibility)
atlas_name = 'Destrieux';      % Atlas to use (index 3 in Brainstorm)
kernel_type = 'sLORETA';       % Source localization method
trial_start = 1;               % First sample to extract
trial_end = 500;               % Last sample (2s @ 250Hz)
num_rois = 148;                % Destrieux atlas ROIs

fprintf('=== Step 04: Source Localization ===\n');
fprintf('Atlas: %s (%d ROIs)\n', atlas_name, num_rois);
fprintf('Method: %s\n\n', kernel_type);

%% FIND SUBJECTS
% List subject folders in Brainstorm data directory
subject_list = dir(fullfile(brainstorm_folder, 'data', 'Subject*'));
num_subjects = length(subject_list);

fprintf('Found %d subjects\n\n', num_subjects);

%% PROCESS EACH SUBJECT
parfor subj = 1:num_subjects
    fprintf('Processing subject %d/%d\n', subj, num_subjects);

    subj_name = subject_list(subj).name;
    subj_id = subj_name(8:9);  % Extract '09' from 'Subject09'

    % Load atlas ROI definitions from anatomy file
    anat_file = fullfile(brainstorm_folder, 'anat', subj_name, 'tess_cortex_pial_low.mat');
    anat_data = load(anat_file, 'Atlas');

    % Find Destrieux atlas (index may vary)
    atlas_idx = find(strcmp({anat_data.Atlas.Name}, atlas_name));
    if isempty(atlas_idx)
        atlas_idx = 3;  % Default position for Destrieux
    end

    atlas = anat_data.Atlas(atlas_idx);

    % Extract ROI vertex indices and labels
    roiindex = cell(length(atlas.Scouts), 2);
    for i = 1:length(atlas.Scouts)
        roiindex{i, 1} = atlas.Scouts(i).Vertices;
        roiindex{i, 2} = atlas.Scouts(i).Label;
    end

    % Load sLORETA imaging kernel
    kernel_files = dir(fullfile(brainstorm_folder, 'data', subj_name, ...
                                ['results_' kernel_type '*']));
    if isempty(kernel_files)
        warning('No kernel found for subject %s', subj_name);
        continue;
    end

    kernel_data = load(fullfile(kernel_files(1).folder, kernel_files(1).name), ...
                       'ImagingKernel', 'GoodChannel');

    % Load preprocessed EEG data
    eeg_file = fullfile(eeg_folder, ['S00' subj_id '_Filters_processed_trials_rejectchan56_u1.set']);
    EEG = pop_loadset(eeg_file);

    % Initialize output variables
    num_trials = EEG.trials;
    condition_data = cell(num_trials, 1);
    condition_data_type = cell(num_trials, 1);

    % Process each trial
    for trial_idx = 1:num_trials
        % Extract trial EEG data
        trial_eeg = EEG.data(:, trial_start:trial_end, trial_idx);

        % Apply imaging kernel to get source activity
        source_data = kernel_data.ImagingKernel * trial_eeg(kernel_data.GoodChannel, :);

        % Average source activity within each ROI
        roi_means = zeros(size(roiindex, 1), size(source_data, 2));
        for roi = 1:size(roiindex, 1)
            roi_vertices = roiindex{roi, 1};
            roi_means(roi, :) = mean(source_data(roi_vertices, :), 1);
        end

        % Store results
        condition_data{trial_idx} = roi_means;
        condition_data_type{trial_idx} = EEG.event(trial_idx).type;
    end

    % Save source-localized data
    output_file = fullfile(output_folder, [subj_name '_sLORETA_raw.mat']);
    parsave(output_file, condition_data, condition_data_type, roiindex);

    fprintf('  Saved: %s (%d trials)\n', output_file, num_trials);
end

fprintf('\n=== Step 04 Complete ===\n');
fprintf('Source-localized data saved to: %s\n', output_folder);

%% HELPER FUNCTION (for parfor compatibility)
function parsave(filename, condition_data, condition_data_type, roiindex)
    save(filename, 'condition_data', 'condition_data_type', 'roiindex', '-v7.3');
end
