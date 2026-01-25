%% opensource_step02_ica_decomposition.m
% Stage 2: Run ICA decomposition with ICLabel classification
%
% This script corresponds to Methods Section 2.2 in the manuscript.
%
% Processing steps:
%   1. Load preprocessed EEG from Step 01
%   2. Run ICA decomposition (runica)
%   3. Classify components using ICLabel
%   4. Save ICA results for manual review (Step 03)
%
% Key Parameters (from manuscript):
%   - ICA algorithm: runica (EEGLAB default)
%   - Component classification: ICLabel
%
% Output:
%   - S00##_Filters_processed_trials_precut_ICA.set/.fdt
%
% Requirements:
%   - EEGLAB 2024 with ICLabel plugin
%   - Parallel Computing Toolbox (optional, for parfor)
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

% Verify path exists
if ~exist(eeg_folder, 'dir')
    error('Preprocessed EEG folder not found: %s', eeg_folder);
end

%% INITIALIZE EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

%% FIND SUBJECTS
file_list = dir(fullfile(eeg_folder, '*_Filters_processed_trials.set'));
num_subjects = length(file_list);

fprintf('=== Step 02: ICA Decomposition ===\n');
fprintf('Found %d subjects\n\n', num_subjects);

%% PROCESS EACH SUBJECT (parallelized)
parfor subj = 1:num_subjects
    fprintf('Processing subject %d/%d\n', subj, num_subjects);

    filename = file_list(subj).name;
    subj_id = filename(1:5);

    % Load preprocessed EEG
    EEG = pop_loadset('filename', filename, 'filepath', eeg_folder);

    % Run ICA decomposition
    EEG = pop_runica(EEG, 'icatype', 'runica', 'interrupt', 'on');

    % Classify ICA components using ICLabel
    EEG = pop_iclabel(EEG, 'default');

    % ICLabel categories:
    %   1 = Brain, 2 = Muscle, 3 = Eye, 4 = Heart
    %   5 = Line Noise, 6 = Channel Noise, 7 = Other

    % Save ICA results
    output_filename = [subj_id '_Filters_processed_trials_precut_ICA.set'];
    EEG = pop_saveset(EEG, 'filename', output_filename, 'filepath', eeg_folder);

    fprintf('  Saved: %s\n', output_filename);
end

fprintf('\n=== Step 02 Complete ===\n');
fprintf('ICLabel classifications saved. Proceed to Step 03 for artifact rejection.\n');
