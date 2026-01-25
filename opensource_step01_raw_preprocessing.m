%% opensource_step01_raw_preprocessing.m
% Stage 1: Preprocess and epoch raw EEG data
%
% This script corresponds to Methods Section 2.2 in the manuscript.
%
% NOTE: All datasets are publicly available through NTU's data repository:
%   - Raw EEG: https://doi.org/[pending]
%
% Processing steps:
%   1. Load BrainVision format EEG
%   2. Resample from 1000Hz to 250Hz
%   3. Bandpass filter 1-100Hz
%   4. Notch filter 49-51Hz (line noise)
%   5. Extract event markers and assign block numbers
%   6. Expand to 5-word trials per event
%   7. Epoch: -0.5s to +1.5s per word
%
% Key Parameters (from manuscript):
%   - Original sampling rate: 1000 Hz
%   - Resampled rate: 250 Hz
%   - Bandpass: 1-100 Hz (FIR)
%   - Notch: 49-51 Hz (remove 50Hz line noise)
%   - Epoch: -0.5s to +1.5s per word (500 samples)
%   - Blocks: 10 (alternating O/C)
%   - Words per trial: 5
%
% Output:
%   - S00##_Filters_processed_trials.set/.fdt
%
% Requirements:
%   - EEGLAB 2024 or later
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
raw_eeg_folder = fullfile(data_path, 'raw_eeg');
output_folder = fullfile(data_path, 'preprocessed_eeg');

% Verify paths exist
if ~exist(raw_eeg_folder, 'dir')
    error('Raw EEG folder not found: %s', raw_eeg_folder);
end
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% INITIALIZE EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

%% PARAMETERS (fixed for reproducibility)
fs_original = 1000;        % Original sampling rate (Hz)
fs_target = 250;           % Target sampling rate (Hz)
bandpass_low = 1;          % Bandpass low cutoff (Hz)
bandpass_high = 100;       % Bandpass high cutoff (Hz)
notch_low = 49;            % Notch filter low (Hz)
notch_high = 51;           % Notch filter high (Hz)
pre_event_time = 0.5;      % Pre-event baseline (s)
post_event_time = 1.5;     % Post-event duration (s)
num_words = 5;             % Words per trial
word_interval = 2.0;       % Interval between words (s)

valid_event_types = {'S  1', 'S  2', 'S  3', 'S  4', 'S  5'};

fprintf('=== Step 01: Raw EEG Preprocessing ===\n');

%% PROCESS EACH SUBJECT
file_list = dir(fullfile(raw_eeg_folder, '*.vhdr'));
num_subjects = length(file_list);

fprintf('Found %d subjects\n\n', num_subjects);

for subj = 1:num_subjects
    fprintf('Processing subject %d/%d: %s\n', subj, num_subjects, file_list(subj).name);

    % Load raw EEG (BrainVision format)
    EEG = pop_loadbv(raw_eeg_folder, file_list(subj).name);

    % Resample to 250 Hz
    EEG = pop_resample(EEG, fs_target);

    % Bandpass filter 1-100 Hz
    EEG = pop_eegfiltnew(EEG, bandpass_low, bandpass_high, [], 0, [], 0);

    % Notch filter 49-51 Hz (remove 50Hz line noise)
    EEG = pop_eegfiltnew(EEG, notch_low, notch_high, [], 1, [], 0);

    % Extract valid event markers
    valid_indices = find(ismember({EEG.event.type}, valid_event_types));
    EEG.event = EEG.event(valid_indices);

    % Assign block numbers based on inter-event gaps (>20s = new block)
    latencies = [EEG.event.latency];
    latency_diffs = diff(latencies) / EEG.srate;
    block_breaks = find(latency_diffs > 20);

    block_numbers = ones(1, length(EEG.event));
    current_block = 1;
    for i = 1:length(block_breaks)
        block_numbers(block_breaks(i)+1:end) = current_block + 1;
        current_block = current_block + 1;
    end

    % Verify 10 blocks
    if max(block_numbers) ~= 10
        warning('Subject %d: Expected 10 blocks, found %d blocks', subj, max(block_numbers));
        continue;
    end

    % Add block_number field to events
    event_cell = struct2cell(EEG.event);
    field_names = fieldnames(EEG.event);
    event_cell(end+1, :) = num2cell(block_numbers);
    field_names{end+1} = 'block_number';
    EEG.event = cell2struct(event_cell, field_names, 1);

    % Remove duplicate events (<5s apart)
    to_remove = [];
    for j = 2:length(EEG.event)
        if (EEG.event(j).latency - EEG.event(j-1).latency) < 5 * EEG.srate
            to_remove = [to_remove, j];
        end
    end
    EEG.event(to_remove) = [];

    % Rename events: Odd blocks = 'O', Even blocks = 'C'
    for j = 1:length(EEG.event)
        current_block = EEG.event(j).block_number;
        if mod(current_block, 2) == 1
            EEG.event(j).type = ['O' EEG.event(j).type(3:end)];
        else
            EEG.event(j).type = ['C' EEG.event(j).type(3:end)];
        end
    end

    % Expand events: Each trial marker -> 5 word events
    expanded_events = struct('type', {}, 'latency', {}, 'urevent', {}, 'block_number', {});
    event_counter = 0;

    for j = 1:length(EEG.event)
        for w = 1:num_words
            event_counter = event_counter + 1;
            word_latency = EEG.event(j).latency + (w-1) * word_interval * EEG.srate;

            % Label format: 'O 1_u_1_b_3' = Odd block, Word 1, Utterance 1, Block 3
            event_label = sprintf('%s_u_%d_b_%d', EEG.event(j).type, w, EEG.event(j).block_number);

            expanded_events(event_counter).type = event_label;
            expanded_events(event_counter).latency = word_latency;
            expanded_events(event_counter).urevent = event_counter;
            expanded_events(event_counter).block_number = EEG.event(j).block_number;
        end
    end

    EEG.event = expanded_events;
    EEG.urevent = expanded_events;

    % Epoch data
    epoch_times = [-pre_event_time, post_event_time];
    EEG = pop_epoch(EEG, {}, epoch_times);

    % Extract subject ID from filename
    [~, fname, ~] = fileparts(file_list(subj).name);
    subj_id = fname(1:5);  % e.g., 'S0009'

    % Save preprocessed data
    output_filename = [subj_id '_Filters_processed_trials.set'];
    EEG = pop_saveset(EEG, 'filename', output_filename, 'filepath', output_folder);

    fprintf('  Saved: %s (%d epochs)\n', output_filename, EEG.trials);
end

fprintf('\n=== Step 01 Complete ===\n');
fprintf('Output directory: %s\n', output_folder);
