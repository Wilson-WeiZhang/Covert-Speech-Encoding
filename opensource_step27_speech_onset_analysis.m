%% opensource_step27_speech_onset_analysis.m
% Analyze overt speech onset and offset timing
%
% This script corresponds to Figure 2a and Supplementary Section S7 in the manuscript.
%
% Method:
%   - Extract audio envelope using Hilbert transform
%   - Detect speech onset (envelope exceeds threshold)
%   - Detect speech offset (envelope falls below threshold)
%   - Average across subjects
%
% Key Results (from manuscript):
%   - Overt speech onset: 594 +/- 171 ms
%   - Overt speech offset: 1228 +/- 194 ms
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
audio_folder = fullfile(data_path, 'audio');
output_folder = fullfile(data_path, 'results');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
fs_audio = 48000;   % Original audio sampling rate
fs_target = 250;    % Downsampled rate (match EEG)
threshold_pct = 10; % Threshold: 10% of max envelope

fprintf('=== Step 27: Speech Onset Analysis ===\n');

%% FIND AUDIO FILES
audio_files = dir(fullfile(audio_folder, 'S*.mat'));
num_subjects = length(audio_files);

fprintf('Found %d subjects with audio\n\n', num_subjects);

%% ANALYZE EACH SUBJECT
onset_times = [];
offset_times = [];

for s = 1:num_subjects
    fprintf('Subject %d/%d\n', s, num_subjects);

    % Load audio data
    data = load(fullfile(audio_folder, audio_files(s).name));

    % Assume audio is in variable 'audio' or similar
    if isfield(data, 'audio')
        audio = data.audio;
    elseif isfield(data, 'envelope')
        envelope = data.envelope;
    else
        warning('Audio data not found for subject %s', audio_files(s).name);
        continue;
    end

    % Compute envelope if needed
    if ~exist('envelope', 'var')
        analytic = hilbert(audio);
        envelope = abs(analytic);

        % Downsample to 250 Hz
        envelope = resample(envelope, fs_target, fs_audio);
    end

    % Process each trial
    % Assume envelope is organized as trials x time
    if isvector(envelope)
        envelope = envelope(:)';  % Ensure row vector
        envelope = {envelope};
    end

    for t = 1:length(envelope)
        if iscell(envelope)
            trial_env = envelope{t};
        else
            trial_env = envelope(t, :);
        end

        % Threshold
        thresh = max(trial_env) * threshold_pct / 100;

        % Find onset (first crossing)
        above_thresh = find(trial_env > thresh);
        if ~isempty(above_thresh)
            onset_sample = above_thresh(1);
            onset_ms = (onset_sample / fs_target) * 1000;
            onset_times = [onset_times; onset_ms];
        end

        % Find offset (last crossing)
        if ~isempty(above_thresh)
            offset_sample = above_thresh(end);
            offset_ms = (offset_sample / fs_target) * 1000;
            offset_times = [offset_times; offset_ms];
        end
    end

    clear envelope;
end

%% STATISTICS
fprintf('\n=== Speech Timing Results ===\n');
fprintf('Onset: %.0f +/- %.0f ms (N = %d)\n', ...
        mean(onset_times), std(onset_times), length(onset_times));
fprintf('Offset: %.0f +/- %.0f ms (N = %d)\n', ...
        mean(offset_times), std(offset_times), length(offset_times));
fprintf('Duration: %.0f +/- %.0f ms\n', ...
        mean(offset_times - onset_times), std(offset_times - onset_times));

%% SAVE
speech_timing = struct();
speech_timing.onset_times = onset_times;
speech_timing.offset_times = offset_times;
speech_timing.mean_onset = mean(onset_times);
speech_timing.std_onset = std(onset_times);
speech_timing.mean_offset = mean(offset_times);
speech_timing.std_offset = std(offset_times);

save(fullfile(output_folder, 'speech_timing_results.mat'), 'speech_timing');

fprintf('\n=== Step 27 Complete ===\n');
