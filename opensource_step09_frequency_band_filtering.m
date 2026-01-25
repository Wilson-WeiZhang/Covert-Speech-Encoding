%% opensource_step09_frequency_band_filtering.m
% Filter source-localized data into frequency bands
%
% This script corresponds to Supplementary Section S2 in the manuscript.
%
% Frequency Bands:
%   - Delta: 1-4 Hz
%   - Theta: 4-8 Hz
%   - Alpha: 8-13 Hz
%   - Beta: 13-30 Hz
%
% Method:
%   - Butterworth bandpass filter (4th order)
%   - Zero-phase filtering (filtfilt)
%   - Apply to each ROI time series
%
% Output:
%   - Subject##_sLORETA_delta.mat
%   - Subject##_sLORETA_theta.mat
%   - Subject##_sLORETA_alpha.mat
%   - Subject##_sLORETA_beta.mat
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
output_base = fullfile(data_path, 'sourcedata_filtered');

if ~exist(output_base, 'dir')
    mkdir(output_base);
end

%% PARAMETERS
fs = 250;  % Sampling rate

% Frequency band definitions
bands = struct();
bands.delta = [1 4];
bands.theta = [4 8];
bands.alpha = [8 13];
bands.beta = [13 30];

band_names = fieldnames(bands);
filter_order = 4;  % Butterworth order

fprintf('=== Step 09: Frequency Band Filtering ===\n');

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

fprintf('Found %d subjects\n', num_subjects);
fprintf('Bands: %s\n\n', strjoin(band_names, ', '));

%% PROCESS EACH BAND
for b = 1:length(band_names)
    band_name = band_names{b};
    freq_range = bands.(band_name);

    fprintf('\n--- Processing %s band (%d-%d Hz) ---\n', band_name, freq_range(1), freq_range(2));

    % Create output folder
    output_folder = fullfile(output_base, band_name);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end

    % Design Butterworth filter
    [b_filt, a_filt] = butter(filter_order, freq_range / (fs/2), 'bandpass');

    for s = 1:num_subjects
        fprintf('  Subject %d/%d\n', s, num_subjects);

        % Load source data
        data = load(fullfile(source_folder, source_files(s).name));

        condition_data = data.condition_data;
        condition_data_type = data.condition_data_type;
        roiindex = data.roiindex;

        % Filter each trial
        for t = 1:length(condition_data)
            if isempty(condition_data{t})
                continue;
            end

            trial_data = condition_data{t};  % 148 x 500

            % Apply filter to each ROI
            filtered_data = zeros(size(trial_data));
            for r = 1:size(trial_data, 1)
                % Zero-phase filtering
                filtered_data(r, :) = filtfilt(b_filt, a_filt, trial_data(r, :));
            end

            condition_data{t} = filtered_data;
        end

        % Save filtered data
        subj_name = source_files(s).name(1:end-4);  % Remove '_raw.mat'
        output_file = fullfile(output_folder, [subj_name(1:9) '_sLORETA_' band_name '.mat']);
        save(output_file, 'condition_data', 'condition_data_type', 'roiindex', '-v7.3');
    end
end

fprintf('\n=== Step 09 Complete ===\n');
fprintf('Filtered data saved to: %s\n', output_base);
