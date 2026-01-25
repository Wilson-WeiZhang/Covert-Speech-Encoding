%% opensource_step20_wpli_connectivity.m
% Compute weighted Phase Lag Index (wPLI) connectivity
%
% This script corresponds to Figure 6 and Methods Section 2.5 in the manuscript.
%
% Method:
%   - wPLI: Weighted Phase Lag Index (Vinck et al., 2011)
%   - Hilbert transform for instantaneous phase
%   - Computed between all ROI pairs
%   - Two periods: Plan (0-600ms), Exec (600-1200ms)
%   - Frequency bands: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz)
%
% Key Parameters:
%   - ROIs: 148 (Destrieux atlas)
%   - Subjects: 57
%   - Trials: ~100 per subject
%
% Output:
%   - wpli_results.mat: connectivity matrices per subject/band/period
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
output_folder = fullfile(data_path, 'results');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
fs = 250;
baseline_samples = 125;
num_rois = 148;

% Time periods
periods = struct();
periods(1).name = 'Plan';
periods(1).window = [0 600];  % ms
periods(2).name = 'Exec';
periods(2).window = [600 1200];  % ms

% Frequency bands
bands = struct();
bands(1).name = 'Delta';
bands(1).freq = [1 4];
bands(2).name = 'Theta';
bands(2).freq = [4 8];
bands(3).name = 'Alpha';
bands(3).freq = [8 13];
bands(4).name = 'Beta';
bands(4).freq = [13 30];

num_periods = length(periods);
num_bands = length(bands);

fprintf('=== Step 20: wPLI Connectivity ===\n');
fprintf('Periods: %d, Bands: %d\n\n', num_periods, num_bands);

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

fprintf('Found %d subjects\n\n', num_subjects);

%% COMPUTE wPLI
% Pre-allocate: subjects x rois x rois x bands x periods
wpli_all = zeros(num_subjects, num_rois, num_rois, num_bands, num_periods);

for s = 1:num_subjects
    fprintf('Subject %d/%d\n', s, num_subjects);

    % Load data
    data = load(fullfile(source_folder, source_files(s).name));

    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    n_trials = length(condition_data);

    for b = 1:num_bands
        % Design bandpass filter
        [b_filt, a_filt] = butter(4, bands(b).freq / (fs/2), 'bandpass');

        for p = 1:num_periods
            % Window samples
            win_start = baseline_samples + round(periods(p).window(1) * fs / 1000) + 1;
            win_end = baseline_samples + round(periods(p).window(2) * fs / 1000);

            % Accumulator for wPLI
            csd_imag_sum = zeros(num_rois, num_rois);
            csd_imag_abs_sum = zeros(num_rois, num_rois);

            for t = 1:n_trials
                trial_data = condition_data{t};  % 148 x 500

                % Filter and extract window
                filtered_data = zeros(num_rois, win_end - win_start + 1);
                for r = 1:num_rois
                    filt_signal = filtfilt(b_filt, a_filt, trial_data(r, :));
                    filtered_data(r, :) = filt_signal(win_start:win_end);
                end

                % Hilbert transform for phase
                analytic_signal = hilbert(filtered_data')';
                phase = angle(analytic_signal);

                % Cross-spectral density (imaginary part)
                for i = 1:num_rois
                    for j = i+1:num_rois
                        phase_diff = phase(i, :) - phase(j, :);
                        csd_imag = mean(sin(phase_diff));
                        csd_imag_abs = mean(abs(sin(phase_diff)));

                        csd_imag_sum(i, j) = csd_imag_sum(i, j) + csd_imag;
                        csd_imag_abs_sum(i, j) = csd_imag_abs_sum(i, j) + csd_imag_abs;
                    end
                end
            end

            % Compute wPLI
            for i = 1:num_rois
                for j = i+1:num_rois
                    if csd_imag_abs_sum(i, j) > 0
                        wpli = abs(csd_imag_sum(i, j)) / csd_imag_abs_sum(i, j);
                    else
                        wpli = 0;
                    end
                    wpli_all(s, i, j, b, p) = wpli;
                    wpli_all(s, j, i, b, p) = wpli;  % Symmetric
                end
            end
        end
    end
end

%% SAVE
save(fullfile(output_folder, 'wpli_results.mat'), ...
     'wpli_all', 'bands', 'periods', 'num_subjects', 'num_rois', '-v7.3');

fprintf('\n=== Step 20 Complete ===\n');
fprintf('wPLI results saved to: %s\n', fullfile(output_folder, 'wpli_results.mat'));
