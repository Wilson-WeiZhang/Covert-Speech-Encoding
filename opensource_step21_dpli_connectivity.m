%% opensource_step21_dpli_connectivity.m
% Compute directed Phase Lag Index (dPLI) for information flow direction
%
% This script corresponds to Figure 6e in the manuscript.
%
% Method:
%   - dPLI: Directed Phase Lag Index (Stam & van Straaten, 2012)
%   - dPLI > 0.5: ROI i leads ROI j
%   - dPLI < 0.5: ROI j leads ROI i
%   - dPLI = 0.5: No directional preference
%
% Key Results (from manuscript):
%   - Information flow: Angular -> Postcentral -> Fusiform
%   - Direction reversed compared to DIVA model (overt speech)
%
% Output:
%   - dpli_results.mat: directed connectivity matrices
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

%% PARAMETERS
fs = 250;
baseline_samples = 125;
num_rois = 148;

% Focus on Plan period (0-600ms) and Delta band
analysis_window = [0 600];  % ms
freq_band = [1 4];  % Delta

fprintf('=== Step 21: dPLI Connectivity ===\n');
fprintf('Window: %d-%dms, Band: %d-%dHz\n\n', ...
        analysis_window(1), analysis_window(2), freq_band(1), freq_band(2));

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

fprintf('Found %d subjects\n\n', num_subjects);

%% COMPUTE dPLI
dpli_all = zeros(num_subjects, num_rois, num_rois);

[b_filt, a_filt] = butter(4, freq_band / (fs/2), 'bandpass');
win_start = baseline_samples + round(analysis_window(1) * fs / 1000) + 1;
win_end = baseline_samples + round(analysis_window(2) * fs / 1000);

for s = 1:num_subjects
    fprintf('Subject %d/%d\n', s, num_subjects);

    % Load data
    data = load(fullfile(source_folder, source_files(s).name));

    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    n_trials = length(condition_data);

    % Accumulator
    lead_count = zeros(num_rois, num_rois);
    total_count = zeros(num_rois, num_rois);

    for t = 1:n_trials
        trial_data = condition_data{t};

        % Filter
        filtered_data = zeros(num_rois, win_end - win_start + 1);
        for r = 1:num_rois
            filt_signal = filtfilt(b_filt, a_filt, trial_data(r, :));
            filtered_data(r, :) = filt_signal(win_start:win_end);
        end

        % Hilbert transform
        analytic_signal = hilbert(filtered_data')';
        phase = angle(analytic_signal);

        % Count phase leads
        for i = 1:num_rois
            for j = i+1:num_rois
                phase_diff = phase(i, :) - phase(j, :);

                % Wrap to [-pi, pi]
                phase_diff = mod(phase_diff + pi, 2*pi) - pi;

                % Count samples where i leads j (positive phase diff)
                leads = sum(phase_diff > 0);
                total = length(phase_diff);

                lead_count(i, j) = lead_count(i, j) + leads;
                lead_count(j, i) = lead_count(j, i) + (total - leads);
                total_count(i, j) = total_count(i, j) + total;
                total_count(j, i) = total_count(j, i) + total;
            end
        end
    end

    % Compute dPLI
    for i = 1:num_rois
        for j = 1:num_rois
            if total_count(i, j) > 0
                dpli_all(s, i, j) = lead_count(i, j) / total_count(i, j);
            else
                dpli_all(s, i, j) = 0.5;
            end
        end
    end
end

%% ANALYZE DIRECTION
fprintf('\n--- Analyzing Information Flow Direction ---\n');

% Average dPLI across subjects
dpli_mean = squeeze(mean(dpli_all, 1));

% Key ROIs from manuscript (approximate Destrieux indices)
roi_angular = 47;      % G_pariet_inf-Angular L
roi_postcentral = 55;  % G_postcentral L
roi_fusiform = 31;     % G_oc-temp_lat-fusifor L

fprintf('Angular -> Postcentral: dPLI = %.3f\n', dpli_mean(roi_angular, roi_postcentral));
fprintf('Postcentral -> Fusiform: dPLI = %.3f\n', dpli_mean(roi_postcentral, roi_fusiform));
fprintf('Angular -> Fusiform: dPLI = %.3f\n', dpli_mean(roi_angular, roi_fusiform));

%% SAVE
save(fullfile(output_folder, 'dpli_results.mat'), ...
     'dpli_all', 'dpli_mean', 'freq_band', 'analysis_window', 'num_subjects');

fprintf('\n=== Step 21 Complete ===\n');
