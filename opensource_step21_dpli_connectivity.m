%% opensource_step21_dpli_connectivity.m
% Compute directed Phase Lag Index (dPLI) for information flow direction
%
% This script corresponds to Figure 6e in the manuscript.
%
% Method:
%   - dPLI: Directed Phase Lag Index (Stam & van Straaten, 2012)
%   - dPLI = (mean(sign(sin(dphi))) + 1) / 2, where dphi is the phase
%     difference between two ROIs
%   - dPLI > 0.5: ROI i leads ROI j
%   - dPLI < 0.5: ROI j leads ROI i
%   - dPLI = 0.5: No directional preference
%   - Each analysis period is extracted first, then bandpass filtered
%     (third-order Butterworth, zero-phase) and Hilbert transformed
%   - Computed per trial, averaged within each of the five phrases, then
%     across phrases
%
% Key Parameters:
%   - ROIs: 148 (Destrieux atlas)
%   - Band: Delta (1-4Hz)
%   - Periods: Plan (0-600ms), Exec (600-1200ms)
%
% Input:
%   - sourcedata/Subject*_sLORETA_raw.mat (Step 04)
%
% Output:
%   - dpli_results.mat
%       dpli_all:    subjects x rois x rois x periods
%       dpli_mean:   rois x rois x periods (group average)
%       subject_ids: numeric participant identifiers, one per row
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
num_words = 5;

freq_band = [1 4];  % Delta

periods = struct();
periods(1).name = 'Plan';
periods(1).window = [0 600];  % ms
periods(2).name = 'Exec';
periods(2).window = [600 1200];  % ms
num_periods = length(periods);

period_samples = cell(num_periods, 1);
for p = 1:num_periods
    win_start = baseline_samples + round(periods(p).window(1) * fs / 1000) + 1;
    win_end = baseline_samples + round(periods(p).window(2) * fs / 1000);
    period_samples{p} = win_start:win_end;
end

[b_filt, a_filt] = butter(3, freq_band / (fs/2), 'bandpass');

fprintf('=== Step 21: dPLI Connectivity ===\n');
fprintf('Band: %d-%dHz, Periods: %d\n\n', freq_band(1), freq_band(2), num_periods);

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

subject_ids = zeros(num_subjects, 1);
for s = 1:num_subjects
    id_token = regexp(source_files(s).name, 'Subject(\d+)_', 'tokens');
    subject_ids(s) = str2double(id_token{1}{1});
end

fprintf('Found %d subjects\n\n', num_subjects);

%% COMPUTE dPLI
dpli_all = zeros(num_subjects, num_rois, num_rois, num_periods);

for s = 1:num_subjects
    fprintf('Subject %d/%d\n', s, num_subjects);

    data = load(fullfile(source_folder, source_files(s).name), ...
                'condition_data', 'condition_data_type');

    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    condition_data_type = data.condition_data_type(valid_idx);
    n_trials = length(condition_data);

    % Phrase identity of each trial, e.g. 'C 1_u_1_b_3' -> 1
    word_labels = zeros(n_trials, 1);
    for t = 1:n_trials
        word_match = regexp(condition_data_type{t}, '[OC]\s*(\d)', 'tokens');
        if ~isempty(word_match)
            word_labels(t) = str2double(word_match{1}{1});
        end
    end

    dpli_sum = zeros(num_words, num_rois, num_rois, num_periods);
    word_count = zeros(num_words, 1);

    for t = 1:n_trials
        w = word_labels(t);
        if w < 1 || w > num_words
            continue;
        end

        trial_data = double(condition_data{t});
        if size(trial_data, 2) < period_samples{end}(end)
            continue;
        end
        word_count(w) = word_count(w) + 1;

        for p = 1:num_periods
            period_data = trial_data(:, period_samples{p});
            filtered = filtfilt(b_filt, a_filt, period_data')';
            phase = angle(hilbert(filtered')');

            dpli_sum(w, :, :, p) = squeeze(dpli_sum(w, :, :, p)) + dpli_matrix(phase);
        end
    end

    % Trial average within each phrase, then the phrase average
    word_mean = dpli_sum ./ reshape(max(word_count, 1), [num_words 1 1 1]);
    dpli_all(s, :, :, :) = reshape(mean(word_mean, 1), [1 num_rois num_rois num_periods]);
end

dpli_mean = squeeze(mean(dpli_all, 1));

%% SAVE
save(fullfile(output_folder, 'dpli_results.mat'), ...
     'dpli_all', 'dpli_mean', 'freq_band', 'periods', 'subject_ids', ...
     'num_subjects', 'num_rois', '-v7.3');

fprintf('\n=== Step 21 Complete ===\n');
fprintf('dPLI results saved to: %s\n', fullfile(output_folder, 'dpli_results.mat'));

%% LOCAL FUNCTIONS
function dpli_mat = dpli_matrix(phase_data)
% dPLI between every pair of ROIs, from instantaneous phase [rois x samples]
n_rois = size(phase_data, 1);
n_samples = size(phase_data, 2);

phase_i = repmat(reshape(phase_data, [n_rois, 1, n_samples]), [1, n_rois, 1]);
phase_j = repmat(reshape(phase_data, [1, n_rois, n_samples]), [n_rois, 1, 1]);
imag_part = sin(phase_i - phase_j);

dpli_mat = (mean(sign(imag_part), 3) + 1) / 2;
dpli_mat(1:n_rois+1:end) = 0;
end
