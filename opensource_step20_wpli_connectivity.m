%% opensource_step20_wpli_connectivity.m
% Compute weighted Phase Lag Index (wPLI) connectivity
%
% This script corresponds to Figure 6 and Methods Section 2.5 in the manuscript.
%
% Method:
%   - wPLI: Weighted Phase Lag Index (Vinck et al., 2011)
%   - wPLI = |mean(sin(dphi))| / mean(|sin(dphi)|), where dphi is the phase
%     difference between two ROIs
%   - Each analysis period is extracted first, then bandpass filtered
%     (third-order Butterworth, zero-phase) and Hilbert transformed
%   - Computed for every ROI pair, for every trial, then averaged within
%     each of the five phrases
%   - Two periods: Plan (0-600ms), Exec (600-1200ms)
%   - Frequency bands: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz)
%
% Key Parameters:
%   - ROIs: 148 (Destrieux atlas)
%   - Subjects: 57
%   - Trials: ~100 per subject (5 phrases x 20 repetitions)
%
% Input:
%   - sourcedata/Subject*_sLORETA_raw.mat (Step 04)
%
% Output:
%   - wpli_results.mat
%       wpli_by_word: subjects x phrases x rois x rois x bands x periods
%       wpli_all:     subjects x rois x rois x bands x periods (phrase average)
%       subject_ids:  numeric participant identifiers, one per row
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

% Sample index of each period inside the epoch
period_samples = cell(num_periods, 1);
for p = 1:num_periods
    win_start = baseline_samples + round(periods(p).window(1) * fs / 1000) + 1;
    win_end = baseline_samples + round(periods(p).window(2) * fs / 1000);
    period_samples{p} = win_start:win_end;
end

% Bandpass filters
filters = cell(num_bands, 1);
for b = 1:num_bands
    [b_filt, a_filt] = butter(3, bands(b).freq / (fs/2), 'bandpass');
    filters{b} = struct('b', b_filt, 'a', a_filt);
end

fprintf('=== Step 20: wPLI Connectivity ===\n');
fprintf('Periods: %d, Bands: %d\n\n', num_periods, num_bands);

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

subject_ids = zeros(num_subjects, 1);
for s = 1:num_subjects
    id_token = regexp(source_files(s).name, 'Subject(\d+)_', 'tokens');
    subject_ids(s) = str2double(id_token{1}{1});
end

fprintf('Found %d subjects\n\n', num_subjects);

%% COMPUTE wPLI
wpli_by_word = zeros(num_subjects, num_words, num_rois, num_rois, ...
                     num_bands, num_periods, 'single');
wpli_all = zeros(num_subjects, num_rois, num_rois, num_bands, num_periods);

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

    wpli_sum = zeros(num_words, num_rois, num_rois, num_bands, num_periods);
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

            for b = 1:num_bands
                filtered = filtfilt(filters{b}.b, filters{b}.a, period_data')';
                phase = angle(hilbert(filtered')');

                wpli_sum(w, :, :, b, p) = squeeze(wpli_sum(w, :, :, b, p)) + ...
                                          wpli_matrix(phase);
            end
        end
    end

    % Trial average within each phrase, then the phrase average
    word_mean = wpli_sum ./ reshape(max(word_count, 1), [num_words 1 1 1 1]);

    wpli_by_word(s, :, :, :, :, :) = reshape(word_mean, ...
        [1 num_words num_rois num_rois num_bands num_periods]);
    wpli_all(s, :, :, :, :) = reshape(mean(word_mean, 1), ...
        [1 num_rois num_rois num_bands num_periods]);
end

%% SAVE
save(fullfile(output_folder, 'wpli_results.mat'), ...
     'wpli_by_word', 'wpli_all', 'bands', 'periods', 'subject_ids', ...
     'num_subjects', 'num_rois', '-v7.3');

fprintf('\n=== Step 20 Complete ===\n');
fprintf('wPLI results saved to: %s\n', fullfile(output_folder, 'wpli_results.mat'));

%% LOCAL FUNCTIONS
function wpli_mat = wpli_matrix(phase_data)
% wPLI between every pair of ROIs, from instantaneous phase [rois x samples]
n_rois = size(phase_data, 1);
n_samples = size(phase_data, 2);

phase_i = repmat(reshape(phase_data, [n_rois, 1, n_samples]), [1, n_rois, 1]);
phase_j = repmat(reshape(phase_data, [1, n_rois, n_samples]), [n_rois, 1, 1]);
imag_part = sin(phase_i - phase_j);

wpli_mat = abs(mean(imag_part, 3)) ./ (mean(abs(imag_part), 3) + eps);
wpli_mat(1:n_rois+1:end) = 0;
end
