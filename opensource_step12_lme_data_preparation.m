%% opensource_step12_lme_data_preparation.m
% Prepare data for Linear Mixed Effects (LME) variance decomposition
%
% This script corresponds to Results Section 3.2 and Figure 4c-e in the manuscript.
%
% Data structure for LME:
%   - Response: Neural activity (ROI amplitude)
%   - Fixed effect: Word (categorical, 5 levels)
%   - Random effects: Subject intercept, Subject-by-Word slope
%
% Output:
%   - lme_data.mat: Long-format table for LME analysis
%     Columns: SubjectID, Word, ROI, Window, Activity
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

% Analysis window: 0-600ms (12 x 50ms windows)
win_ms = 50;
num_windows = 12;
win_starts = baseline_samples + round((0:num_windows-1) * win_ms * fs / 1000) + 1;
win_ends = baseline_samples + round((1:num_windows) * win_ms * fs / 1000);

fprintf('=== Step 12: LME Data Preparation ===\n');

%% FIND SUBJECTS
source_files = dir(fullfile(source_folder, 'Subject*_sLORETA_raw.mat'));
num_subjects = length(source_files);

fprintf('Found %d subjects\n', num_subjects);
fprintf('Data dimensions: %d subjects x %d words x %d ROIs x %d windows\n\n', ...
        num_subjects, num_words, num_rois, num_windows);

%% PRE-ALLOCATE
% Estimate total rows: subjects * words * rois * windows = 57 * 5 * 148 * 12 = ~5M
% Too large for single table, use cell arrays first
max_rows = num_subjects * num_words * num_rois * num_windows;

SubjectID = zeros(max_rows, 1);
Word = zeros(max_rows, 1);
ROI = zeros(max_rows, 1);
Window = zeros(max_rows, 1);
Activity = zeros(max_rows, 1);

row_idx = 0;

%% EXTRACT DATA
for s = 1:num_subjects
    fprintf('Subject %d/%d\n', s, num_subjects);

    % Load data
    data = load(fullfile(source_folder, source_files(s).name));

    valid_idx = ~cellfun(@isempty, data.condition_data);
    condition_data = data.condition_data(valid_idx);
    condition_data_type = data.condition_data_type(valid_idx);

    % Average trials per word
    word_data = zeros(num_words, num_rois, num_windows);
    word_count = zeros(num_words, 1);

    for t = 1:length(condition_data)
        label = condition_data_type{t};
        word_id = str2double(label(3));

        if word_id < 1 || word_id > 5, continue; end

        trial_data = condition_data{t};  % 148 x 500

        % Baseline correction
        baseline = mean(trial_data(:, 1:baseline_samples), 2);
        trial_data = trial_data - baseline;

        % Extract windows
        for w = 1:num_windows
            win_mean = mean(trial_data(:, win_starts(w):win_ends(w)), 2);
            word_data(word_id, :, w) = squeeze(word_data(word_id, :, w)) + win_mean';
        end
        word_count(word_id) = word_count(word_id) + 1;
    end

    % Average
    for w_id = 1:num_words
        if word_count(w_id) > 0
            word_data(w_id, :, :) = word_data(w_id, :, :) / word_count(w_id);
        end
    end

    % Add to long format
    for w_id = 1:num_words
        for r = 1:num_rois
            for win = 1:num_windows
                row_idx = row_idx + 1;
                SubjectID(row_idx) = s;
                Word(row_idx) = w_id;
                ROI(row_idx) = r;
                Window(row_idx) = win;
                Activity(row_idx) = word_data(w_id, r, win);
            end
        end
    end
end

% Trim to actual size
SubjectID = SubjectID(1:row_idx);
Word = Word(1:row_idx);
ROI = ROI(1:row_idx);
Window = Window(1:row_idx);
Activity = Activity(1:row_idx);

%% CREATE TABLE
lme_table = table(SubjectID, Word, ROI, Window, Activity);

% Convert to categorical
lme_table.SubjectID = categorical(lme_table.SubjectID);
lme_table.Word = categorical(lme_table.Word);

fprintf('\nData table created: %d rows\n', height(lme_table));

%% SAVE
save(fullfile(output_folder, 'lme_data.mat'), 'lme_table', '-v7.3');

fprintf('=== Step 12 Complete ===\n');
