%% opensource_step27_speech_onset_analysis.m
% Analyze overt speech onset and offset timing (Figure 2a)
%
% This script corresponds to Figure 2a and Supplementary Section S7 in the manuscript.
%
% Method:
%   - Load per-subject Hilbert envelope organized by phrase label
%   - Restrict each trial to the segment containing the first overt utterance
%   - Detect speech onset/offset per trial via adaptive envelope threshold
%   - Pool trials for the grand mean +/- SD of onset and offset latency
%   - Average trials within subject x phrase and run a repeated-measures ANOVA
%     with phrase as the within-subject factor (5 levels)
%
% Cohort:
%   - S0013 is excluded because it has no structural MRI and therefore lies
%     outside the source-localized cohort; exclusion is by subject ID.
%
% Expected input format (one .mat per subject in audio_folder):
%   results.trials_by_label.label_0  % [n_trials x n_samples]
%   results.trials_by_label.label_1
%   results.trials_by_label.label_2
%   results.trials_by_label.label_3
%   results.trials_by_label.label_4
% Envelopes are sampled at 48 kHz and span the whole trial; the first overt
% utterance occupies 2.0-3.5 s after trial onset.
%
% Output:
%   - speech_timing_results.mat: per-trial detections, subject x phrase means,
%     grand means and the repeated-measures ANOVA statistics
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
audio_folder = fullfile(data_path, 'audio_envelopes');
output_folder = fullfile(data_path, 'results');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
fs = 48000;                     % Envelope sampling rate (Hz)
analysis_window_s = [2 3.5];    % Segment of the first overt utterance (s)
window_samples = analysis_window_s * fs;
n_phrases = 5;
excluded_subjects = {'S0013'};  % No structural MRI

fprintf('=== Step 27: Speech Onset Analysis (Figure 2a) ===\n');

%% FIND SUBJECT FILES
audio_files = dir(fullfile(audio_folder, 'S*_envelope.mat'));

keep_file = true(length(audio_files), 1);
for s = 1:length(audio_files)
    subject_id = audio_files(s).name(1:end-13);   % strip '_envelope.mat'
    keep_file(s) = ~ismember(subject_id, excluded_subjects);
end
audio_files = audio_files(keep_file);

num_subjects = length(audio_files);
fprintf('Found %d subjects after exclusion\n\n', num_subjects);

%% PER-TRIAL ONSET/OFFSET DETECTION
onset_results = struct([]);   % subject, label, trial, onset_sample, offset_sample

for s = 1:num_subjects
    subject_file = audio_files(s).name;
    subject_id = subject_file(1:end-13);
    fprintf('Subject %d/%d: %s\n', s, num_subjects, subject_id);

    data = load(fullfile(audio_folder, subject_file));
    trials_by_label = data.results.trials_by_label;

    for label_val = 0:(n_phrases-1)
        label_field = sprintf('label_%d', label_val);
        if ~isfield(trials_by_label, label_field) || isempty(trials_by_label.(label_field))
            continue;
        end

        trials_data = trials_by_label.(label_field);
        [n_trials, n_samples] = size(trials_data);

        start_sample = max(window_samples(1), 1);
        end_sample   = min(window_samples(2), n_samples);
        if end_sample <= start_sample; continue; end

        for trial_idx = 1:n_trials
            envelope_trial = trials_data(trial_idx, start_sample:end_sample);
            [onset_sample, offset_sample] = detect_speech_onset_offset(envelope_trial, fs);

            row = struct();
            row.subject = subject_id;
            row.label = label_val;
            row.trial = trial_idx;
            row.onset_sample = onset_sample;
            row.offset_sample = offset_sample;
            onset_results = [onset_results; row];
        end
    end
end

%% GRAND MEAN (ALL TRIALS)
all_onset = [onset_results.onset_sample];
all_offset = [onset_results.offset_sample];
onset_ms_all = (all_onset(~isnan(all_onset)) / fs) * 1000;
offset_ms_all = (all_offset(~isnan(all_offset)) / fs) * 1000;

fprintf('\n=== Grand mean (pooled across trials) ===\n');
fprintf('Onset:  %.0f +/- %.0f ms (N = %d trials)\n', ...
        mean(onset_ms_all), std(onset_ms_all), length(onset_ms_all));
fprintf('Offset: %.0f +/- %.0f ms (N = %d trials)\n', ...
        mean(offset_ms_all), std(offset_ms_all), length(offset_ms_all));

%% PER-SUBJECT PER-PHRASE MEANS (for rmANOVA)
unique_subjects = unique({onset_results.subject});
n_subj = length(unique_subjects);
onset_matrix  = nan(n_subj, n_phrases);
offset_matrix = nan(n_subj, n_phrases);

for subj_idx = 1:n_subj
    subj_id = unique_subjects{subj_idx};
    for label_val = 0:(n_phrases-1)
        mask = strcmp({onset_results.subject}, subj_id) & ...
               [onset_results.label] == label_val;
        subj_label_data = onset_results(mask);
        if isempty(subj_label_data); continue; end

        onset_samples  = [subj_label_data.onset_sample];
        offset_samples = [subj_label_data.offset_sample];

        if any(~isnan(onset_samples))
            onset_matrix(subj_idx, label_val+1) = ...
                mean((onset_samples(~isnan(onset_samples)) / fs) * 1000);
        end
        if any(~isnan(offset_samples))
            offset_matrix(subj_idx, label_val+1) = ...
                mean((offset_samples(~isnan(offset_samples)) / fs) * 1000);
        end
    end
end

%% rmANOVA (phrase as within-subject factor)
complete_onset  = all(~isnan(onset_matrix), 2);
complete_offset = all(~isnan(offset_matrix), 2);

within_design = table((1:n_phrases)', 'VariableNames', {'Phrase'});

% Onset
t_onset = array2table(onset_matrix(complete_onset, :), ...
                      'VariableNames', {'P1','P2','P3','P4','P5'});
rm_onset = fitrm(t_onset, 'P1-P5~1', 'WithinDesign', within_design);
ranova_onset = ranova(rm_onset);
F_onset   = ranova_onset.F(1);
p_onset   = ranova_onset.pValue(1);
df1_onset = ranova_onset.DF(1);
df2_onset = ranova_onset.DF(2);
eta2_onset = ranova_onset.SumSq(1) / (ranova_onset.SumSq(1) + ranova_onset.SumSq(2));

% Offset
t_offset = array2table(offset_matrix(complete_offset, :), ...
                       'VariableNames', {'P1','P2','P3','P4','P5'});
rm_offset = fitrm(t_offset, 'P1-P5~1', 'WithinDesign', within_design);
ranova_offset = ranova(rm_offset);
F_offset   = ranova_offset.F(1);
p_offset   = ranova_offset.pValue(1);
df1_offset = ranova_offset.DF(1);
df2_offset = ranova_offset.DF(2);
eta2_offset = ranova_offset.SumSq(1) / (ranova_offset.SumSq(1) + ranova_offset.SumSq(2));

fprintf('\n=== rmANOVA (phrase as within-subject factor) ===\n');
fprintf('Onset  rmANOVA: F(%d,%d) = %.2f, p = %.6f, partial eta2 = %.3f (N = %d subjects)\n', ...
        df1_onset, df2_onset, F_onset, p_onset, eta2_onset, sum(complete_onset));
fprintf('Offset rmANOVA: F(%d,%d) = %.2f, p = %.6f, partial eta2 = %.3f (N = %d subjects)\n', ...
        df1_offset, df2_offset, F_offset, p_offset, eta2_offset, sum(complete_offset));

%% SAVE
speech_timing = struct();
speech_timing.onset_results = onset_results;
speech_timing.onset_matrix  = onset_matrix;
speech_timing.offset_matrix = offset_matrix;
speech_timing.unique_subjects = unique_subjects;
speech_timing.grand_mean = struct( ...
    'onset_ms',  mean(onset_ms_all),  'onset_sd',  std(onset_ms_all), ...
    'offset_ms', mean(offset_ms_all), 'offset_sd', std(offset_ms_all), ...
    'n_onset_trials', length(onset_ms_all), 'n_offset_trials', length(offset_ms_all));
speech_timing.rmanova.onset  = struct('F', F_onset,  'p', p_onset,  ...
                                       'df1', df1_onset,  'df2', df2_onset, ...
                                       'eta2_partial', eta2_onset, ...
                                       'n', sum(complete_onset));
speech_timing.rmanova.offset = struct('F', F_offset, 'p', p_offset, ...
                                       'df1', df1_offset, 'df2', df2_offset, ...
                                       'eta2_partial', eta2_offset, ...
                                       'n', sum(complete_offset));

save(fullfile(output_folder, 'speech_timing_results.mat'), 'speech_timing');

fprintf('\n=== Step 27 Complete ===\n');

%% LOCAL FUNCTIONS
function [onset_sample, offset_sample] = detect_speech_onset_offset(envelope_data, fs)
% Adaptive threshold: baseline (quietest 100ms samples) + 5% of dynamic range.
% An onset/offset is accepted only if the envelope remains above threshold
% over the following context windows (100-600 ms), preventing spurious blips.
baseline_duration = round(0.1 * fs);
sorted_data = sort(envelope_data);
baseline_points = sorted_data(1:min(baseline_duration, length(sorted_data)));
baseline_mean = mean(baseline_points);
peak_value = max(envelope_data);
threshold = baseline_mean + (peak_value - baseline_mean) * 0.05;
context_windows = round((0.1:0.1:0.6) * fs);

onset_sample = NaN;
for i = 1:length(envelope_data)
    if envelope_data(i) > threshold
        all_satisfied = true;
        for context_samples = context_windows
            end_idx = min(i + context_samples - 1, length(envelope_data));
            if ~(mean(envelope_data(i:end_idx)) > threshold && ...
                 max(envelope_data(i:end_idx)) > envelope_data(i))
                all_satisfied = false;
                break;
            end
        end
        if all_satisfied
            onset_sample = i;
            break;
        end
    end
end

offset_sample = NaN;
for i = length(envelope_data):-1:1
    if envelope_data(i) > threshold
        all_satisfied = true;
        for context_samples = context_windows
            start_idx = max(i - context_samples + 1, 1);
            if ~(mean(envelope_data(start_idx:i)) > threshold && ...
                 max(envelope_data(start_idx:i)) > envelope_data(i))
                all_satisfied = false;
                break;
            end
        end
        if all_satisfied
            offset_sample = i;
            break;
        end
    end
end
end
