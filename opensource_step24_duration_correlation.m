%% opensource_step24_duration_correlation.m
% Relate left postcentral node strength to spoken phrase duration
%
% This script corresponds to Figure 6g-h in the manuscript.
%
% Method:
%   - Node strength of the left postcentral gyrus (ROI 55, Destrieux atlas):
%     mean delta-band wPLI between ROI 55 and all other ROIs, per phrase,
%     taken from the node strength computed in Step 22
%   - Two periods: Plan (0-600 ms) and Exec (600-1200 ms)
%   - Spoken phrase duration: offset minus onset of the overt recordings,
%     per participant and phrase (Step 27 output)
%   - Statistic: within-participant Spearman correlation across the five
%     phrases, then a one-sample t-test of those rho values against zero
%   - Linear mixed-effects model on the pooled observations, with a random
%     intercept per participant, as a repeated-measures check
%
% Cohort:
%   - Participants with both source-localized EEG and overt audio
%   - n = 53 participants x 5 phrases = 265 observations
%
% Input:
%   - results/hub_analysis_results.mat   (Step 22), field node_strength_by_word
%   - results/wpli_results.mat           (Step 20), field subject_ids
%   - results/speech_timing_results.mat  (Step 27)
%
% Output:
%   - duration_correlation_results.mat: node strength, durations, statistics
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
results_folder = fullfile(data_path, 'results');
output_folder = results_folder;

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
num_phrases = 5;

hub_roi = 55;               % G_postcentral L (Destrieux index)
band_idx = 1;               % Delta

period_names = {'Plan', 'Exec'};
num_periods = length(period_names);

% Participant S0013 has no structural MRI and therefore no source model,
% so the participant is absent from the source-localized data set.
excluded_subject_ids = 13;

fprintf('=== Step 24: Node Strength vs Phrase Duration ===\n');

%% LOAD SPOKEN PHRASE DURATION (Step 27)
timing = load(fullfile(results_folder, 'speech_timing_results.mat'));
speech_timing = timing.speech_timing;

audio_ids = zeros(length(speech_timing.unique_subjects), 1);
for i = 1:length(speech_timing.unique_subjects)
    subj_str = speech_timing.unique_subjects{i};
    audio_ids(i) = str2double(subj_str(2:end));
end

duration_all = speech_timing.offset_matrix - speech_timing.onset_matrix;   % ms

%% LOAD HUB NODE STRENGTH (Step 20 and Step 22)
connectivity = load(fullfile(results_folder, 'wpli_results.mat'), 'subject_ids');
source_ids = connectivity.subject_ids;

hub = load(fullfile(results_folder, 'hub_analysis_results.mat'), 'hub_results');
% node_strength_by_word: subjects x phrases x rois x bands x periods
node_strength_all = squeeze(hub.hub_results.node_strength_by_word(:, :, hub_roi, band_idx, :));

%% BUILD COHORT
cohort_ids = intersect(source_ids, audio_ids);
cohort_ids = setdiff(cohort_ids, excluded_subject_ids);

% A participant enters the analysis only with a duration for all five phrases
has_all_phrases = false(length(cohort_ids), 1);
for i = 1:length(cohort_ids)
    audio_idx = find(audio_ids == cohort_ids(i));
    has_all_phrases(i) = all(~isnan(duration_all(audio_idx, :)));
end
cohort_ids = cohort_ids(has_all_phrases);
num_subjects = length(cohort_ids);

fprintf('Participants with EEG and audio: %d\n', num_subjects);

%% SELECT NODE STRENGTH AND DURATION OF THE COHORT
node_strength = nan(num_subjects, num_phrases, num_periods);
duration_by_phrase = nan(num_subjects, num_phrases);

for s = 1:num_subjects
    subj_id = cohort_ids(s);
    node_strength(s, :, :) = node_strength_all(source_ids == subj_id, :, :);
    duration_by_phrase(s, :) = duration_all(audio_ids == subj_id, :);
end

%% WITHIN-PARTICIPANT SPEARMAN CORRELATION
rho_within = nan(num_subjects, num_periods);
for p = 1:num_periods
    for s = 1:num_subjects
        rho_within(s, p) = corr(duration_by_phrase(s, :)', node_strength(s, :, p)', ...
                                'Type', 'Spearman');
    end
end

fprintf('\n=== Within-participant Spearman correlation ===\n');
mean_rho = nan(num_periods, 1);
p_rho = nan(num_periods, 1);
t_rho = nan(num_periods, 1);
df_rho = nan(num_periods, 1);

for p = 1:num_periods
    [~, p_rho(p), ~, stats_p] = ttest(rho_within(:, p));
    mean_rho(p) = mean(rho_within(:, p));
    t_rho(p) = stats_p.tstat;
    df_rho(p) = stats_p.df;
    fprintf('%s period: mean rho = %.3f, t(%d) = %.3f, p = %.4f\n', ...
            period_names{p}, mean_rho(p), df_rho(p), t_rho(p), p_rho(p));
end

%% MIXED-EFFECTS MODEL ON POOLED OBSERVATIONS
participant = repmat((1:num_subjects)', num_phrases, 1);
duration_flat = duration_by_phrase(:);

lme_t = nan(num_periods, 1);
lme_df = nan(num_periods, 1);
lme_p = nan(num_periods, 1);

fprintf('\n=== Mixed-effects model: node strength ~ duration + (1|participant) ===\n');
for p = 1:num_periods
    strength_flat = reshape(node_strength(:, :, p), [], 1);
    tbl = table(strength_flat, duration_flat / 1000, categorical(participant), ...
                'VariableNames', {'NodeStrength', 'Duration', 'Participant'});
    lme = fitlme(tbl, 'NodeStrength ~ Duration + (1|Participant)');
    row = strcmp(lme.Coefficients.Name, 'Duration');
    lme_t(p) = lme.Coefficients.tStat(row);
    lme_df(p) = lme.Coefficients.DF(row);
    lme_p(p) = lme.Coefficients.pValue(row);
    fprintf('%s period: t(%d) = %.3f, p = %.4f\n', ...
            period_names{p}, lme_df(p), lme_t(p), lme_p(p));
end

%% SAVE
duration_results = struct();
duration_results.subject_ids = cohort_ids;
duration_results.num_subjects = num_subjects;
duration_results.period_names = period_names;
duration_results.duration_by_phrase = duration_by_phrase;       % subjects x phrases (ms)
duration_results.word_durations = mean(duration_by_phrase, 1);  % phrase means (ms)
duration_results.node_strength = node_strength;                 % subjects x phrases x periods
duration_results.ns_per_word = node_strength(:, :, 1);          % Plan period
duration_results.rho_within = rho_within;
duration_results.mean_rho = mean_rho;
duration_results.t_rho = t_rho;
duration_results.df_rho = df_rho;
duration_results.p_rho = p_rho;
duration_results.lme_t = lme_t;
duration_results.lme_df = lme_df;
duration_results.lme_p = lme_p;
duration_results.rho = mean_rho(1);
duration_results.p_value = p_rho(1);

save(fullfile(output_folder, 'duration_correlation_results.mat'), 'duration_results');

fprintf('\n=== Step 24 Complete ===\n');
