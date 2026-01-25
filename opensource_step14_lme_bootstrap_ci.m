%% opensource_step14_lme_bootstrap_ci.m
% Bootstrap confidence intervals for LME variance components
%
% This script corresponds to Figure 4e error bars in the manuscript.
%
% Method:
%   - Resample subjects with replacement (N = 57)
%   - Refit LME models
%   - Compute 95% CI from bootstrap distribution
%
% Output:
%   - Bootstrap distributions and 95% CIs for each R^2 component
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

%% PARAMETERS
num_bootstrap = 1000;  % Bootstrap iterations
alpha = 0.05;          % For 95% CI

fprintf('=== Step 14: LME Bootstrap CI ===\n');
fprintf('Bootstrap iterations: %d\n\n', num_bootstrap);

%% LOAD DATA
load(fullfile(results_folder, 'lme_data.mat'), 'lme_table');

% Get unique subjects
subjects = unique(lme_table.SubjectID);
num_subjects = length(subjects);

fprintf('Subjects: %d\n', num_subjects);

%% BOOTSTRAP
R2m_boot = zeros(num_bootstrap, 1);
R2c_ri_boot = zeros(num_bootstrap, 1);
R2c_rs_boot = zeros(num_bootstrap, 1);

% Use a single ROI-window pair for demonstration (fastest)
% In full analysis, average across all pairs
target_roi = 55;  % Example: postcentral gyrus
target_window = 7;  % Example: 300-350ms

idx = (lme_table.ROI == target_roi) & (lme_table.Window == target_window);
subset_full = lme_table(idx, :);

fprintf('Running bootstrap on ROI %d, Window %d...\n', target_roi, target_window);

parfor b = 1:num_bootstrap
    if mod(b, 100) == 0
        fprintf('  Iteration %d/%d\n', b, num_bootstrap);
    end

    % Resample subjects
    boot_subjects = subjects(randi(num_subjects, num_subjects, 1));

    % Build bootstrap sample
    boot_data = [];
    for s = 1:num_subjects
        orig_subj = boot_subjects(s);
        subj_data = subset_full(subset_full.SubjectID == orig_subj, :);

        % Rename subject to avoid duplicate issues
        subj_data.SubjectID = categorical(repmat(s, height(subj_data), 1));
        boot_data = [boot_data; subj_data];
    end

    try
        % Fit models
        lme_fixed = fitlme(boot_data, 'Activity ~ Word');
        lme_ri = fitlme(boot_data, 'Activity ~ Word + (1|SubjectID)');
        lme_rs = fitlme(boot_data, 'Activity ~ Word + (Word|SubjectID)');

        % Extract R^2
        R2m_boot(b) = compute_R2m(lme_fixed, boot_data.Activity);
        [~, R2c_ri_boot(b)] = compute_R2(lme_ri, boot_data.Activity);
        [~, R2c_rs_boot(b)] = compute_R2(lme_rs, boot_data.Activity);

    catch
        R2m_boot(b) = NaN;
        R2c_ri_boot(b) = NaN;
        R2c_rs_boot(b) = NaN;
    end
end

%% COMPUTE CONFIDENCE INTERVALS
R2m_boot = R2m_boot(~isnan(R2m_boot));
R2c_ri_boot = R2c_ri_boot(~isnan(R2c_ri_boot));
R2c_rs_boot = R2c_rs_boot(~isnan(R2c_rs_boot));

ci_R2m = prctile(R2m_boot, [100*alpha/2, 100*(1-alpha/2)]) * 100;
ci_R2c_ri = prctile(R2c_ri_boot, [100*alpha/2, 100*(1-alpha/2)]) * 100;
ci_R2c_rs = prctile(R2c_rs_boot, [100*alpha/2, 100*(1-alpha/2)]) * 100;

fprintf('\n=== 95%% Confidence Intervals ===\n');
fprintf('R^2_m (Fixed):  [%.2f%%, %.2f%%]\n', ci_R2m(1), ci_R2m(2));
fprintf('R^2_c (RI):     [%.2f%%, %.2f%%]\n', ci_R2c_ri(1), ci_R2c_ri(2));
fprintf('R^2_c (RS):     [%.2f%%, %.2f%%]\n', ci_R2c_rs(1), ci_R2c_rs(2));

%% SAVE
bootstrap_results = struct();
bootstrap_results.R2m_boot = R2m_boot;
bootstrap_results.R2c_ri_boot = R2c_ri_boot;
bootstrap_results.R2c_rs_boot = R2c_rs_boot;
bootstrap_results.ci_R2m = ci_R2m;
bootstrap_results.ci_R2c_ri = ci_R2c_ri;
bootstrap_results.ci_R2c_rs = ci_R2c_rs;

save(fullfile(results_folder, 'lme_bootstrap_results.mat'), 'bootstrap_results');

fprintf('\n=== Step 14 Complete ===\n');

%% HELPER FUNCTIONS
function R2m = compute_R2m(lme, y)
    y_pred = fitted(lme);
    SS_res = sum((y - y_pred).^2);
    SS_tot = sum((y - mean(y)).^2);
    R2m = max(0, 1 - SS_res / SS_tot);
end

function [R2m, R2c] = compute_R2(lme, y)
    [~, ~, stats] = covarianceParameters(lme);
    y_fixed = fitted(lme, 'Conditional', false);
    var_fixed = var(y_fixed);
    var_resid = stats{end}.Estimate;
    var_random = 0;
    for i = 1:length(stats)-1
        var_random = var_random + sum(stats{i}.Estimate);
    end
    var_total = var_fixed + var_random + var_resid;
    R2m = var_fixed / var_total;
    R2c = (var_fixed + var_random) / var_total;
end
