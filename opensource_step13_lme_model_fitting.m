%% opensource_step13_lme_model_fitting.m
% Fit Linear Mixed Effects models for variance decomposition
%
% This script corresponds to Results Section 3.2 and Figure 4d-e in the manuscript.
%
% Models (Nakagawa & Schielzeth 2013):
%   - Fixed: Activity ~ Word
%   - RI: Activity ~ Word + (1|Subject)
%   - RS: Activity ~ Word + (Word|Subject)
%
% Variance Decomposition:
%   - R^2_marginal: Variance explained by fixed effects (Word)
%   - R^2_conditional: Variance explained by fixed + random effects
%   - Delta R^2: Individual contribution beyond group effect
%
% Key Results (from manuscript):
%   - R^2_marginal (Fixed): 0.14%
%   - Delta R^2 (RI): +6.66%
%   - Delta R^2 (RS): +11.61%
%   - Individual/Group ratio: ~131x
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

%% LOAD LME DATA
load(fullfile(results_folder, 'lme_data.mat'), 'lme_table');

fprintf('=== Step 13: LME Model Fitting ===\n');
fprintf('Data: %d observations\n\n', height(lme_table));

%% PARAMETERS
num_rois = 148;
num_windows = 12;

%% FIT MODELS PER ROI-WINDOW PAIR
R2m_fixed = zeros(num_rois, num_windows);
R2m_ri = zeros(num_rois, num_windows);
R2c_ri = zeros(num_rois, num_windows);
R2m_rs = zeros(num_rois, num_windows);
R2c_rs = zeros(num_rois, num_windows);

fprintf('Fitting models for %d ROI-window pairs...\n', num_rois * num_windows);

for r = 1:num_rois
    if mod(r, 10) == 0
        fprintf('  ROI %d/%d\n', r, num_rois);
    end

    for w = 1:num_windows
        % Subset data for this ROI-window
        idx = (lme_table.ROI == r) & (lme_table.Window == w);
        subset = lme_table(idx, :);

        if height(subset) < 10
            continue;
        end

        try
            % Model 1: Fixed effects only
            lme_fixed = fitlme(subset, 'Activity ~ Word');
            R2m_fixed(r, w) = compute_R2m(lme_fixed, subset.Activity);

            % Model 2: Random Intercept
            lme_ri = fitlme(subset, 'Activity ~ Word + (1|SubjectID)');
            [R2m_ri(r, w), R2c_ri(r, w)] = compute_R2(lme_ri, subset.Activity);

            % Model 3: Random Slope
            lme_rs = fitlme(subset, 'Activity ~ Word + (Word|SubjectID)');
            [R2m_rs(r, w), R2c_rs(r, w)] = compute_R2(lme_rs, subset.Activity);

        catch ME
            % Model fitting failed
            warning('Model failed for ROI %d, Window %d: %s', r, w, ME.message);
        end
    end
end

%% COMPUTE VARIANCE DECOMPOSITION
fprintf('\n=== Variance Decomposition Results ===\n');

% Average across ROI-window pairs
mean_R2m_fixed = mean(R2m_fixed(:)) * 100;
mean_R2m_ri = mean(R2m_ri(:)) * 100;
mean_R2c_ri = mean(R2c_ri(:)) * 100;
mean_R2m_rs = mean(R2m_rs(:)) * 100;
mean_R2c_rs = mean(R2c_rs(:)) * 100;

% Delta R^2 (individual contribution)
delta_R2_ri = mean_R2c_ri - mean_R2m_ri;
delta_R2_rs = mean_R2c_rs - mean_R2m_rs;

fprintf('R^2_marginal (Fixed):     %.2f%%\n', mean_R2m_fixed);
fprintf('R^2_marginal (RI):        %.2f%%\n', mean_R2m_ri);
fprintf('R^2_conditional (RI):     %.2f%%\n', mean_R2c_ri);
fprintf('Delta R^2 (RI):           +%.2f%%\n', delta_R2_ri);
fprintf('R^2_marginal (RS):        %.2f%%\n', mean_R2m_rs);
fprintf('R^2_conditional (RS):     %.2f%%\n', mean_R2c_rs);
fprintf('Delta R^2 (RS):           +%.2f%%\n', delta_R2_rs);

% Individual/Group ratio
if mean_R2m_fixed > 0
    ratio = delta_R2_rs / mean_R2m_fixed;
    fprintf('\nIndividual/Group ratio:   %.1fx\n', ratio);
end

%% SAVE RESULTS
results = struct();
results.R2m_fixed = R2m_fixed;
results.R2m_ri = R2m_ri;
results.R2c_ri = R2c_ri;
results.R2m_rs = R2m_rs;
results.R2c_rs = R2c_rs;
results.mean_R2m_fixed = mean_R2m_fixed;
results.delta_R2_ri = delta_R2_ri;
results.delta_R2_rs = delta_R2_rs;

save(fullfile(results_folder, 'lme_results.mat'), 'results');

fprintf('\n=== Step 13 Complete ===\n');

%% HELPER FUNCTIONS
function R2m = compute_R2m(lme, y)
    % R^2 marginal (fixed effects only)
    y_pred = fitted(lme);
    SS_res = sum((y - y_pred).^2);
    SS_tot = sum((y - mean(y)).^2);
    R2m = 1 - SS_res / SS_tot;
    R2m = max(0, R2m);  % Clamp to non-negative
end

function [R2m, R2c] = compute_R2(lme, y)
    % Nakagawa & Schielzeth (2013) R^2 for LME
    % R^2_marginal: fixed effects only
    % R^2_conditional: fixed + random effects

    % Extract variance components
    [~, ~, stats] = covarianceParameters(lme);

    % Fixed effect variance
    y_fixed = fitted(lme, 'Conditional', false);
    var_fixed = var(y_fixed);

    % Residual variance
    var_resid = stats{end}.Estimate;

    % Random effect variance
    var_random = 0;
    for i = 1:length(stats)-1
        var_random = var_random + sum(stats{i}.Estimate);
    end

    % Total variance
    var_total = var_fixed + var_random + var_resid;

    % R^2 values
    R2m = var_fixed / var_total;
    R2c = (var_fixed + var_random) / var_total;
end
