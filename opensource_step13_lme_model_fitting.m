%% opensource_step13_lme_model_fitting.m
% Fit Linear Mixed Effects models for variance decomposition
%
% This script corresponds to Results Section 3.2 and Figure 4d-e in the manuscript.
%
% Models, fitted per significant ROI-window pair on single-trial data:
%   - Fixed: Activity ~ WordType
%   - RI:    Activity ~ WordType + (1|Subject)
%   - RS:    Activity ~ WordType + (WordType|Subject)
%
% Variance decomposition (Nakagawa & Schielzeth 2013):
%   var_fixed    = var(X * beta)
%   var_random   = sum of the diagonals of the random-effects covariance
%   var_residual = model MSE
%   R2_marginal    = var_fixed / (var_fixed + var_random + var_residual)
%   R2_conditional = (var_fixed + var_random) / (var_fixed + var_random + var_residual)
%
% Increments reported in Figure 4e:
%   Delta_RI = R2_conditional(RI) - R2_marginal(Fixed)
%   Delta_RS = R2_conditional(RS) - R2_conditional(RI)
%
% Pairs with fewer than 50 observations or fewer than 10 subjects are skipped.
%
% Output:
%   - lme_results.mat: per-pair R2 components, increments, AIC/BIC,
%     likelihood ratio tests and the selected model
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
load(fullfile(results_folder, 'lme_data_all.mat'), 'lme_data_all', ...
     'num_rois', 'num_windows', 'num_subjects');

total_pairs = length(lme_data_all);

fprintf('=== Step 13: LME Model Fitting ===\n');
fprintf('ROI-window pairs: %d\n\n', total_pairs);

%% INITIALIZE RESULTS
results = struct();
results.roi = zeros(total_pairs, 1);
results.window = zeros(total_pairs, 1);
results.window_ms = zeros(total_pairs, 2);
results.num_obs = zeros(total_pairs, 1);

results.R2_marginal_fixed = nan(total_pairs, 1);
results.R2_cond_fixed = nan(total_pairs, 1);
results.R2_marginal_RI = nan(total_pairs, 1);
results.R2_cond_RI = nan(total_pairs, 1);
results.R2_marginal_RS = nan(total_pairs, 1);
results.R2_cond_RS = nan(total_pairs, 1);

results.Delta_RI = nan(total_pairs, 1);
results.Delta_RS = nan(total_pairs, 1);

results.AIC_fixed = nan(total_pairs, 1);
results.AIC_RI = nan(total_pairs, 1);
results.AIC_RS = nan(total_pairs, 1);
results.BIC_fixed = nan(total_pairs, 1);
results.BIC_RI = nan(total_pairs, 1);
results.BIC_RS = nan(total_pairs, 1);

results.LRT_RI_vs_Fixed = nan(total_pairs, 1);
results.p_RI_vs_Fixed = nan(total_pairs, 1);
results.LRT_RS_vs_RI = nan(total_pairs, 1);
results.p_RS_vs_RI = nan(total_pairs, 1);

results.best_model = cell(total_pairs, 1);
results.fit_status = cell(total_pairs, 1);

%% FIT MODELS PER PAIR
for pair_idx = 1:total_pairs
    if mod(pair_idx, 50) == 0
        fprintf('  Pair %d/%d\n', pair_idx, total_pairs);
    end

    pair_data = lme_data_all{pair_idx};
    tbl = pair_data.data_table;

    results.roi(pair_idx) = pair_data.roi;
    results.window(pair_idx) = pair_data.window;
    results.window_ms(pair_idx, :) = pair_data.window_ms;
    results.num_obs(pair_idx) = height(tbl);
    results.fit_status{pair_idx} = 'pending';

    if height(tbl) < 50 || length(unique(tbl.Subject)) < 10
        results.fit_status{pair_idx} = 'insufficient_data';
        continue;
    end

    try
        mdl_fixed = fitlme(tbl, 'Activity ~ WordType');
        [R2_m_fixed, R2_c_fixed] = calculate_R2_lme(mdl_fixed);
        results.R2_marginal_fixed(pair_idx) = R2_m_fixed;
        results.R2_cond_fixed(pair_idx) = R2_c_fixed;
        results.AIC_fixed(pair_idx) = mdl_fixed.ModelCriterion.AIC;
        results.BIC_fixed(pair_idx) = mdl_fixed.ModelCriterion.BIC;

        mdl_RI = fitlme(tbl, 'Activity ~ WordType + (1|Subject)');
        [R2_m_RI, R2_c_RI] = calculate_R2_lme(mdl_RI);
        results.R2_marginal_RI(pair_idx) = R2_m_RI;
        results.R2_cond_RI(pair_idx) = R2_c_RI;
        results.AIC_RI(pair_idx) = mdl_RI.ModelCriterion.AIC;
        results.BIC_RI(pair_idx) = mdl_RI.ModelCriterion.BIC;

        mdl_RS = fitlme(tbl, 'Activity ~ WordType + (WordType|Subject)');
        [R2_m_RS, R2_c_RS] = calculate_R2_lme(mdl_RS);
        results.R2_marginal_RS(pair_idx) = R2_m_RS;
        results.R2_cond_RS(pair_idx) = R2_c_RS;
        results.AIC_RS(pair_idx) = mdl_RS.ModelCriterion.AIC;
        results.BIC_RS(pair_idx) = mdl_RS.ModelCriterion.BIC;

        results.Delta_RI(pair_idx) = R2_c_RI - R2_m_fixed;
        results.Delta_RS(pair_idx) = R2_c_RS - R2_c_RI;

        comp_RI_Fixed = compare(mdl_fixed, mdl_RI);
        results.LRT_RI_vs_Fixed(pair_idx) = comp_RI_Fixed.LRStat(2);
        results.p_RI_vs_Fixed(pair_idx) = comp_RI_Fixed.pValue(2);

        comp_RS_RI = compare(mdl_RI, mdl_RS);
        results.LRT_RS_vs_RI(pair_idx) = comp_RS_RI.LRStat(2);
        results.p_RS_vs_RI(pair_idx) = comp_RS_RI.pValue(2);

        if results.p_RS_vs_RI(pair_idx) < 0.05
            results.best_model{pair_idx} = 'RandomSlope';
        elseif results.AIC_RI(pair_idx) < results.AIC_fixed(pair_idx)
            results.best_model{pair_idx} = 'RandomIntercept';
        else
            results.best_model{pair_idx} = 'Fixed';
        end

        results.fit_status{pair_idx} = 'success';

    catch ME
        results.fit_status{pair_idx} = ['error: ' ME.message];
    end
end

%% AVERAGE ACROSS PAIRS
success_fits = strcmp(results.fit_status, 'success');

mean_R2m_fixed = mean(results.R2_marginal_fixed(success_fits), 'omitnan') * 100;
mean_R2c_RI    = mean(results.R2_cond_RI(success_fits), 'omitnan') * 100;
mean_R2c_RS    = mean(results.R2_cond_RS(success_fits), 'omitnan') * 100;
delta_R2_RI = mean_R2c_RI - mean_R2m_fixed;
delta_R2_RS = mean_R2c_RS - mean_R2c_RI;

fprintf('\n=== Variance Decomposition (mean over %d pairs) ===\n', sum(success_fits));
fprintf('R2_marginal (Fixed):      %.2f%%\n', mean_R2m_fixed);
fprintf('R2_conditional (RI):      %.2f%%\n', mean_R2c_RI);
fprintf('Delta R2 (RI - Fixed):    %.2f%%\n', delta_R2_RI);
fprintf('R2_conditional (RS):      %.2f%%\n', mean_R2c_RS);
fprintf('Delta R2 (RS - RI):       %.2f%%\n', delta_R2_RS);

if mean_R2m_fixed > 0
    fprintf('Individual/Group ratio:   %.0fx\n', ...
            (mean_R2c_RS - mean_R2m_fixed) / mean_R2m_fixed);
end

fprintf('\nLikelihood ratio tests:\n');
fprintf('  RI over Fixed (p<.05): %.1f%%\n', 100*mean(results.p_RI_vs_Fixed(success_fits) < 0.05));
fprintf('  RS over RI (p<.05):    %.1f%%\n', 100*mean(results.p_RS_vs_RI(success_fits) < 0.05));

%% SAVE RESULTS
results.mean_R2m_fixed = mean_R2m_fixed;
results.mean_R2c_RI = mean_R2c_RI;
results.mean_R2c_RS = mean_R2c_RS;
results.delta_R2_RI = delta_R2_RI;
results.delta_R2_RS = delta_R2_RS;

save(fullfile(results_folder, 'lme_results.mat'), 'results', ...
     'num_rois', 'num_windows', 'num_subjects', '-v7.3');

fprintf('\n=== Step 13 Complete ===\n');

%% HELPER FUNCTIONS
function [R2_marginal, R2_conditional] = calculate_R2_lme(mdl)
% Nakagawa & Schielzeth (2013) R2 for linear mixed models
    X = mdl.designMatrix('Fixed');
    beta = mdl.fixedEffects;
    var_fixed = var(X * beta);

    [psi, ~] = covarianceParameters(mdl);
    var_random = 0;
    for i = 1:length(psi)
        var_random = var_random + sum(diag(psi{i}));
    end

    var_residual = mdl.MSE;
    var_total = var_fixed + var_random + var_residual;

    R2_marginal = max(0, var_fixed / var_total);
    R2_conditional = max(0, (var_fixed + var_random) / var_total);
end
