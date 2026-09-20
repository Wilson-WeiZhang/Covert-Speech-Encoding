%% opensource_step14_lme_bootstrap_ci.m
% Bootstrap confidence intervals for the LME variance components
%
% This script corresponds to the Figure 4e error bars in the manuscript.
%
% Method (cluster bootstrap over participants):
%   - Draw N participants with replacement; the draw is the same for every
%     ROI-window pair and is generated once with a fixed random seed
%   - Refit the three models on the resampled data of every significant pair
%   - Average each R2 component across pairs within a bootstrap replicate
%   - Report the 2.5th and 97.5th percentiles of the replicate distribution
%
% Output:
%   - lme_bootstrap_results.mat: bootstrap samples, means and 95% CIs for
%     R2_marginal (Fixed), R2_conditional (RI), R2_conditional (RS),
%     the two increments and their ratio
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
n_boot = 1000;
alpha = 0.05;
rng(42);

fprintf('=== Step 14: LME Bootstrap CI ===\n');
fprintf('Bootstrap iterations: %d\n', n_boot);

%% LOAD DATA
load(fullfile(results_folder, 'lme_data_all.mat'), 'lme_data_all');

total_pairs = length(lme_data_all);
all_subjects = unique(lme_data_all{1}.data_table.Subject);
n_subj = length(all_subjects);

fprintf('ROI-window pairs: %d\n', total_pairs);
fprintf('Subjects: %d\n\n', n_subj);

%% GENERATE BOOTSTRAP DRAWS
boot_subj_all = zeros(n_boot, n_subj);
for b = 1:n_boot
    boot_subj_all(b, :) = randsample(n_subj, n_subj, true);
end

%% BOOTSTRAP EACH PAIR
pair_R2m_all    = zeros(total_pairs, n_boot);
pair_R2c_ri_all = zeros(total_pairs, n_boot);
pair_R2c_rs_all = zeros(total_pairs, n_boot);
pair_valid_all  = false(total_pairs, n_boot);

parfor p = 1:total_pairs
    if mod(p, 50) == 0
        fprintf('  Pair %d/%d\n', p, total_pairs);
    end

    tbl = lme_data_all{p}.data_table;

    subj_idx = cell(n_subj, 1);
    rows_per_subj = zeros(n_subj, 1);
    for s = 1:n_subj
        subj_idx{s} = find(tbl.Subject == all_subjects(s));
        rows_per_subj(s) = length(subj_idx{s});
    end

    p_R2m = zeros(n_boot, 1);
    p_R2c_ri = zeros(n_boot, 1);
    p_R2c_rs = zeros(n_boot, 1);
    p_valid = false(n_boot, 1);

    for b = 1:n_boot
        boot_subj = boot_subj_all(b, :);

        total_rows = sum(rows_per_subj(boot_subj));
        if total_rows < 50
            continue;
        end

        Activity = zeros(total_rows, 1);
        WordType_raw = strings(total_rows, 1);
        Subject = zeros(total_rows, 1);

        row_ptr = 0;
        for s = 1:n_subj
            idx = subj_idx{boot_subj(s)};
            n_rows = length(idx);
            if n_rows > 0
                Activity(row_ptr+1:row_ptr+n_rows) = tbl.Activity(idx);
                WordType_raw(row_ptr+1:row_ptr+n_rows) = string(tbl.WordType(idx));
                Subject(row_ptr+1:row_ptr+n_rows) = s;
                row_ptr = row_ptr + n_rows;
            end
        end

        boot_tbl = table(Activity, categorical(WordType_raw), categorical(Subject), ...
            'VariableNames', {'Activity', 'WordType', 'Subject'});

        if length(unique(boot_tbl.Subject)) < 10
            continue;
        end

        try
            mdl_fixed = fitlme(boot_tbl, 'Activity ~ WordType');
            mdl_RI = fitlme(boot_tbl, 'Activity ~ WordType + (1|Subject)');
            mdl_RS = fitlme(boot_tbl, 'Activity ~ WordType + (WordType|Subject)');

            p_R2m(b) = calculate_R2_lme(mdl_fixed);
            [~, p_R2c_ri(b)] = calculate_R2_lme(mdl_RI);
            [~, p_R2c_rs(b)] = calculate_R2_lme(mdl_RS);
            p_valid(b) = true;
        catch
            continue;
        end
    end

    pair_R2m_all(p, :) = p_R2m;
    pair_R2c_ri_all(p, :) = p_R2c_ri;
    pair_R2c_rs_all(p, :) = p_R2c_rs;
    pair_valid_all(p, :) = p_valid;
end

%% AGGREGATE ACROSS PAIRS WITHIN EACH REPLICATE
boot_R2m_fixed = zeros(n_boot, 1);
boot_R2c_RI = zeros(n_boot, 1);
boot_R2c_RS = zeros(n_boot, 1);

for b = 1:n_boot
    valid_mask = pair_valid_all(:, b);
    if any(valid_mask)
        boot_R2m_fixed(b) = mean(pair_R2m_all(valid_mask, b));
        boot_R2c_RI(b) = mean(pair_R2c_ri_all(valid_mask, b));
        boot_R2c_RS(b) = mean(pair_R2c_rs_all(valid_mask, b));
    end
end

boot_delta_RI = boot_R2c_RI - boot_R2m_fixed;
boot_delta_RS = boot_R2c_RS - boot_R2c_RI;
boot_ratio = (boot_R2c_RS - boot_R2m_fixed) ./ max(boot_R2m_fixed, 1e-4);

%% CONFIDENCE INTERVALS
pct = [alpha/2, 1-alpha/2] * 100;

ci_R2m     = prctile(boot_R2m_fixed, pct);
ci_R2c_RI  = prctile(boot_R2c_RI, pct);
ci_R2c_RS  = prctile(boot_R2c_RS, pct);
ci_delta_RI = prctile(boot_delta_RI, pct);
ci_delta_RS = prctile(boot_delta_RS, pct);
ci_ratio   = prctile(boot_ratio, pct);

fprintf('\n=== Bootstrap means and 95%% CIs ===\n');
fprintf('R2_marginal (Fixed):   %.2f%% [%.2f%%, %.2f%%]\n', ...
        mean(boot_R2m_fixed)*100, ci_R2m(1)*100, ci_R2m(2)*100);
fprintf('R2_conditional (RI):   %.2f%% [%.2f%%, %.2f%%]\n', ...
        mean(boot_R2c_RI)*100, ci_R2c_RI(1)*100, ci_R2c_RI(2)*100);
fprintf('R2_conditional (RS):   %.2f%% [%.2f%%, %.2f%%]\n', ...
        mean(boot_R2c_RS)*100, ci_R2c_RS(1)*100, ci_R2c_RS(2)*100);
fprintf('Delta R2 (RI - Fixed): %.2f%% [%.2f%%, %.2f%%]\n', ...
        mean(boot_delta_RI)*100, ci_delta_RI(1)*100, ci_delta_RI(2)*100);
fprintf('Delta R2 (RS - RI):    %.2f%% [%.2f%%, %.2f%%]\n', ...
        mean(boot_delta_RS)*100, ci_delta_RS(1)*100, ci_delta_RS(2)*100);
fprintf('Individual/Group ratio: %.0fx [%.0fx, %.0fx]\n', ...
        mean(boot_ratio), ci_ratio(1), ci_ratio(2));

%% SAVE
bootstrap_results = struct();
bootstrap_results.n_boot = n_boot;
bootstrap_results.n_pairs = total_pairs;
bootstrap_results.R2m_fixed = struct('mean', mean(boot_R2m_fixed), 'ci', ci_R2m, 'samples', boot_R2m_fixed);
bootstrap_results.R2c_RI = struct('mean', mean(boot_R2c_RI), 'ci', ci_R2c_RI, 'samples', boot_R2c_RI);
bootstrap_results.R2c_RS = struct('mean', mean(boot_R2c_RS), 'ci', ci_R2c_RS, 'samples', boot_R2c_RS);
bootstrap_results.delta_RI = struct('mean', mean(boot_delta_RI), 'ci', ci_delta_RI, 'samples', boot_delta_RI);
bootstrap_results.delta_RS = struct('mean', mean(boot_delta_RS), 'ci', ci_delta_RS, 'samples', boot_delta_RS);
bootstrap_results.ratio = struct('mean', mean(boot_ratio), 'ci', ci_ratio, 'samples', boot_ratio);

save(fullfile(results_folder, 'lme_bootstrap_results.mat'), 'bootstrap_results', '-v7.3');

fprintf('\n=== Step 14 Complete ===\n');

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
