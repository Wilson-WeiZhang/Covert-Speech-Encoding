%% opensource_step22_hub_node_analysis.m
% Identify phrase-discriminative hub nodes from wPLI node strength
%
% This script corresponds to Figure 6b-c in the manuscript.
%
% Method:
%   - Node strength: average wPLI of one ROI to all other ROIs,
%     computed separately for each of the five phrases
%   - rmANOVA: within-subject effect of phrase on node strength,
%     run for every ROI x band x period cell (sphericity-assumed p value)
%   - Benjamini-Hochberg FDR: applied within each frequency band across
%     the 148 ROIs, separately for each period
%
% Key Parameters:
%   - ROIs: 148 (Destrieux atlas)
%   - Phrases: 5
%   - Subjects: 57
%   - Bands: Delta (1-4Hz), Theta (4-8Hz), Alpha (8-13Hz), Beta (13-30Hz)
%   - Periods: Plan (0-600ms), Exec (600-1200ms)
%
% Input:
%   - wpli_results.mat (Step 20), including the per-phrase connectivity array
%     wpli_by_word: subjects x phrases x rois x rois x bands x periods
%
% Output:
%   - hub_analysis_results.mat: node strength and statistics
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

%% LOAD wPLI RESULTS
load(fullfile(results_folder, 'wpli_results.mat'));

num_words = size(wpli_by_word, 2);
num_bands = length(bands);
num_periods = length(periods);

fprintf('=== Step 22: Hub Node Analysis ===\n');
fprintf('Subjects: %d, ROIs: %d, Phrases: %d, Bands: %d, Periods: %d\n\n', ...
        num_subjects, num_rois, num_words, num_bands, num_periods);

%% COMPUTE NODE STRENGTH PER PHRASE
% Node strength = mean connectivity of an ROI to all other ROIs

node_strength_by_word = zeros(num_subjects, num_words, num_rois, num_bands, num_periods);

for s = 1:num_subjects
    for w = 1:num_words
        for b = 1:num_bands
            for p = 1:num_periods
                conn_matrix = squeeze(wpli_by_word(s, w, :, :, b, p));
                conn_matrix(1:num_rois+1:end) = 0;  % exclude self-connection
                node_strength_by_word(s, w, :, b, p) = mean(conn_matrix, 2);
            end
        end
    end
end

% Phrase-averaged node strength
node_strength = squeeze(mean(node_strength_by_word, 2));

%% rmANOVA ON NODE STRENGTH
fprintf('Running rmANOVA for the phrase effect...\n');

word_vars = arrayfun(@(w) sprintf('W%d', w), 1:num_words, 'UniformOutput', false);
model_spec = sprintf('%s-%s ~ 1', word_vars{1}, word_vars{end});
within = table(categorical((1:num_words)'), 'VariableNames', {'Phrase'});

F_matrix = zeros(num_rois, num_bands, num_periods);
p_matrix = zeros(num_rois, num_bands, num_periods);
q_matrix = zeros(num_rois, num_bands, num_periods);

for p = 1:num_periods
    for b = 1:num_bands
        for r = 1:num_rois
            data_roi = squeeze(node_strength_by_word(:, :, r, b, p));  % subjects x phrases

            tbl = array2table(data_roi, 'VariableNames', word_vars);
            rm = fitrm(tbl, model_spec, 'WithinDesign', within);
            ranova_tbl = ranova(rm);

            F_matrix(r, b, p) = ranova_tbl.F(1);
            p_matrix(r, b, p) = ranova_tbl.pValue(1);
        end

        % FDR is applied within each frequency band across the 148 ROIs
        q_matrix(:, b, p) = bh_fdr(p_matrix(:, b, p));
    end
    fprintf('  %s period done\n', periods(p).name);
end

%% IDENTIFY SIGNIFICANT HUBS
fprintf('\n=== Significant Hub Nodes (FDR q<0.05) ===\n');
for p = 1:num_periods
    [sig_roi, sig_band] = find(q_matrix(:, :, p) < 0.05);
    if isempty(sig_roi)
        fprintf('%s: none\n', periods(p).name);
    else
        for i = 1:length(sig_roi)
            fprintf('%s, %s: ROI %d, F = %.2f, q = %.3f\n', ...
                    periods(p).name, bands(sig_band(i)).name, sig_roi(i), ...
                    F_matrix(sig_roi(i), sig_band(i), p), ...
                    q_matrix(sig_roi(i), sig_band(i), p));
        end
    end
end

%% HUB ROI DETAIL
hub_roi = 55;      % G_postcentral L
hub_band = 1;      % Delta
hub_period = 1;    % Plan

fprintf('\n--- ROI %d (G_postcentral L), %s, %s ---\n', ...
        hub_roi, bands(hub_band).name, periods(hub_period).name);
fprintf('F(%d,%d) = %.2f, p = %.5f, q = %.4f\n', ...
        num_words - 1, (num_subjects - 1) * (num_words - 1), ...
        F_matrix(hub_roi, hub_band, hub_period), ...
        p_matrix(hub_roi, hub_band, hub_period), ...
        q_matrix(hub_roi, hub_band, hub_period));

%% SAVE
hub_results = struct();
hub_results.node_strength = node_strength;
hub_results.node_strength_by_word = node_strength_by_word;
hub_results.F_matrix = F_matrix;
hub_results.p_matrix = p_matrix;
hub_results.q_matrix = q_matrix;
hub_results.F_values = F_matrix(:, hub_band, hub_period);
hub_results.p_values = p_matrix(:, hub_band, hub_period);
hub_results.q_values = q_matrix(:, hub_band, hub_period);
hub_results.hub_roi = hub_roi;
hub_results.hub_band = bands(hub_band).name;
hub_results.hub_period = periods(hub_period).name;

save(fullfile(results_folder, 'hub_analysis_results.mat'), 'hub_results');

fprintf('\n=== Step 22 Complete ===\n');

%% LOCAL FUNCTIONS
function q = bh_fdr(p)
% Benjamini-Hochberg FDR correction for a vector of p values
m = length(p);
[p_sorted, sort_idx] = sort(p(:));
q_sorted = min(1, cummin(p_sorted .* m ./ (1:m)', 'reverse'));
q = zeros(m, 1);
q(sort_idx) = q_sorted;
end
