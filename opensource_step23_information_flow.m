%% opensource_step23_information_flow.m
% Analyze information flow direction using dPLI
%
% This script corresponds to Figure 6e in the manuscript.
%
% Method:
%   - dPLI deviation from 0.5 indicates directional flow
%   - Permutation test for significance
%   - FDR correction across ROI pairs
%
% Key Results (from manuscript):
%   - Angular -> Postcentral -> Fusiform pathway
%   - Reversed direction compared to overt speech (DIVA model)
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

%% LOAD dPLI RESULTS
load(fullfile(results_folder, 'dpli_results.mat'));

fprintf('=== Step 23: Information Flow Analysis ===\n');

%% PARAMETERS
num_perms = 1000;
alpha_level = 0.05;

%% TEST DIRECTIONAL FLOW
fprintf('Testing directional flow (dPLI deviation from 0.5)...\n');

% One-sample t-test against 0.5 for each ROI pair
p_direction = zeros(num_rois, num_rois);
t_direction = zeros(num_rois, num_rois);

for i = 1:num_rois
    for j = i+1:num_rois
        dpli_values = dpli_all(:, i, j);
        [~, p, ~, stats] = ttest(dpli_values, 0.5);
        p_direction(i, j) = p;
        t_direction(i, j) = stats.tstat;
    end
end

%% FDR CORRECTION
% Only test upper triangle
upper_tri = triu(true(num_rois), 1);
p_flat = p_direction(upper_tri);

[p_sorted, sort_idx] = sort(p_flat);
m = length(p_flat);
q_flat = zeros(m, 1);
for i = 1:m
    q_flat(sort_idx(i)) = min(p_sorted(i) * m / i, 1);
end

q_direction = zeros(num_rois, num_rois);
q_direction(upper_tri) = q_flat;

%% IDENTIFY SIGNIFICANT DIRECTIONAL CONNECTIONS
[sig_i, sig_j] = find(q_direction < alpha_level & q_direction > 0);

fprintf('\nSignificant directional connections (FDR q<0.05): %d\n', length(sig_i));

%% ANALYZE KEY PATHWAY (from manuscript)
% Angular (47) -> Postcentral (55) -> Fusiform (31)

key_rois = struct();
key_rois(1).name = 'G_pariet_inf-Angular L';
key_rois(1).idx = 47;
key_rois(2).name = 'G_postcentral L';
key_rois(2).idx = 55;
key_rois(3).name = 'G_oc-temp_lat-fusifor L';
key_rois(3).idx = 31;

fprintf('\n=== Key Pathway Analysis ===\n');

for i = 1:length(key_rois)-1
    roi_from = key_rois(i);
    roi_to = key_rois(i+1);

    idx_i = min(roi_from.idx, roi_to.idx);
    idx_j = max(roi_from.idx, roi_to.idx);

    dpli_value = dpli_mean(idx_i, idx_j);
    p_value = p_direction(idx_i, idx_j);
    q_value = q_direction(idx_i, idx_j);

    % Determine direction
    if roi_from.idx < roi_to.idx
        if dpli_value > 0.5
            direction = sprintf('%s -> %s', roi_from.name, roi_to.name);
        else
            direction = sprintf('%s -> %s', roi_to.name, roi_from.name);
        end
    else
        if dpli_value > 0.5
            direction = sprintf('%s -> %s', roi_to.name, roi_from.name);
        else
            direction = sprintf('%s -> %s', roi_from.name, roi_to.name);
        end
    end

    sig_marker = '';
    if q_value < 0.05
        sig_marker = '*';
    end

    fprintf('%s\n', direction);
    fprintf('  dPLI = %.3f, p = %.4f, q = %.4f %s\n', ...
            dpli_value, p_value, q_value, sig_marker);
end

%% SAVE
flow_results = struct();
flow_results.p_direction = p_direction;
flow_results.q_direction = q_direction;
flow_results.t_direction = t_direction;
flow_results.key_rois = key_rois;
flow_results.dpli_mean = dpli_mean;

save(fullfile(results_folder, 'information_flow_results.mat'), 'flow_results');

fprintf('\n=== Step 23 Complete ===\n');
