%% opensource_step23_information_flow.m
% Analyze information flow direction around the hub node using dPLI
%
% This script corresponds to Figure 6e in the manuscript.
%
% Method:
%   - Step 1: for every connection of the hub ROI, rmANOVA of the phrase
%     effect on delta-band wPLI, followed by Benjamini-Hochberg FDR across
%     the connections of that period
%   - Step 2: for the connections retained in Step 1, one-sample t-test of
%     the delta-band dPLI against 0.5 across participants, followed by
%     Benjamini-Hochberg FDR across those connections
%   - dPLI above 0.5 means the hub leads the target ROI, below 0.5 means the
%     target ROI leads the hub
%
% Input:
%   - wpli_results.mat (Step 20), field wpli_by_word
%   - dpli_results.mat (Step 21), field dpli_all
%
% Output:
%   - information_flow_results.mat
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

%% LOAD CONNECTIVITY RESULTS
wpli = load(fullfile(results_folder, 'wpli_results.mat'), ...
            'wpli_by_word', 'bands', 'periods', 'num_rois');
dpli = load(fullfile(results_folder, 'dpli_results.mat'), 'dpli_all');

wpli_by_word = wpli.wpli_by_word;
dpli_all = dpli.dpli_all;
bands = wpli.bands;
periods = wpli.periods;
num_rois = wpli.num_rois;

num_subjects = size(wpli_by_word, 1);
num_words = size(wpli_by_word, 2);
num_periods = length(periods);

fprintf('=== Step 23: Information Flow Analysis ===\n');

%% PARAMETERS
alpha_level = 0.05;
hub_roi = 55;       % G_postcentral L
band_idx = 1;       % Delta

% ROIs named in the manuscript pathway (Destrieux indices)
key_rois = struct();
key_rois(1).name = 'G_pariet_inf-Angular R';
key_rois(1).idx = 50;
key_rois(2).name = 'G_postcentral L';
key_rois(2).idx = hub_roi;
key_rois(3).name = 'G_oc-temp_lat-fusifor L';
key_rois(3).idx = 37;

target_rois = setdiff(1:num_rois, hub_roi);

%% STEP 1: PHRASE EFFECT ON HUB CONNECTIONS
fprintf('Testing the phrase effect on the connections of ROI %d (%s)...\n', ...
        hub_roi, bands(band_idx).name);

word_vars = arrayfun(@(w) sprintf('W%d', w), 1:num_words, 'UniformOutput', false);
model_spec = sprintf('%s-%s ~ 1', word_vars{1}, word_vars{end});
within = table(categorical((1:num_words)'), 'VariableNames', {'Phrase'});

F_conn = nan(num_rois, num_periods);
p_conn = nan(num_rois, num_periods);
q_conn = nan(num_rois, num_periods);

for p = 1:num_periods
    for r = target_rois
        conn_data = double(squeeze(wpli_by_word(:, :, hub_roi, r, band_idx, p)));

        tbl = array2table(conn_data, 'VariableNames', word_vars);
        rm = fitrm(tbl, model_spec, 'WithinDesign', within);
        ranova_tbl = ranova(rm);

        F_conn(r, p) = ranova_tbl.F(1);
        p_conn(r, p) = ranova_tbl.pValue(1);
    end

    q_conn(:, p) = bh_fdr(p_conn(:, p));
    fprintf('  %s period: %d connections with q < %.2f\n', ...
            periods(p).name, sum(q_conn(:, p) < alpha_level), alpha_level);
end

%% STEP 2: FLOW DIRECTION OF THE RETAINED CONNECTIONS
fprintf('\nTesting the dPLI deviation from 0.5 of the retained connections...\n');

t_direction = nan(num_rois, num_periods);
p_direction = nan(num_rois, num_periods);
q_direction = nan(num_rois, num_periods);
dpli_hub = squeeze(mean(dpli_all(:, hub_roi, :, :), 1));  % rois x periods

sig_connections = cell(num_periods, 1);

for p = 1:num_periods
    sel = find(q_conn(:, p) < alpha_level);
    sig_connections{p} = sel;

    for r = sel'
        [~, pv, ~, stats] = ttest(dpli_all(:, hub_roi, r, p), 0.5);
        t_direction(r, p) = stats.tstat;
        p_direction(r, p) = pv;
    end

    q_direction(sel, p) = bh_fdr(p_direction(sel, p));

    n_lead = sum(q_direction(sel, p) < alpha_level & dpli_hub(sel, p) > 0.5);
    n_lag = sum(q_direction(sel, p) < alpha_level & dpli_hub(sel, p) < 0.5);
    fprintf('  %s period: hub leads %d, hub lags %d (FDR q < %.2f)\n', ...
            periods(p).name, n_lead, n_lag, alpha_level);
end

%% PATHWAY NAMED IN THE MANUSCRIPT
fprintf('\n=== Key Pathway ===\n');

for k = [1 3]
    r = key_rois(k).idx;
    [~, pv, ~, stats] = ttest(dpli_all(:, hub_roi, r, 1), 0.5);

    if dpli_hub(r, 1) > 0.5
        direction = sprintf('%s -> %s', key_rois(2).name, key_rois(k).name);
    else
        direction = sprintf('%s -> %s', key_rois(k).name, key_rois(2).name);
    end

    fprintf('%s\n', direction);
    fprintf('  dPLI = %.3f, t(%d) = %.2f, p = %.5f, phrase effect q = %.4f\n', ...
            dpli_hub(r, 1), stats.df, stats.tstat, pv, q_conn(r, 1));
end

%% SAVE
flow_results = struct();
flow_results.hub_roi = hub_roi;
flow_results.band = bands(band_idx).name;
flow_results.period_names = {periods.name};
flow_results.F_conn = F_conn;
flow_results.p_conn = p_conn;
flow_results.q_conn = q_conn;
flow_results.sig_connections = sig_connections;
flow_results.t_direction = t_direction;
flow_results.p_direction = p_direction;
flow_results.q_direction = q_direction;
flow_results.dpli_hub = dpli_hub;
flow_results.key_rois = key_rois;

save(fullfile(results_folder, 'information_flow_results.mat'), 'flow_results');

fprintf('\n=== Step 23 Complete ===\n');

%% LOCAL FUNCTIONS
function q = bh_fdr(p)
% Benjamini-Hochberg FDR correction for a vector of p values
p = p(:);
valid = ~isnan(p);
p_valid = p(valid);
m = length(p_valid);

[p_sorted, sort_idx] = sort(p_valid);
q_sorted = min(1, cummin(p_sorted .* m ./ (1:m)', 'reverse'));

q_valid = nan(m, 1);
q_valid(sort_idx) = q_sorted;

q = nan(size(p));
q(valid) = q_valid;
end
