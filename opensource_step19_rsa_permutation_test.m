%% opensource_step19_rsa_permutation_test.m
% RSA permutation test for EEG-fMRI correspondence
%
% This script corresponds to Figure 5c-e in the manuscript.
%
% Method:
%   - Correlate EEG and fMRI RDMs (lower triangle only)
%   - Permutation test (N=1000): shuffle word labels
%   - Test significance per cluster
%
% Key Results (from manuscript):
%   - Left Sensorimotor: r = 0.094, p = 0.043*
%   - Right Sensorimotor: r = 0.079, p = 0.070
%   - Occipital: r = 0.044, p = 0.190
%
% Output:
%   - rsa_results.mat: correlations and p-values per cluster
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
output_folder = fullfile(data_path, 'figures');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
num_perms = 1000;  % Permutation iterations
num_words = 5;

fprintf('=== Step 19: RSA Permutation Test ===\n');
fprintf('Permutations: %d\n\n', num_perms);

%% LOAD RDMs
load(fullfile(results_folder, 'rdm_results.mat'));

num_clusters = length(clusters);

%% RSA ANALYSIS
rsa_results = struct();

for c = 1:num_clusters
    fprintf('Cluster %d: %s\n', c, clusters(c).name);

    eeg_rdm = eeg_rdms{c};    % subjects x 5 x 5
    fmri_rdm = fmri_rdms{c};  % subjects x 5 x 5

    % Extract lower triangle indices
    [row, col] = find(tril(ones(num_words), -1));
    n_pairs = length(row);

    % Calculate observed correlation per subject
    r_observed = zeros(num_subjects, 1);

    for s = 1:num_subjects
        eeg_vec = zeros(n_pairs, 1);
        fmri_vec = zeros(n_pairs, 1);

        for p = 1:n_pairs
            eeg_vec(p) = eeg_rdm(s, row(p), col(p));
            fmri_vec(p) = fmri_rdm(s, row(p), col(p));
        end

        r_observed(s) = corr(eeg_vec, fmri_vec, 'Type', 'Spearman');
    end

    % Average correlation across subjects
    mean_r_observed = mean(r_observed, 'omitnan');

    % Permutation test
    r_perm = zeros(num_perms, 1);

    for perm = 1:num_perms
        r_perm_subj = zeros(num_subjects, 1);

        for s = 1:num_subjects
            % Shuffle word labels
            perm_order = randperm(num_words);

            % Apply permutation to fMRI RDM
            fmri_rdm_perm = fmri_rdm(s, perm_order, perm_order);

            eeg_vec = zeros(n_pairs, 1);
            fmri_vec = zeros(n_pairs, 1);

            for p = 1:n_pairs
                eeg_vec(p) = eeg_rdm(s, row(p), col(p));
                fmri_vec(p) = fmri_rdm_perm(row(p), col(p));
            end

            r_perm_subj(s) = corr(eeg_vec, fmri_vec, 'Type', 'Spearman');
        end

        r_perm(perm) = mean(r_perm_subj, 'omitnan');
    end

    % Calculate p-value
    p_value = mean(r_perm >= mean_r_observed);

    fprintf('  r = %.3f, p = %.3f', mean_r_observed, p_value);
    if p_value < 0.05
        fprintf(' *\n');
    else
        fprintf('\n');
    end

    % Store results
    rsa_results(c).cluster_name = clusters(c).name;
    rsa_results(c).mean_r = mean_r_observed;
    rsa_results(c).r_per_subject = r_observed;
    rsa_results(c).p_value = p_value;
    rsa_results(c).null_distribution = r_perm;
end

%% PLOT RESULTS
figure('Position', [100 100 1200 400]);

for c = 1:num_clusters
    subplot(1, 3, c);

    % Histogram of null distribution
    histogram(rsa_results(c).null_distribution, 30, ...
              'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
    hold on;

    % Observed correlation
    xline(rsa_results(c).mean_r, 'r-', 'LineWidth', 2);

    xlabel('Spearman r');
    ylabel('Frequency');
    title(sprintf('Figure 5: %s\nr = %.3f, p = %.3f', ...
          rsa_results(c).cluster_name, ...
          rsa_results(c).mean_r, ...
          rsa_results(c).p_value));

    if rsa_results(c).p_value < 0.05
        text(rsa_results(c).mean_r, max(ylim)*0.9, ' *', ...
             'FontSize', 20, 'Color', 'r', 'FontWeight', 'bold');
    end
end

saveas(gcf, fullfile(output_folder, 'figure5_rsa_results.png'));

%% SUMMARY
fprintf('\n=== RSA Results Summary ===\n');
fprintf('%-20s  %8s  %8s\n', 'Cluster', 'r', 'p-value');
fprintf('%s\n', repmat('-', 1, 40));
for c = 1:num_clusters
    sig_marker = '';
    if rsa_results(c).p_value < 0.05
        sig_marker = '*';
    end
    fprintf('%-20s  %8.3f  %8.3f %s\n', ...
            rsa_results(c).cluster_name, ...
            rsa_results(c).mean_r, ...
            rsa_results(c).p_value, ...
            sig_marker);
end

%% SAVE
save(fullfile(results_folder, 'rsa_results.mat'), 'rsa_results');

fprintf('\n=== Step 19 Complete ===\n');
