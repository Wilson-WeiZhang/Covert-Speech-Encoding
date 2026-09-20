%% opensource_step19_rsa_permutation_test.m
% RSA permutation test for EEG-fMRI correspondence
%
% This script corresponds to Figure 5c-e in the manuscript.
%
% Method:
%   - Correlate the EEG and fMRI RDM of a participant (Spearman, lower
%     triangle) and average the correlations across participants
%   - Permutation test (N=1000): the phrase labels of the EEG RDM and of
%     the fMRI RDM are shuffled independently for every participant, which
%     breaks the correspondence between the two modalities while leaving
%     each RDM intact
%   - One-tailed p = proportion of permutations whose mean correlation
%     reaches the observed mean correlation
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
alpha_level = 0.05;

fprintf('=== Step 19: RSA Permutation Test ===\n');
fprintf('Permutations: %d\n\n', num_perms);

%% LOAD RDMs
load(fullfile(results_folder, 'rdm_results.mat'));

num_clusters = length(clusters);

% Lower triangle of a 5x5 RDM
[row, col] = find(tril(ones(num_words), -1));
tri_idx = sub2ind([num_words, num_words], row, col);

%% RSA ANALYSIS
rsa_results = struct();

for c = 1:num_clusters
    fprintf('Cluster %d: %s\n', c, clusters(c).name);

    % One 5x5 RDM per participant and modality
    eeg_rdm = cell(num_subjects, 1);
    fmri_rdm = cell(num_subjects, 1);
    r_observed = nan(num_subjects, 1);

    for s = 1:num_subjects
        eeg_rdm{s} = reshape(eeg_rdms{c}(s, :, :), num_words, num_words);
        fmri_rdm{s} = reshape(fmri_rdms{c}(s, :, :), num_words, num_words);

        r_observed(s) = corr(eeg_rdm{s}(tri_idx), fmri_rdm{s}(tri_idx), ...
                             'Type', 'Spearman', 'rows', 'complete');
    end

    mean_r_observed = mean(r_observed, 'omitnan');

    % Permutation test: shuffle the phrase labels of both modalities
    r_perm = zeros(num_perms, 1);

    for perm = 1:num_perms
        r_perm_subj = nan(num_subjects, 1);

        for s = 1:num_subjects
            eeg_order = randperm(num_words);
            fmri_order = randperm(num_words);

            eeg_perm = eeg_rdm{s}(eeg_order, eeg_order);
            fmri_perm = fmri_rdm{s}(fmri_order, fmri_order);

            r_perm_subj(s) = corr(eeg_perm(tri_idx), fmri_perm(tri_idx), ...
                                  'Type', 'Spearman', 'rows', 'complete');
        end

        r_perm(perm) = mean(r_perm_subj, 'omitnan');
    end

    p_value = mean(r_perm >= mean_r_observed);

    fprintf('  rho = %.3f, p = %.3f', mean_r_observed, p_value);
    if p_value < alpha_level
        fprintf(' *\n');
    else
        fprintf('\n');
    end

    % Store results
    rsa_results(c).cluster_name = clusters(c).name;
    rsa_results(c).mean_r = mean_r_observed;
    rsa_results(c).r_per_subject = r_observed;
    rsa_results(c).num_valid_subjects = sum(~isnan(r_observed));
    rsa_results(c).p_value = p_value;
    rsa_results(c).null_distribution = r_perm;
end

%% PLOT RESULTS
figure('Position', [100 100 300*num_clusters 400]);

for c = 1:num_clusters
    subplot(1, num_clusters, c);

    % Histogram of null distribution
    histogram(rsa_results(c).null_distribution, 30, ...
              'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
    hold on;

    % Observed correlation
    xline(rsa_results(c).mean_r, 'r-', 'LineWidth', 2);

    xlabel('Spearman rho');
    ylabel('Frequency');
    title(sprintf('Figure 5: %s\nrho = %.3f, p = %.3f', ...
          rsa_results(c).cluster_name, ...
          rsa_results(c).mean_r, ...
          rsa_results(c).p_value));

    if rsa_results(c).p_value < alpha_level
        text(rsa_results(c).mean_r, max(ylim)*0.9, ' *', ...
             'FontSize', 20, 'Color', 'r', 'FontWeight', 'bold');
    end
end

saveas(gcf, fullfile(output_folder, 'figure5_rsa_results.png'));

%% SUMMARY
fprintf('\n=== RSA Results Summary ===\n');
fprintf('%-22s  %8s  %8s\n', 'Cluster', 'rho', 'p-value');
fprintf('%s\n', repmat('-', 1, 42));
for c = 1:num_clusters
    sig_marker = '';
    if rsa_results(c).p_value < alpha_level
        sig_marker = '*';
    end
    fprintf('%-22s  %8.3f  %8.3f %s\n', ...
            rsa_results(c).cluster_name, ...
            rsa_results(c).mean_r, ...
            rsa_results(c).p_value, ...
            sig_marker);
end

%% SAVE
save(fullfile(results_folder, 'rsa_results.mat'), 'rsa_results');

fprintf('\n=== Step 19 Complete ===\n');
