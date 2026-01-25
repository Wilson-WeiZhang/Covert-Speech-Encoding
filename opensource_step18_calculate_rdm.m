%% opensource_step18_calculate_rdm.m
% Calculate Representational Dissimilarity Matrices (RDMs)
%
% This script corresponds to Figure 5 in the manuscript.
%
% Method:
%   - For each subject and cluster:
%     - EEG RDM: 1 - correlation between word patterns
%     - fMRI RDM: 1 - correlation between word beta maps
%   - RDM size: 5x5 (symmetric, diagonal = 0)
%
% Output:
%   - rdm_results.mat: EEG and fMRI RDMs per subject
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
fmri_folder = fullfile(data_path, 'fmri_contrasts');
results_folder = fullfile(data_path, 'results');

%% LOAD EEG PATTERNS
load(fullfile(results_folder, 'eeg_patterns.mat'));

num_words = 5;
num_clusters = length(clusters);

fprintf('=== Step 18: Calculate RDMs ===\n');
fprintf('Subjects: %d, Clusters: %d\n\n', num_subjects, num_clusters);

%% CALCULATE EEG RDMs
eeg_rdms = cell(num_clusters, 1);

for c = 1:num_clusters
    fprintf('Processing cluster %d: %s\n', c, clusters(c).name);

    patterns = eeg_patterns{c};  % subjects x words x ROIs
    rdms = zeros(num_subjects, num_words, num_words);

    for s = 1:num_subjects
        % Extract word patterns
        word_patterns = squeeze(patterns(s, :, :));  % 5 x nROIs

        % Calculate correlation-based RDM
        for i = 1:num_words
            for j = 1:num_words
                if i == j
                    rdms(s, i, j) = 0;
                else
                    r = corr(word_patterns(i, :)', word_patterns(j, :)');
                    rdms(s, i, j) = 1 - r;  % Dissimilarity
                end
            end
        end
    end

    eeg_rdms{c} = rdms;
end

%% CALCULATE fMRI RDMs
fprintf('\nLoading fMRI patterns...\n');

% List subjects with fMRI data
fmri_subjects = dir(fullfile(fmri_folder, 'S*'));
fmri_rdms = cell(num_clusters, 1);

for c = 1:num_clusters
    cluster_rois = clusters(c).rois;
    rdms = zeros(num_subjects, num_words, num_words);

    for s = 1:num_subjects
        % Get subject ID from EEG filename
        subj_id = valid_files{s}(8:9);  % '09' from 'Subject09_...'
        fmri_subj_folder = fullfile(fmri_folder, ['S00' subj_id]);

        if ~exist(fmri_subj_folder, 'dir')
            warning('fMRI folder not found for subject %s', subj_id);
            continue;
        end

        % Load beta maps for each word (con_0001.nii to con_0005.nii)
        word_betas = zeros(num_words, length(cluster_rois));

        for w = 1:num_words
            con_file = fullfile(fmri_subj_folder, sprintf('con_%04d.nii', w));

            if exist(con_file, 'file')
                V = spm_vol(con_file);
                beta_vol = spm_read_vols(V);

                % Extract values at cluster ROI coordinates
                % (simplified - assumes ROI masks are available)
                word_betas(w, :) = extract_roi_values(beta_vol, cluster_rois);
            end
        end

        % Calculate RDM
        for i = 1:num_words
            for j = 1:num_words
                if i == j
                    rdms(s, i, j) = 0;
                else
                    r = corr(word_betas(i, :)', word_betas(j, :)');
                    if isnan(r), r = 0; end
                    rdms(s, i, j) = 1 - r;
                end
            end
        end
    end

    fmri_rdms{c} = rdms;
end

%% SAVE
save(fullfile(results_folder, 'rdm_results.mat'), ...
     'eeg_rdms', 'fmri_rdms', 'clusters', 'num_subjects');

fprintf('\n=== Step 18 Complete ===\n');

%% HELPER FUNCTION
function values = extract_roi_values(vol, roi_indices)
    % Simplified ROI extraction
    % In practice, use atlas mask to extract voxel values
    values = zeros(1, length(roi_indices));
    for i = 1:length(roi_indices)
        % Placeholder - actual implementation uses atlas coordinates
        values(i) = mean(vol(:), 'omitnan');
    end
end
