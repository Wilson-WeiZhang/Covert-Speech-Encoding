%% opensource_step18_calculate_rdm.m
% Calculate Representational Dissimilarity Matrices (RDMs)
%
% This script corresponds to Figure 5 in the manuscript.
%
% Method:
%   - EEG pattern: phrase-wise activity vector from step 17
%   - fMRI pattern: single-subject contrast images of the five phrases,
%     sampled voxel-wise inside the Destrieux ROIs of each cluster
%   - The fMRI side uses every atlas ROI that overlaps the cluster mask
%     (clusters(c).fmri_rois, fMRI atlas indices, from step 16); the EEG
%     side uses the phrase-discriminative subset of those ROIs
%     (clusters(c).rois, EEG indices, used in step 17)
%   - RDM entry (i,j) = 1 - Spearman correlation between the patterns of
%     phrase i and phrase j; 5x5, symmetric, diagonal = 0
%
% Requirements:
%   - SPM12 for NIfTI handling
%
% Input:
%   - eeg_patterns.mat (step 17)
%   - Destrieux atlas resliced to the fMRI voxel grid
%   - First-level fMRI contrast images, one folder per subject
%
% Output:
%   - rdm_results.mat: EEG and fMRI RDMs per subject and cluster
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
fmri_folder = fullfile(data_path, 'fmri_first_level');
atlas_file = fullfile(data_path, 'atlas', 'rdestrieux2009_roisi_lateralized.nii');
results_folder = fullfile(data_path, 'results');

%% PARAMETERS
num_words = 5;
% The five phrase regressors are contrast images con_0002 to con_0006
contrast_files = arrayfun(@(x) sprintf('con_%04d.nii', x), ...
                          2:num_words+1, 'UniformOutput', false);

%% LOAD EEG PATTERNS
load(fullfile(results_folder, 'eeg_patterns.mat'));

num_clusters = length(clusters);

fprintf('=== Step 18: Calculate RDMs ===\n');
fprintf('Subjects: %d, Clusters: %d\n\n', num_subjects, num_clusters);

%% CALCULATE EEG RDMs
eeg_rdms = cell(num_clusters, 1);

for c = 1:num_clusters
    fprintf('EEG RDM, cluster %d: %s\n', c, clusters(c).name);

    patterns = eeg_patterns{c};  % subjects x words x ROIs
    rdms = nan(num_subjects, num_words, num_words);

    for s = 1:num_subjects
        word_patterns = reshape(patterns(s, :, :), num_words, []);
        rdms(s, :, :) = calculate_rdm(word_patterns);
    end

    eeg_rdms{c} = rdms;
end

%% LOAD ATLAS AND INDEX LOOKUP TABLE
fprintf('\nLoading atlas...\n');

V_atlas = spm_vol(atlas_file);
atlas = spm_read_vols(V_atlas);

%% CALCULATE fMRI RDMs
fprintf('\nCalculating fMRI RDMs...\n');

fmri_rdms = cell(num_clusters, 1);
for c = 1:num_clusters
    fmri_rdms{c} = nan(num_subjects, num_words, num_words);
end

for s = 1:num_subjects
    % 'Subject09_sLORETA_raw.mat' -> 'S0009'
    subj_id = ['S00' valid_files{s}(8:9)];
    subj_folder = fullfile(fmri_folder, subj_id);

    % Load the five contrast volumes of this subject
    contrast_data = cell(num_words, 1);
    all_loaded = true;

    for w = 1:num_words
        con_file = fullfile(subj_folder, contrast_files{w});
        if ~exist(con_file, 'file')
            all_loaded = false;
            break;
        end
        contrast_data{w} = spm_read_vols(spm_vol(con_file));
    end

    if ~all_loaded
        warning('fMRI contrast images not found for %s', subj_id);
        continue;
    end

    for c = 1:num_clusters
        % Atlas ROIs overlapping this cluster, in fMRI atlas indexing
        fmri_rois = clusters(c).fmri_rois;

        word_patterns = extract_cluster_pattern(contrast_data, atlas, fmri_rois);

        if isempty(word_patterns)
            continue;
        end

        fmri_rdms{c}(s, :, :) = calculate_rdm(word_patterns);
    end
end

%% SAVE
save(fullfile(results_folder, 'rdm_results.mat'), ...
     'eeg_rdms', 'fmri_rdms', 'clusters', 'num_subjects');

fprintf('\n=== Step 18 Complete ===\n');

%% HELPER FUNCTIONS
function patterns = extract_cluster_pattern(contrast_data, atlas, fmri_rois)
    % Concatenate the voxels of all cluster ROIs into one pattern vector
    % per phrase. Voxels that are invalid in any phrase are dropped.
    num_words = length(contrast_data);
    patterns = zeros(num_words, 0);

    for i = 1:length(fmri_rois)
        roi_voxels = find(atlas == fmri_rois(i));
        if isempty(roi_voxels)
            continue;
        end

        roi_patterns = zeros(num_words, length(roi_voxels));
        for w = 1:num_words
            roi_patterns(w, :) = contrast_data{w}(roi_voxels)';
        end

        valid = ~any(isnan(roi_patterns), 1);
        patterns = [patterns, roi_patterns(:, valid)];
    end
end

function rdm = calculate_rdm(patterns)
    % patterns: [num_words x num_features]; rdm: 1 - Spearman correlation
    num_words = size(patterns, 1);
    rdm = zeros(num_words, num_words);

    for i = 1:num_words
        for j = 1:num_words
            if i == j
                continue;
            end
            r = corr(patterns(i, :)', patterns(j, :)', ...
                     'Type', 'Spearman', 'rows', 'complete');
            if isnan(r)
                rdm(i, j) = 1;
            else
                rdm(i, j) = 1 - r;
            end
        end
    end
end
