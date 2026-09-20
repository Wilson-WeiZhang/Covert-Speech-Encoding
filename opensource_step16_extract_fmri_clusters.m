%% opensource_step16_extract_fmri_clusters.m
% Extract significant clusters from the fMRI F-map and map them to ROIs
%
% This script corresponds to Figure 5a-b in the manuscript.
%
% Method:
%   - Load the cluster masks of the group-level fMRI F-test (five-phrase
%     ANOVA); every non-zero voxel belongs to the cluster
%   - Overlap each cluster with the Destrieux atlas and keep the ROIs
%     covering more than 5% of the ROI or more than 10 voxels; these are
%     the fMRI ROIs of the cluster
%   - Translate the fMRI ROI indices into EEG (Brainstorm) indices with the
%     atlas lookup table and keep the ROIs that also discriminated the five
%     phrases in the EEG source analysis; these are the EEG ROIs used to
%     read out the source time courses in step 17
%   - Add a whole-brain cluster (all cortical ROIs) as a baseline control
%
% The two sides of a cluster are therefore not the same ROI list: the fMRI
% pattern is sampled from every ROI overlapping the cluster, the EEG pattern
% only from the phrase-discriminating subset.
%
% Requirements:
%   - SPM12 for NIfTI handling
%   - Destrieux atlas resliced to the fMRI voxel grid
%
% Output:
%   - fmri_clusters.mat: cluster definitions, EEG and fMRI ROI indices
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
fmri_folder = fullfile(data_path, 'fmri_results');
atlas_file = fullfile(data_path, 'atlas', 'rdestrieux2009_roisi_lateralized.nii');
label_file = fullfile(data_path, 'atlas', 'EEG_ROI_LABELS.csv');
output_folder = fullfile(data_path, 'results');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

%% PARAMETERS
% Cluster masks of the second-level F-test, one file per cluster
cluster_files = {'spmF_0001_lpre.nii', 'spmF_0001_rpre.nii', 'spmF_0001_occ.nii'};
cluster_names = {'Left_Sensorimotor', 'Right_Sensorimotor', 'Occipital'};

% A ROI belongs to a cluster if the cluster covers this much of it
min_overlap_percent = 5;
min_overlap_voxels = 10;

% EEG ROIs (Brainstorm indices) that discriminated the five phrases in the
% source-level analysis; the EEG side of each cluster is restricted to them
discriminative_rois = [22, 23, 24, 45, 46, 52, 55, 56, 57, 88, ...
                       89, 90, 110, 112, 119, 120, 130, 137];

fprintf('=== Step 16: Extract fMRI Clusters ===\n');

%% LOAD ATLAS AND INDEX LOOKUP TABLE
V_atlas = spm_vol(atlas_file);
atlas = spm_read_vols(V_atlas);

roi_table = readtable(label_file);
fmri_to_eeg = containers.Map(roi_table.fmri_idx, roi_table.eeg_idx);
eeg_to_name = containers.Map(roi_table.eeg_idx, roi_table.eeg_name);

num_atlas_rois = max(roi_table.fmri_idx);

fprintf('Atlas size: %s, cortical ROIs: %d\n', ...
        mat2str(size(atlas)), height(roi_table));

%% MAP EACH CLUSTER TO ATLAS ROIs
clusters = struct('name', {}, 'rois', {}, 'roi_names', {}, 'fmri_rois', {});

for c = 1:length(cluster_files)
    mask_file = fullfile(fmri_folder, cluster_files{c});

    if ~exist(mask_file, 'file')
        error('Cluster mask not found: %s', mask_file);
    end

    cluster_mask = spm_read_vols(spm_vol(mask_file)) > 0;

    % Overlap of the cluster with every atlas ROI
    overlap_voxels = zeros(num_atlas_rois, 1);
    roi_voxels = zeros(num_atlas_rois, 1);

    for r = 1:num_atlas_rois
        roi_mask = (atlas == r);
        roi_voxels(r) = sum(roi_mask(:));
        overlap_voxels(r) = sum(cluster_mask(:) & roi_mask(:));
    end

    overlap_percent = (overlap_voxels ./ roi_voxels) * 100;
    overlap_percent(isnan(overlap_percent)) = 0;

    fmri_rois = find(overlap_percent > min_overlap_percent | ...
                     overlap_voxels > min_overlap_voxels);

    % Sort by how much of the ROI the cluster covers
    [~, order] = sort(overlap_percent(fmri_rois), 'descend');
    fmri_rois = fmri_rois(order);

    % Translate to EEG indices and keep the phrase-discriminating ROIs
    eeg_rois = zeros(1, length(fmri_rois));
    for i = 1:length(fmri_rois)
        eeg_rois(i) = fmri_to_eeg(fmri_rois(i));
    end
    eeg_rois = sort(intersect(eeg_rois, discriminative_rois));

    roi_names = cell(1, length(eeg_rois));
    for i = 1:length(eeg_rois)
        roi_names{i} = eeg_to_name(eeg_rois(i));
    end

    clusters(c).name = cluster_names{c};
    clusters(c).rois = eeg_rois;
    clusters(c).roi_names = roi_names;
    clusters(c).fmri_rois = fmri_rois(:)';

    fprintf('\n%s: %d cluster voxels, %d fMRI ROIs, %d EEG ROIs\n', ...
            cluster_names{c}, sum(cluster_mask(:)), ...
            length(fmri_rois), length(eeg_rois));
    for i = 1:length(eeg_rois)
        fprintf('    %3d  %s\n', eeg_rois(i), roi_names{i});
    end
end

%% WHOLE-BRAIN BASELINE
% All cortical ROIs of the atlas, used as a control for the cluster results
c = length(clusters) + 1;
clusters(c).name = 'Whole_Brain_Baseline';
clusters(c).rois = sort(roi_table.eeg_idx)';
clusters(c).roi_names = {};
clusters(c).fmri_rois = sort(roi_table.fmri_idx)';

fprintf('\n%s: %d EEG ROIs, %d fMRI ROIs\n', clusters(c).name, ...
        length(clusters(c).rois), length(clusters(c).fmri_rois));

%% SAVE
save(fullfile(output_folder, 'fmri_clusters.mat'), 'clusters');

fprintf('\n=== Step 16 Complete ===\n');
fprintf('Clusters saved to: %s\n', fullfile(output_folder, 'fmri_clusters.mat'));
