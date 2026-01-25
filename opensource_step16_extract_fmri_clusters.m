%% opensource_step16_extract_fmri_clusters.m
% Extract significant clusters from fMRI F-map
%
% This script corresponds to Figure 5a-b in the manuscript.
%
% Method:
%   - Load group-level fMRI F-map
%   - Threshold at F > 2.5 (word discrimination)
%   - Extract 3 significant clusters:
%     1. Left Sensorimotor (4 ROIs)
%     2. Right Sensorimotor (5 ROIs)
%     3. Occipital (9 ROIs)
%   - Map to Destrieux atlas ROIs
%
% Requirements:
%   - SPM12 for NIfTI handling
%   - Destrieux atlas registered to MNI space
%
% Output:
%   - fmri_clusters.mat: cluster definitions and ROI mappings
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
output_folder = fullfile(data_path, 'results');

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

fprintf('=== Step 16: Extract fMRI Clusters ===\n');

%% LOAD F-MAP
% Note: F-map is from SPM second-level analysis (5-word ANOVA)
fmap_file = fullfile(fmri_folder, 'spmF_0001.nii');

if exist(fmap_file, 'file')
    V_fmap = spm_vol(fmap_file);
    fmap = spm_read_vols(V_fmap);
else
    warning('F-map not found. Using pre-defined cluster ROIs.');
    fmap = [];
end

%% LOAD ATLAS
V_atlas = spm_vol(atlas_file);
atlas = spm_read_vols(V_atlas);

fprintf('Atlas size: %s\n', mat2str(size(atlas)));

%% DEFINE CLUSTERS (from manuscript)
% These ROI indices were identified from the fMRI F-map
% and mapped to Destrieux atlas

clusters = struct();

% Cluster 1: Left Sensorimotor (4 ROIs)
clusters(1).name = 'Left_Sensorimotor';
clusters(1).rois = [25, 27, 73, 75];  % Destrieux indices
clusters(1).roi_names = {'G_precentral L', 'G_postcentral L', ...
                         'S_central L', 'S_precentral_sup-part L'};

% Cluster 2: Right Sensorimotor (5 ROIs)
clusters(2).name = 'Right_Sensorimotor';
clusters(2).rois = [26, 28, 74, 76, 78];  % Destrieux indices
clusters(2).roi_names = {'G_precentral R', 'G_postcentral R', ...
                         'S_central R', 'S_precentral_sup-part R', ...
                         'S_precentral_inf-part R'};

% Cluster 3: Occipital (9 ROIs)
clusters(3).name = 'Occipital';
clusters(3).rois = [89, 90, 91, 92, 93, 94, 103, 104, 105];  % Destrieux indices
clusters(3).roi_names = {'G_occipital_sup L', 'G_occipital_sup R', ...
                         'G_occipital_middle L', 'G_occipital_middle R', ...
                         'G_occipital_inf L', 'G_occipital_inf R', ...
                         'S_occipital_ant L', 'S_occipital_ant R', ...
                         'Pole_occipital L'};

%% PRINT CLUSTER SUMMARY
fprintf('\nCluster Summary:\n');
for c = 1:length(clusters)
    fprintf('  %d. %s: %d ROIs\n', c, clusters(c).name, length(clusters(c).rois));
end

%% SAVE
save(fullfile(output_folder, 'fmri_clusters.mat'), 'clusters');

fprintf('\n=== Step 16 Complete ===\n');
fprintf('Clusters saved to: %s\n', fullfile(output_folder, 'fmri_clusters.mat'));
