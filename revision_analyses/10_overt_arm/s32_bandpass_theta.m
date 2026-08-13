%% =========================================================================
%  s32_bandpass_theta.m
%
%  Theta band-pass (4-8 Hz) of the overt source dataset, so that the
%  overt-versus-covert spatiotemporal comparison can also be run in the band
%  the main analysis uses. The covert theta tree is part of the data
%  distribution; this script produces its overt counterpart.
%
%  The filter is the one used for the covert band trees: a fourth-order
%  Butterworth band-pass, applied with filtfilt to each ROI time course
%  separately, so the filtered data are zero-phase. The output variable is
%  named condition_data_save, matching the covert band trees.
%
%  Resumable: subjects whose output already exists are skipped.
%
%  INPUT   cfg.source_overt/Subject##_sLORETA_raw.mat
%  OUTPUT  <cfg.out>/sourcedata_theta_overt/Subject##_sLORETA_raw.mat
% =========================================================================

clearvars; clc;

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

fprintf('=== s32: theta band-pass of the overt source dataset ===\n%s\n\n', ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));

src = fullfile(cfg.source_overt, filesep);
dst = fullfile(cfg.out, 'sourcedata_theta_overt', filesep);
if ~isfolder(dst), mkdir(dst); end

fs = 250; fLow = 4; fHigh = 8;
[bb, aa] = butter(2, [fLow fHigh]/(fs/2), 'bandpass');   % order 4 band-pass

files = dir(fullfile(src, 'Subject*_sLORETA_raw.mat'));
ns = numel(files);
fprintf('%d subjects\n', ns);
assert(ns == 57, 'expected 57 overt subjects, found %d', ns);

if isempty(gcp('nocreate')), parpool('Processes', cfg.n_workers); end
tic;
parfor s = 1:ns
    fin  = fullfile(src, files(s).name);
    fout = fullfile(dst, files(s).name);
    if isfile(fout), continue; end                      % resume

    S = load(fin, 'condition_data', 'condition_data_type', 'roiindex');
    n_trials = numel(S.condition_data);
    condition_data_save = cell(size(S.condition_data));
    for t = 1:n_trials
        td = S.condition_data{t};
        if isempty(td), condition_data_save{t} = []; continue; end
        ft = zeros(size(td));
        for roi = 1:size(td, 1)
            ft(roi, :) = filtfilt(bb, aa, td(roi, :));
        end
        condition_data_save{t} = ft;
    end
    condition_data_type = S.condition_data_type;
    roiindex = S.roiindex;
    save_v73(fout, condition_data_save, condition_data_type, roiindex);
    fprintf('[%2d/%d] %s done\n', s, ns, files(s).name);
end
fprintf('total: %.1f min\n', toc/60);

out_files = dir(fullfile(dst, 'Subject*_sLORETA_raw.mat'));
fprintf('output files: %d\n', numel(out_files));
assert(numel(out_files) == 57, 'incomplete output: %d of 57 files', numel(out_files));

function save_v73(fout, condition_data_save, condition_data_type, roiindex) %#ok<INUSD>
save(fout, 'condition_data_save', 'condition_data_type', 'roiindex', '-v7.3');
end
