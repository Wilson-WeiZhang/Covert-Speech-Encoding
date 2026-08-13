%% =========================================================================
%  s06_verify_source_reconstruction.m
%
%  Check that the control run of the add-back source export reproduces the
%  published source data.
%
%  The control variant (AB_VARIANT=standard in s04, AB_SET=brain in s05)
%  reinstates no components, so its output must equal the published source
%  data. This script compares the two files for one participant: trial count,
%  trial labels, ROI labels, and every sample of every trial. It is the
%  external gate on the source export; the add-back datasets are only
%  comparable with the published classification results if it passes.
%
%  The cleaned EEG is stored in single precision, so the two paths need not
%  agree bit for bit. Script s07 therefore repeats the comparison on the
%  classifier output, which is the quantity the analysis depends on.
%
%  INPUT
%    cfg.out/verify_standard_dest   control-run output (override with AB_VERIFY_DIR)
%    cfg.source                     published source data
%
%  OUTPUT
%    printed report: maximum absolute and relative difference, and a verdict
%
%  ENV
%    AB_VERIFY_SUBJ  participant to check (default Subject09)
%    AB_VERIFY_DIR   absolute path of the control-run directory
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

subj = getenv('AB_VERIFY_SUBJ');
if isempty(subj), subj = 'Subject09'; end

vdir = getenv('AB_VERIFY_DIR');
if isempty(vdir), vdir = fullfile(cfg.out, 'verify_standard_dest'); end
fnew = fullfile(vdir, [subj '_sLORETA_raw.mat']);
fpub = fullfile(cfg.source, [subj '_sLORETA_raw.mat']);
fprintf('control   : %s\n', fnew);
fprintf('published : %s\n\n', fpub);
assert(isfile(fnew), 'control run output not found');
assert(isfile(fpub), 'published file not found');

N = load(fnew);
P = load(fpub);

fprintf('trials      control %d   published %d\n', numel(N.condition_data), numel(P.condition_data));
assert(numel(N.condition_data) == numel(P.condition_data), 'trial count differs');

% trial labels
sameLab = isequal(N.condition_data_type, P.condition_data_type);
fprintf('labels identical : %d\n', sameLab);
if ~sameLab
    for k = 1:min(5, numel(N.condition_data_type))
        fprintf('   %2d  control "%s"  published "%s"\n', ...
            k, N.condition_data_type{k}, P.condition_data_type{k});
    end
end

% ROI labels
newLab = N.roiindex(:,2);
pubLab = P.roiindex(:,2);
fprintf('ROI count   control %d   published %d\n', numel(newLab), numel(pubLab));
if numel(newLab) == numel(pubLab)
    fprintf('ROI labels identical : %d\n', isequal(newLab, pubLab));
    dl = find(~cellfun(@isequal, newLab, pubLab));
    if ~isempty(dl)
        for k = dl(1:min(5,end))'
            fprintf('   ROI %3d  control "%s"  published "%s"\n', k, newLab{k}, pubLab{k});
        end
    end
end

% numeric comparison
maxAbs = 0; maxRel = 0; worst = 0;
for k = 1:numel(N.condition_data)
    a = N.condition_data{k};
    b = P.condition_data{k};
    if ~isequal(size(a), size(b))
        fprintf('*** trial %d size mismatch: %s vs %s\n', k, mat2str(size(a)), mat2str(size(b)));
        continue
    end
    m = max(abs(a(:) - b(:)));
    if m > maxAbs, maxAbs = m; worst = k; end
    sc = max(abs(b(:)));
    if sc > 0, maxRel = max(maxRel, m / sc); end
end
fprintf('\nmax |control - published| : %.6g   (worst trial %d)\n', maxAbs, worst);
fprintf('max relative difference   : %.6g\n', maxRel);
fprintf('data scale (max |published|) : %.6g\n', max(abs(P.condition_data{1}(:))));

if maxAbs == 0
    fprintf('\nVERDICT: bit-identical. The source export reproduces the published pipeline.\n');
elseif maxRel < 1e-10
    fprintf('\nVERDICT: identical to numerical precision. Safe to proceed.\n');
else
    fprintf('\nVERDICT: values differ. Resolve with the downstream check in s07.\n');
end
