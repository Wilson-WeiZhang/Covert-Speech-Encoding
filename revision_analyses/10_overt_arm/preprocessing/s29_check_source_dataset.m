%% ========================================================================
%  s29_check_source_dataset.m -- quality control for the overt source dataset
%
%  Read-only verification of the dataset written by s28, against the covert
%  source dataset that defines the target layout. For every subject it
%  checks that
%    - the file exists and holds exactly the five expected variables
%    - roiindex is identical to the covert one, both the scout labels and
%      the vertex lists, so ROI i means the same parcel in both conditions
%    - every trial label is an overt label of the form 'O <phrase>_u_1_b_<block>'
%    - each epoch is 148 x 500
%    - kernel_name and freq_band are 'sLORETA' and 'raw'
%  and it tabulates trial counts per phrase and per block.
%
%  Finally one trial is recomputed end to end from the cleaned .set file and
%  the imaging kernel, and compared against the stored result. This catches
%  errors that leave the file structurally valid, such as a channel or
%  vertex ordering mismatch.
%
%  INPUT   <cfg.out>/overt_source_dataset/   (from s28)
%          cfg.source                        covert source dataset
%          <cfg.out>/overt_clean/            cleaned overt epochs
%          cfg.bst_protocol/data             imaging kernels
%  OUTPUT  none; a table and a verdict are printed
%% ========================================================================

clearvars; clc;

addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'config'));
cfg = set_paths();

bsD  = fullfile(cfg.bst_protocol, 'data', filesep);
setO = fullfile(cfg.out, 'overt_clean', filesep);
pubD = fullfile(cfg.source, filesep);
ovtD = fullfile(cfg.out, 'overt_source_dataset', filesep);

pf = dir([pubD 'Subject*_sLORETA_raw.mat']);
assert(~isempty(pf), 'no covert source files in %s', pubD);
subjects = cellfun(@(x) x(1:strfind(x, '_sLORETA')-1), {pf.name}, 'UniformOutput', false);

fields_expect = {'condition_data','condition_data_type','freq_band','kernel_name','roiindex'};
fprintf('%-10s %-7s %-7s %-9s %-7s %-8s %-8s %-8s\n', ...
    'subject','nCovert','nOvert','size{1}','class','fields','roiindex','labels');
bad = {};
nO = zeros(numel(subjects), 1); nC = zeros(numel(subjects), 1);
phr = zeros(numel(subjects), 5); blk = zeros(numel(subjects), 10);

for ii = 1:numel(subjects)
    subj = subjects{ii};
    fo = [ovtD subj '_sLORETA_raw.mat'];
    if ~isfile(fo), bad{end+1} = [subj ': output missing']; continue; end %#ok<SAGROW>
    O = load(fo);
    C = load([pubD subj '_sLORETA_raw.mat'], 'roiindex', 'condition_data', 'condition_data_type');

    okF = isequal(sort(fieldnames(O))', fields_expect);
    okR = isequal(O.roiindex(:,2), C.roiindex(:,2));
    if okR
        for i = 1:size(O.roiindex, 1)
            if ~isequal(double(O.roiindex{i,1}(:)), double(C.roiindex{i,1}(:))), okR = false; break; end
        end
    end
    sz  = size(O.condition_data{1});
    lab = O.condition_data_type;
    okL = all(strncmp(lab, 'O ', 2)) && all(~cellfun(@isempty, regexp(lab, '_u_1_b_', 'once')));

    nO(ii) = numel(O.condition_data);
    nC(ii) = numel(C.condition_data);
    for t = 1:numel(lab)
        p = str2double(regexp(lab{t}, '^O\s*(\d)', 'tokens', 'once'));
        b = str2double(regexp(lab{t}, '_b_(\d+)$', 'tokens', 'once'));
        if ~isnan(p) && p >= 1 && p <= 5,  phr(ii,p) = phr(ii,p) + 1; end
        if ~isnan(b) && b >= 1 && b <= 10, blk(ii,b) = blk(ii,b) + 1; end
    end

    fprintf('%-10s %-7d %-7d %-9s %-7s %-8d %-8d %-8d\n', subj, nC(ii), nO(ii), ...
        sprintf('%dx%d', sz), class(O.condition_data{1}), okF, okR, okL);
    if ~okF, bad{end+1} = [subj ': field set differs']; end %#ok<SAGROW>
    if ~okR, bad{end+1} = [subj ': roiindex differs from covert']; end %#ok<SAGROW>
    if ~okL, bad{end+1} = [subj ': label format unexpected']; end %#ok<SAGROW>
    if ~isequal(sz, [148 500]), bad{end+1} = [subj ': epoch size ' sprintf('%dx%d', sz)]; end %#ok<SAGROW>
    if ~strcmp(O.kernel_name, 'sLORETA') || ~strcmp(O.freq_band, 'raw')
        bad{end+1} = [subj ': kernel_name/freq_band differ']; %#ok<SAGROW>
    end
end

fprintf('\n--- trial counts ---\n');
fprintf('overt : n=%d  mean=%.1f  median=%d  min=%d  max=%d  (covert always %d)\n', ...
    sum(nO > 0), mean(nO(nO > 0)), median(nO(nO > 0)), min(nO(nO > 0)), max(nO(nO > 0)), ...
    median(nC(nC > 0)));
fprintf('phrase totals 1..5 : %s\n', mat2str(sum(phr, 1)));
fprintf('block  totals 1..10: %s\n', mat2str(sum(blk, 1)));

%% ---------------- independent recompute of one trial ----------------
subj = subjects{1};
sid  = ['S00' subj(8:9)];
fprintf('\n--- independent recompute (%s, trial 1) ---\n', subj);
addpath(cfg.eeglab); eeglab nogui;

O = load([ovtD subj '_sLORETA_raw.mat']);
EEG = pop_loadset([setO sid '_Filters_overt_processed_trials_precut_ICA_ICACUT_rejectchan56_o1.set']);
sel = find(~cellfun(@isempty, regexp({EEG.event.type}, '_u_1_b_', 'once')));
kdir = [bsD subj filesep sid '_Filters_processed_trials_precut_ICA_ICACUT_rejectchan_u1_only' filesep];
if ~isfolder(kdir)
    kdir = [bsD subj filesep sid '_Filters_processed_trials_precut_ICA_ICACUT_rejectchan_o' filesep];
end
kf = dir([kdir 'results_sLORETA*']);
K = load([kf(1).folder filesep kf(1).name], 'ImagingKernel', 'GoodChannel');
seg = EEG.data(:, 1:500, sel(1));
src = K.ImagingKernel * seg(K.GoodChannel, :);
rec = zeros(148, 500);
for r = 1:148, rec(r,:) = mean(src(O.roiindex{r,1}, :), 1); end
fprintf('label file="%s"  set="%s"  match=%d\n', O.condition_data_type{1}, ...
    EEG.event(sel(1)).type, strcmp(O.condition_data_type{1}, EEG.event(sel(1)).type));
fprintf('maxAbsDiff=%.4e  corr=%.12f\n', ...
    max(abs(O.condition_data{1}(:) - rec(:))), corr(O.condition_data{1}(:), rec(:)));

fprintf('\n--- verdict ---\n');
if isempty(bad)
    fprintf('all %d subjects pass\n', numel(subjects));
else
    fprintf('%d problem(s):\n', numel(bad));
    fprintf('  %s\n', bad{:});
end
