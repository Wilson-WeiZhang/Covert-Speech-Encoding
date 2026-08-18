%% =========================================================================
%  s01_build_union_component_sets.m
%
%  Build the union ocular component sets: components flagged automatically by
%  ICLabel together with the manually labelled ones, split into a blink and a
%  lateral-eye subset.
%
%  INPUT
%    cfg.eeg   full ICA decompositions, one per participant
%              (S####_Filters_processed_trials_precut_ICA.set, 64 channels x
%              64 components), each carrying its ICLabel posterior matrix.
%    The manual component table below, one blink and one lateral-eye component
%    per participant, as used by the preprocessing pipeline.
%
%  OUTPUT
%    cfg.out/union_component_metrics.mat   topography metrics of every
%                                          component considered (both passes)
%    cfg.out/union_component_sets.mat      struct array S with one entry per
%                                          participant: .sid .blink .lateral
%                                          .auto_blink .auto_lateral
%                                          (second pass only)
%
%  METHOD
%    The manual table holds exactly one component of each class per
%    participant, so it is a quota rather than an exhaustive labelling: across
%    the cohort ICLabel assigns an Eye posterior above TH_EYE to further
%    components that the table does not list. This script takes the union of
%    the two sources and assigns the components contributed by ICLabel alone to
%    the blink or the lateral subset from their scalp topography. Manual labels
%    are never reassigned.
%
%    Blinks form a compact topographic class: strong frontopolar loading and a
%    steep anterior-posterior gradient with no left-right structure. Two
%    metrics describe it, both computed from the component's scalp map w:
%      |corr(w, X)|        absolute correlation between the component weights
%                          and the anterior-posterior electrode coordinate
%      frontopolar share   mean absolute weight over the frontopolar
%                          electrodes divided by the mean absolute weight over
%                          all electrodes
%    A component is a blink when both metrics reach their threshold; every
%    other ocular component is lateral.
%
%  TWO PASSES
%    Leaving AB_BLINK_AP or AB_BLINK_FP unset runs the diagnostic pass, which
%    prints the distribution of both metrics over the manually labelled blink
%    and lateral components and saves them. The thresholds are read off those
%    distributions, on which the two manually labelled classes separate without
%    overlap. Setting both thresholds runs the second pass, which applies the
%    rule and writes the component sets.
%
%  ENV
%    AB_BLINK_AP  minimum |corr(w, anterior-posterior axis)| for a blink
%    AB_BLINK_FP  minimum frontopolar weight share for a blink
%
%    The component sets reported in the paper were written with
%    AB_BLINK_AP=0.65 AB_BLINK_FP=0, read off the diagnostic pass.
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

if exist('pop_loadset', 'file') ~= 2
    addpath(cfg.eeglab);
    eeglab('nogui');
end

STEM   = '_Filters_processed_trials_precut_ICA.set';
TH_EYE = 0.9;   % ICLabel Eye posterior above which a component is taken as ocular

% blink cluster thresholds; leave either one unset for the diagnostic pass
BLINK_AP = str2double(getenv('AB_BLINK_AP'));   % min |corr(weight, A-P axis)|
BLINK_FP = str2double(getenv('AB_BLINK_FP'));   % min frontopolar share
DIAG = isnan(BLINK_AP) || isnan(BLINK_FP);

FRONTPOLE = {'Fp1','Fp2','Fpz','AF7','AF8','AF3','AF4'};

% {participant ID, blink component(s), lateral eye component(s)}
eyeTable = {09 2 12; 11 2 11; 12 2 5; 14 1 5; 15 2 3; 16 2 9; 17 2 20; 18 3 14; ...
    20 1 5; 21 2 3; 22 2 6; 23 2 7; 24 2 6; 25 2 13; 26 2 7; 27 4 11; 28 2 7; ...
    29 2 [4,5]; 30 2 5; 31 1 7; 32 3 19; 33 3 4; 34 2 5; 35 1 4; 36 4 15; ...
    37 [1,2] [4,6]; 38 2 3; 39 2 12; 40 [3,5] 2; 41 4 5; 42 1 6; 43 2 13; ...
    44 2 [4,8]; 45 1 5; 46 1 3; 47 1 6; 48 2 4; 49 2 5; 50 2 6; 51 4 2; 52 2 10; ...
    53 5 13; 54 2 10; 55 1 4; 56 1 3; 57 1 6; 58 1 8; 59 1 9; 60 2 3; 61 2 4; ...
    62 2 5; 63 1 7; 64 1 5; 65 1 3; 66 2 [11,14]; 67 1 4; 68 2 7};

N_MANUAL_BLINK   = sum(cellfun(@numel, eyeTable(:,2)));
N_MANUAL_LATERAL = sum(cellfun(@numel, eyeTable(:,3)));

rec = [];   % [sid, ic, rAP, rLR, frontshare, pEye, pBrain, srcflag]
            % srcflag 1=manual blink, 2=manual lateral, 0=automatic only

for r = 1:size(eyeTable,1)
    sid  = eyeTable{r,1};
    manB = eyeTable{r,2}(:)';
    manL = eyeTable{r,3}(:)';

    EEG = pop_loadset('filename', sprintf('S%04d%s', sid, STEM), ...
                      'filepath', cfg.eeg, 'loadmode', 'info');
    C = EEG.etc.ic_classification.ICLabel.classifications;
    autoE = find(C(:,3) > TH_EYE)';

    n = numel(EEG.chanlocs);
    X = nan(n,1); Y = nan(n,1);
    for c = 1:n
        if ~isempty(EEG.chanlocs(c).X), X(c) = EEG.chanlocs(c).X; end
        if ~isempty(EEG.chanlocs(c).Y), Y(c) = EEG.chanlocs(c).Y; end
    end
    isfp = ismember({EEG.chanlocs.labels}, FRONTPOLE)';
    ok = ~isnan(X) & ~isnan(Y);

    for k = unique([autoE, manB, manL])
        w = EEG.icawinv(:,k);
        a = abs(w) / max(abs(w));
        rAP = abs(corr(w(ok), X(ok)));
        rLR = abs(corr(w(ok), Y(ok)));
        fp  = mean(a(isfp)) / mean(a);
        if ismember(k, manB),     flag = 1;
        elseif ismember(k, manL), flag = 2;
        else,                     flag = 0;
        end
        rec(end+1,:) = [sid, k, rAP, rLR, fp, C(k,3), C(k,1), flag]; %#ok<SAGROW>
    end
end

mb = rec(rec(:,8)==1, :);
ml = rec(rec(:,8)==2, :);
au = rec(rec(:,8)==0, :);

fprintf('\n=== metric distributions ===\n');
fprintf('%-16s %4s | %-22s | %-22s | %s\n', 'group', 'n', '|corr A-P|', 'frontopolar share', '|corr L-R|');
show = @(tag, M) fprintf('%-16s %4d | %5.2f [%4.2f %4.2f] p5 %4.2f | %5.2f [%4.2f %4.2f] p5 %4.2f | %5.2f\n', ...
    tag, size(M,1), median(M(:,3)), min(M(:,3)), max(M(:,3)), prctile(M(:,3),5), ...
    median(M(:,5)), min(M(:,5)), max(M(:,5)), prctile(M(:,5),5), median(M(:,4)));
show('manual blink',   mb);
show('manual lateral', ml);
show('automatic only', au);

fprintf('\n=== manual blink sorted by |corr A-P| (lowest 8) ===\n');
[~,i] = sort(mb(:,3));
for j = i(1:min(8,end))'
    fprintf('  S%04d IC%-3d  AP %.2f  FP %.2f  LR %.2f  Eye %.2f\n', mb(j,1), mb(j,2), mb(j,3), mb(j,5), mb(j,4), mb(j,6));
end
fprintf('=== manual lateral sorted by |corr A-P| (highest 8) ===\n');
[~,i] = sort(ml(:,3), 'descend');
for j = i(1:min(8,end))'
    fprintf('  S%04d IC%-3d  AP %.2f  FP %.2f  LR %.2f  Eye %.2f\n', ml(j,1), ml(j,2), ml(j,3), ml(j,5), ml(j,4), ml(j,6));
end

if DIAG
    fprintf('\nDIAGNOSTIC PASS. Set AB_BLINK_AP and AB_BLINK_FP to write the sets.\n');
    save(fullfile(cfg.out, 'union_component_metrics.mat'), 'rec');
    return
end

% ---- second pass: apply the rule --------------------------------------------
fprintf('\n=== applying blink cluster rule: |corr A-P| >= %.2f AND frontopolar share >= %.2f ===\n', ...
    BLINK_AP, BLINK_FP);
S = struct('sid', {}, 'blink', {}, 'lateral', {}, 'auto_blink', {}, 'auto_lateral', {});
for r = 1:size(eyeTable,1)
    sid  = eyeTable{r,1};
    manB = eyeTable{r,2}(:)';
    manL = eyeTable{r,3}(:)';
    mine = rec(rec(:,1)==sid, :);
    aut  = mine(mine(:,8)==0, :);
    isB  = aut(:,3) >= BLINK_AP & aut(:,5) >= BLINK_FP;
    nb   = aut(isB, 2)';
    nl   = aut(~isB, 2)';
    S(end+1) = struct('sid', sid, 'blink', sort([manB nb]), 'lateral', sort([manL nl]), ...
                      'auto_blink', sort(nb), 'auto_lateral', sort(nl)); %#ok<SAGROW>
    fprintf('S%04d  blink %-14s (+%s)   lateral %-14s (+%s)\n', sid, ...
        mat2str(sort([manB nb])), mat2str(nb), mat2str(sort([manL nl])), mat2str(nl));
end
tb = sum(arrayfun(@(s) numel(s.blink), S));
tl = sum(arrayfun(@(s) numel(s.lateral), S));
fprintf('\nunion: blink %d (manual %d + automatic %d), lateral %d (manual %d + automatic %d), total %d\n', ...
    tb, N_MANUAL_BLINK, tb-N_MANUAL_BLINK, tl, N_MANUAL_LATERAL, tl-N_MANUAL_LATERAL, tb+tl);
fprintf('mean per participant: blink %.2f, lateral %.2f\n', tb/numel(S), tl/numel(S));

outFile = fullfile(cfg.out, 'union_component_sets.mat');
save(outFile, 'S', 'rec', 'BLINK_AP', 'BLINK_FP');
fprintf('wrote %s\n', outFile);
