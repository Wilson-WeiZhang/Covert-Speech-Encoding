%% =========================================================================
%  s02_plot_addback_topographies.m
%
%  Plot the scalp topography of every component that each add-back variant
%  reinstates, so that the composition of the three variants can be inspected
%  directly.
%
%  INPUT
%    cfg.eeg   full ICA decompositions
%              (S####_Filters_processed_trials_precut_ICA.set, 64 channels x
%              64 components) with their ICLabel posteriors. Topographies are
%              taken from the full decomposition, not from the cleaned files.
%    Component sets: lateral eye and blink from the manual table below;
%    muscle from ICLabel with a Muscle posterior above TH_MUSCLE, the rule the
%    preprocessing pipeline uses for the muscle variant.
%
%  OUTPUT
%    cfg.out/topography_lateral_eye.png
%    cfg.out/topography_blink.png
%    cfg.out/topography_muscle.png
%    Each figure has one row per displayed participant and up to six component
%    maps per row, titled with the participant, the component index, the most
%    probable ICLabel class with its posterior, and the Eye posterior.
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

if exist('pop_loadset', 'file') ~= 2
    addpath(cfg.eeglab);
    eeglab('nogui');
end

STEM = '_Filters_processed_trials_precut_ICA.set';
TH_MUSCLE = 0.9;   % ICLabel Muscle posterior above which a component is taken as muscle

% {participant ID, blink component(s), lateral eye component(s)}
eyeTable = {09 2 12; 11 2 11; 12 2 5; 14 1 5; 15 2 3; 16 2 9; 17 2 20; 18 3 14; ...
    20 1 5; 21 2 3; 22 2 6; 23 2 7; 24 2 6; 25 2 13; 26 2 7; 27 4 11; 28 2 7; ...
    29 2 [4,5]; 30 2 5; 31 1 7; 32 3 19; 33 3 4; 34 2 5; 35 1 4; 36 4 15; ...
    37 [1,2] [4,6]; 38 2 3; 39 2 12; 40 [3,5] 2; 41 4 5; 42 1 6; 43 2 13; ...
    44 2 [4,8]; 45 1 5; 46 1 3; 47 1 6; 48 2 4; 49 2 5; 50 2 6; 51 4 2; 52 2 10; ...
    53 5 13; 54 2 10; 55 1 4; 56 1 3; 57 1 6; 58 1 8; 59 1 9; 60 2 3; 61 2 4; ...
    62 2 5; 63 1 7; 64 1 5; 65 1 3; 66 2 [11,14]; 67 1 4; 68 2 7};

% participants displayed, one row each
show = [9 23 29 37 40 53];

for fig = 1:3
    switch fig
        case 1, tag = 'lateral eye (manual)'; fname = 'topography_lateral_eye.png';
        case 2, tag = 'blink (manual)';       fname = 'topography_blink.png';
        case 3, tag = sprintf('muscle (ICLabel > %.2f)', TH_MUSCLE); fname = 'topography_muscle.png';
    end

    f = figure('Position', [50 50 1500 950], 'Color', 'w');
    tl = tiledlayout(numel(show), 6, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(tl, sprintf('Components added back by the "%s" variant', tag), ...
        'FontSize', 15, 'FontWeight', 'bold');

    for r = 1:numel(show)
        sid = show(r);
        fname_set = sprintf('S%04d%s', sid, STEM);
        p = fullfile(cfg.eeg, fname_set);
        if ~exist(p, 'file')
            fprintf('missing %s\n', p); continue
        end
        EEG = pop_loadset('filename', fname_set, 'filepath', cfg.eeg, 'loadmode', 'info');

        row = find(cell2mat(eyeTable(:,1)) == sid, 1);
        C = EEG.etc.ic_classification.ICLabel.classifications;
        switch fig
            case 1, ics = eyeTable{row, 3};
            case 2, ics = eyeTable{row, 2};
            case 3, ics = find(C(:,2) > TH_MUSCLE)';
        end

        for c = 1:6
            nexttile;
            if c > numel(ics)
                axis off; continue
            end
            k = ics(c);
            topoplot(EEG.icawinv(:, k), EEG.chanlocs, 'electrodes', 'off', ...
                     'style', 'both', 'shading', 'interp');
            [~, top] = max(C(k,:));
            cls = {'Brain','Muscle','Eye','Heart','Line','Chan','Other'};
            title(sprintf('S%04d IC%d\n%s %.2f | Eye %.2f', sid, k, cls{top}, ...
                  C(k,top), C(k,3)), 'FontSize', 9, 'FontWeight', 'normal');
        end
    end
    outFile = fullfile(cfg.out, fname);
    exportgraphics(f, outFile, 'Resolution', 150);
    close(f);
    fprintf('wrote %s\n', outFile);
end

% ---- how many components the manual table holds, cohort-wide ----------------
nb = 0; nl = 0;
for r = 1:size(eyeTable,1)
    nb = nb + numel(eyeTable{r,2});
    nl = nl + numel(eyeTable{r,3});
end
fprintf('\nmanual table totals: %d blink, %d lateral over %d participants\n', ...
    nb, nl, size(eyeTable,1));
