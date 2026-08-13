%% =========================================================================
%  s03_plot_ocular_ic_topographies.m
%
%  Supplementary figure: the blink and lateral eye components that the
%  preprocessing pipeline removes, shown for a fixed subset of participants.
%
%  INPUT
%    cfg.eeg   full ICA decompositions
%              (S####_Filters_processed_trials_precut_ICA.set, 64 channels x
%              64 components).
%    The manual component table below (one blink and one lateral eye component
%    per participant; 57 participants in total).
%
%  OUTPUT
%    cfg.out/ocular_ic_topographies.png   (400 dpi)
%    cfg.out/ocular_ic_topographies.pdf   (vector)
%
%  LAYOUT
%    Five columns (participants) x two blocks.
%      a  blink components
%      b  lateral eye components of the SAME participant, same column
%    Each map is titled with its participant identifier, so the a/b pairing
%    within a column is verifiable on the figure itself.
%
%  PARTICIPANT SELECTION
%    A fixed rule rather than a choice made on appearance: rows 1, 15, 29, 43
%    and 57 of the component table, i.e. evenly spaced across the cohort in
%    acquisition order. Where a participant contributed more than one component
%    of a class, the first is shown.
%
%  TOPOGRAPHIES
%    Columns of icawinv from the full decomposition, normalised to unit maximum
%    absolute weight. The sign of an independent component is arbitrary, so a
%    display convention is applied: blink maps are signed so that the strongest
%    weight is positive, lateral maps so that the positive pole falls on the
%    right hemisphere. Drawing style matches the topography panel of the
%    main-text ERP figure: electrodes marked as small black dots, head outline
%    thickened, jet colormap, colour limits fixed at [-1 1].
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

set(0,'DefaultFigureVisible','off');
set(groot,'defaultAxesFontName','Arial','defaultTextFontName','Arial');
if exist('topoplot', 'file') ~= 2
    addpath(cfg.eeglab);
    eeglab('nogui');
end

STEM = '_Filters_processed_trials_precut_ICA.set';

% {participant ID, blink component(s), lateral eye component(s)}
eyeTable = {09 2 12; 11 2 11; 12 2 5; 14 1 5; 15 2 3; 16 2 9; 17 2 20; 18 3 14; ...
    20 1 5; 21 2 3; 22 2 6; 23 2 7; 24 2 6; 25 2 13; 26 2 7; 27 4 11; 28 2 7; ...
    29 2 [4,5]; 30 2 5; 31 1 7; 32 3 19; 33 3 4; 34 2 5; 35 1 4; 36 4 15; ...
    37 [1,2] [4,6]; 38 2 3; 39 2 12; 40 [3,5] 2; 41 4 5; 42 1 6; 43 2 13; ...
    44 2 [4,8]; 45 1 5; 46 1 3; 47 1 6; 48 2 4; 49 2 5; 50 2 6; 51 4 2; 52 2 10; ...
    53 5 13; 54 2 10; 55 1 4; 56 1 3; 57 1 6; 58 1 8; 59 1 9; 60 2 3; 61 2 4; ...
    62 2 5; 63 1 7; 64 1 5; 65 1 3; 66 2 [11,14]; 67 1 4; 68 2 7};

SELROWS = [1 15 29 43 57];
HEAD_LINE_WIDTH = 6;      % head outline, matching the main-text topography panel
MARKER_SIZE     = 12;     % electrode marker
SUBJ_FS  = 8.5;           % participant label above each map

%% ---------------- gather -----------------------------------------------------
D = struct([]);
for c = 1:numel(SELROWS)
    sid = eyeTable{SELROWS(c),1};
    S = load(fullfile(cfg.eeg, sprintf('S%04d%s',sid,STEM)),'-mat');
    cl = S.chanlocs(S.icachansind);
    for kind = 1:2
        ic = eyeTable{SELROWS(c), kind+1}; ic = ic(1);
        D(c,kind).sid = sprintf('S%04d', sid);
        w  = S.icawinv(:,ic);
        wn = w / max(abs(w));
        if kind == 1
            [~,imx] = max(abs(wn)); sg = sign(wn(imx));
        else
            [~,ip] = max(wn); sg = 1; if cl(ip).Y > 0, sg = -1; end
        end
        D(c,kind).w   = wn * sg;
        D(c,kind).cl  = cl;
    end
    fprintf('loaded S%04d\n', sid);
end

%% ---------------- figure -----------------------------------------------------
f = figure('Position',[50 50 1020 470],'Color','w');
mL=.030; mR=.095; gX=.014;
cW=(1-mL-mR-4*gX)/5;
TOPO_H=.360;
yTopo=[.520 .075];

for kind = 1:2
  for c = 1:5
      x0 = mL + (c-1)*(cW+gX);
      d  = D(c,kind);

      axes('Position',[x0, yTopo(kind), cW, TOPO_H]);
      topoplot(d.w, d.cl, 'electrodes','on', 'conv','off', ...
               'emarker',{'.','k',MARKER_SIZE,1}, 'maplimits',[-1 1]);
      lo = findobj(gca,'Type','line','Color','k');
      for q=1:numel(lo), set(lo(q),'LineWidth',HEAD_LINE_WIDTH); end
      title(d.sid,'FontSize',SUBJ_FS,'FontWeight','normal','Color','k');
  end
end

colormap(jet); clim([-1 1]);
cb = colorbar('Position',[1-mR+.026 .26 .015 .44]);
cb.Ticks=[-1 0 1]; cb.TickLabels={'-1','0','+1'};
cb.Label.String='normalised weight'; cb.Label.FontSize=8.5; cb.FontSize=8;

annotation('textbox',[.001 .945 .05 .050],'String','a','EdgeColor','none', ...
    'FontWeight','bold','FontSize',13,'FontName','Arial');
annotation('textbox',[.001 .500 .05 .050],'String','b','EdgeColor','none', ...
    'FontWeight','bold','FontSize',13,'FontName','Arial');

pngFile = fullfile(cfg.out, 'ocular_ic_topographies.png');
pdfFile = fullfile(cfg.out, 'ocular_ic_topographies.pdf');
exportgraphics(f, pngFile, 'Resolution', 400);
exportgraphics(f, pdfFile, 'ContentType','vector');
fprintf('wrote %s\nwrote %s\n', pngFile, pdfFile);
