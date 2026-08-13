%% s17_export_rdms.m
%  Export the neural RDMs computed by s16_attribute_rsa.m to plain CSV, which
%  is the interchange format for the Python steps (s18-s20).
%
%  Each row holds the 10 unique pairwise dissimilarities of one
%  (participant, window) RDM. The column order follows MATLAB's
%  find(triu(true(5),1)), which is COLUMN-major:
%      1-2, 1-3, 2-3, 1-4, 2-4, 3-4, 1-5, 2-5, 3-5, 4-5
%  The downstream scripts rebuild the attribute RDMs in this same order and
%  verify it numerically, because a row-major ordering would pair the wrong
%  phrases with the wrong dissimilarities while still producing plausible
%  values.
%
%  The visual-extent RDM is exported as well: it was measured with the MATLAB
%  font metrics and cannot be reconstructed from the phrase strings alone.
%
%  IN  : cfg.out/09_stimulus_attribute_rsa/attribute_rsa.mat
%  OUT : cfg.out/09_stimulus_attribute_rsa/
%          neural_rdm.csv               one row per participant per window
%          attribute_visual_extent.csv  the 10 pairwise values
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

D = fullfile(cfg.out, '09_stimulus_attribute_rsa');
L = load(fullfile(D, 'attribute_rsa.mat'), 'RDM', 'edges', 'ATTR', 'subs');

[n, nwin, ~, ~] = size(L.RDM);
iu = find(triu(true(5),1));

fid = fopen(fullfile(D, 'neural_rdm.csv'), 'w');
fprintf(fid, 'subject,win_start_ms,win_end_ms');
fprintf(fid, ',p%d', 1:numel(iu));
fprintf(fid, '\n');
for s = 1:n
    for k = 1:nwin
        v = squeeze(L.RDM(s,k,:,:));
        fprintf(fid, '%s,%g,%g', L.subs{s}, L.edges(k), L.edges(k+1));
        fprintf(fid, ',%.10g', v(iu));
        fprintf(fid, '\n');
    end
end
fclose(fid);

extent_idx = find(strcmp({L.ATTR.name}, 'visual_extent'), 1);
e = L.ATTR(extent_idx).rdm;
fid = fopen(fullfile(D, 'attribute_visual_extent.csv'), 'w');
fprintf(fid, 'pair,value\n');
for t = 1:numel(iu)
    fprintf(fid, '%d,%.10g\n', t, e(iu(t)));
end
fclose(fid);

fprintf('wrote neural_rdm.csv (%d rows) and attribute_visual_extent.csv\n', n*nwin);
fprintf('pair order: ');
[r, c] = ind2sub([5 5], iu);
fprintf('%d-%d ', [r c].');
fprintf('\n');
