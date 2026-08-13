%% s16_attribute_rsa.m
%  Representational similarity analysis between the covert-speech neural
%  geometry of the five phrases and models built from their stimulus
%  attributes.
%
%  QUESTION
%    Phrase discriminability could in principle reflect low-level properties of
%    the written cues rather than speech planning. This script asks whether the
%    neural dissimilarity structure of the five phrases is better explained by
%    VISUAL/ORTHOGRAPHIC attributes (letter count, word count, rendered width)
%    or by PHONOLOGICAL/ARTICULATORY attributes (syllable count, place and
%    manner of the initial phoneme).
%
%  PHRASE MAPPING
%    word_id 1..5 = Go There / Distract Target / Follow Me / Explore Here /
%                   Terminate
%    The mapping is corroborated by the recorded speech durations of the overt
%    blocks (475/575/578/754/775 ms for Terminate/Follow Me/Go There/
%    Explore Here/Distract Target): across the four two-word phrases duration
%    is monotonic in letter count, and Terminate, the only one-word phrase, is
%    shortest because it carries no inter-word pause.
%
%  NEURAL RDM
%    1 - Spearman correlation between phrase-averaged, baseline-corrected
%    source patterns. This is the same RDM definition used for the fMRI-EEG
%    RSA elsewhere in the paper, so both analyses share one definition.
%
%  ATTRIBUTE RDMs
%    Numeric attributes give |a_i - a_j|; categorical attributes give 0 for a
%    match and 1 otherwise. The two family composites are the mean of the
%    z-scored member RDMs.
%
%  INFERENCE
%    Per window, the neural and attribute RDMs are correlated (Spearman over
%    the 10 unique phrase pairs) within participant and tested against zero
%    across participants. The permutation null relabels the five phrases of the
%    ATTRIBUTE RDM, which is the correct exchangeability unit because the
%    neural RDMs are held fixed. Taking the maximum over windows in each
%    permutation controls the family-wise error rate across windows.
%
%  IN  : cfg.source        source-localised covert data (Destrieux, 148 ROIs)
%  OUT : cfg.out/09_stimulus_attribute_rsa/
%          rsa_<attribute>.csv   per-window rho, t, parametric p, FWE p
%          rsa_summary.txt       human-readable summary
%          attribute_rsa.mat     neural RDMs and attribute definitions
% =========================================================================

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config'));
cfg = set_paths();

outdir = fullfile(cfg.out, '09_stimulus_attribute_rsa');
if ~isfolder(outdir), mkdir(outdir); end

FS        = 250;              % Hz, source data sampling rate
PRE_SMP   = 0.5 * FS;         % pre-stimulus baseline samples
POST_SMP  = 1.5 * FS;         % post-stimulus samples retained
N_ROI     = 148;              % Destrieux cortical ROIs
STEP_MS   = 100;              % analysis window width
N_PERM    = 1000;             % permutations for the window-wise FWE test

edges = 0:STEP_MS:1500;
nwin  = numel(edges) - 1;

%% ---------------- stimulus attributes ----------------
phrases = {'Go There','Distract Target','Follow Me','Explore Here','Terminate'};
words   = [2 2 2 2 1];
letters = [7 14 8 11 9];          % letters only, excluding the space
syll    = [2 4 3 3 3];

% Initial phoneme of each phrase:
%   Go /g/ velar plosive | Distract /d/ alveolar plosive | Follow /f/
%   labiodental fricative | Explore /I/ vowel | Terminate /t/ alveolar plosive
place  = {'velar','alveolar','labiodental','vowel','alveolar'};
manner = {'plosive','plosive','fricative','vowel','plosive'};

% Visual extent: width of each phrase rendered in Arial. Absolute visual angle
% is not recoverable because font size, viewing distance and monitor geometry
% were not recorded, but relative width is well defined and is all an RDM uses.
extent = arial_widths(phrases);

ATTR = struct('name', {}, 'family', {}, 'rdm', {});
ATTR(end+1) = mk('letter_count',  'visual',       absdiff(letters));
ATTR(end+1) = mk('word_count',    'visual',       absdiff(words));
ATTR(end+1) = mk('visual_extent', 'visual',       absdiff(extent));
ATTR(end+1) = mk('syllables',     'phonological', absdiff(syll));
ATTR(end+1) = mk('onset_place',   'articulatory', catdiff(place));
ATTR(end+1) = mk('onset_manner',  'articulatory', catdiff(manner));
ATTR(end+1) = mk('VISUAL_combined',     'composite', ...
                 comb({absdiff(letters), absdiff(words), absdiff(extent)}));
ATTR(end+1) = mk('PHON_ARTIC_combined', 'composite', ...
                 comb({absdiff(syll), catdiff(place), catdiff(manner)}));

fprintf('phrases : %s\n', strjoin(phrases, ' | '));
fprintf('letters : %s\nwords   : %s\nsyll    : %s\nextent  : %s\n', ...
        mat2str(letters), mat2str(words), mat2str(syll), mat2str(round(extent,1)));
fprintf('windows : %d x %d ms over 0-1500 ms\n', nwin, STEP_MS);
fprintf('attribute RDMs: %d\n\n', numel(ATTR));

%% ---------------- participants ----------------
f = dir(fullfile(cfg.source, 'Subject*_sLORETA_raw.mat'));
subs = cellfun(@(x) x(1:strfind(x,'_sLORETA')-1), {f.name}, 'UniformOutput', false);
n = numel(subs);
fprintf('participants: %d\n', n);

pool = gcp('nocreate');
if isempty(pool), parpool('Processes', cfg.n_workers); end

srcdir = cfg.source;

%% ---------------- neural RDMs, per participant per window ----------------
RDM = nan(n, nwin, 5, 5);
parfor ii = 1:n
    S = load(fullfile(srcdir, [subs{ii} '_sLORETA_raw.mat']), ...
             'condition_data', 'condition_data_type');
    trials = S.condition_data;
    labels = S.condition_data_type;
    keep = ~cellfun(@isempty, trials);
    trials = trials(keep);
    labels = labels(keep);

    % Trial labels read 'C 1_u_1_b_3'; character 3 is the phrase id.
    w = cellfun(@(s) str2double(s(3)), labels(:)');

    r_i = nan(nwin, 5, 5);
    for k = 1:nwin
        s0 = round(edges(k)   / 1000 * FS) + 1;
        s1 = round(edges(k+1) / 1000 * FS);

        P = zeros(5, N_ROI * (s1 - s0 + 1));
        complete = true;
        for p = 1:5
            sel = find(w == p);
            if isempty(sel), complete = false; break, end
            acc = zeros(N_ROI, s1 - s0 + 1);
            for t = sel
                nd = trials{t};
                base = mean(nd(1:N_ROI, 1:PRE_SMP), 2);
                ev = nd(1:N_ROI, (PRE_SMP+1):(PRE_SMP+POST_SMP)) - base;
                acc = acc + ev(:, s0:s1);
            end
            acc = acc / numel(sel);
            P(p,:) = reshape(acc.', 1, []);   % ROI x time flattened to a pattern
        end
        if complete, r_i(k,:,:) = spearman_rdm(P); end
    end
    RDM(ii,:,:,:) = r_i;
    fprintf('%s done\n', subs{ii});
end

%% ---------------- RSA: neural RDM vs each attribute RDM ----------------
iu = find(triu(true(5),1));                  % the 10 unique phrase pairs
res = struct('name',{},'family',{},'r',{},'p',{},'t',{},'pperm',{});
lines = {sprintf('%-22s %-13s %s', 'attribute', 'family', ...
         sprintf('%9s', string(edges(1:end-1)+STEP_MS/2)))};

for a = 1:numel(ATTR)
    av = ATTR(a).rdm(iu);
    R = nan(n, nwin);
    for k = 1:nwin
        for ii = 1:n
            nv = squeeze(RDM(ii,k,:,:));
            if all(isnan(nv(:))), continue, end
            R(ii,k) = corr(nv(iu), av, 'Type','Spearman');
        end
    end
    m = mean(R, 1, 'omitnan');
    [~, p, ~, st] = ttest(R);

    % Null: relabel the five phrases of the attribute RDM, then take the
    % maximum across windows so the resulting p is FWE-corrected over windows.
    null_max = zeros(N_PERM,1);
    for q = 1:N_PERM
        o = randperm(5);
        pv = ATTR(a).rdm(o,o); pv = pv(iu);
        mq = zeros(1,nwin);
        for k = 1:nwin
            rr = nan(n,1);
            for ii = 1:n
                nv = squeeze(RDM(ii,k,:,:));
                if all(isnan(nv(:))), continue, end
                rr(ii) = corr(nv(iu), pv, 'Type','Spearman');
            end
            mq(k) = mean(rr, 'omitnan');
        end
        null_max(q) = max(mq);
    end
    pperm = arrayfun(@(k) (1+sum(null_max >= m(k)))/(N_PERM+1), 1:nwin);

    res(end+1) = struct('name',ATTR(a).name,'family',ATTR(a).family, ...
                        'r',m,'p',p,'t',[st.tstat],'pperm',pperm); %#ok<SAGROW>
    lines{end+1} = sprintf('%-22s %-13s %s', ATTR(a).name, ATTR(a).family, ...
                   sprintf('%9.3f', m)); %#ok<SAGROW>

    T = table((edges(1:end-1))', (edges(2:end))', m', [st.tstat]', p', pperm', ...
        'VariableNames', {'win_start_ms','win_end_ms','mean_rho','t','p_param','p_perm_fwe'});
    writetable(T, fullfile(outdir, sprintf('rsa_%s.csv', ATTR(a).name)));
end

%% ---------------- summary ----------------
fid = fopen(fullfile(outdir, 'rsa_summary.txt'), 'w');
fprintf(fid, 'RSA: neural RDM (1 - Spearman, phrase-averaged patterns) vs attribute RDMs\n');
fprintf(fid, 'N = %d, %d windows of %d ms, %d permutations (max-over-windows FWE)\n\n', ...
        n, nwin, STEP_MS, N_PERM);
fprintf(fid, 'mean Spearman rho per window (window centres in ms):\n');
for k = 1:numel(lines), fprintf(fid, '%s\n', lines{k}); fprintf('%s\n', lines{k}); end
fprintf(fid, '\nwindows surviving FWE p < .05:\n');
for a = 1:numel(res)
    s = find(res(a).pperm < 0.05);
    if isempty(s)
        L = sprintf('  %-22s none', res(a).name);
    else
        L = sprintf('  %-22s %s', res(a).name, ...
            strjoin(arrayfun(@(k) sprintf('%d-%d ms (rho=%.3f, p=%.4f)', ...
              edges(k), edges(k+1), res(a).r(k), res(a).pperm(k)), s, 'uni', 0), '; '));
    end
    fprintf(fid, '%s\n', L); fprintf('%s\n', L);
end
fclose(fid);

save(fullfile(outdir, 'attribute_rsa.mat'), ...
     'RDM', 'ATTR', 'res', 'edges', 'phrases', 'subs');
fprintf('\nwrote %s\n', fullfile(outdir, 'rsa_summary.txt'));


%% ---------------- helpers ----------------
function s = mk(name, family, rdm)
s = struct('name', name, 'family', family, 'rdm', rdm);
end

function D = absdiff(v)
% Dissimilarity of a numeric attribute: absolute difference.
v = v(:);
D = abs(v - v.');
end

function D = catdiff(c)
% Dissimilarity of a categorical attribute: 0 if the categories match, 1 if not.
n = numel(c);
D = zeros(n);
for i = 1:n
    for j = 1:n
        D(i,j) = ~strcmp(c{i}, c{j});
    end
end
end

function D = comb(list)
% Family composite: mean of the z-scored member RDMs, off-diagonal only.
n = size(list{1}, 1);
iu = find(triu(true(n),1));
Z = zeros(numel(iu), numel(list));
for k = 1:numel(list)
    v = list{k}(iu);
    Z(:,k) = (v - mean(v)) / max(std(v), eps);
end
D = zeros(n);
D(iu) = mean(Z, 2);
D = D + D.';
end

function rdm = spearman_rdm(patterns)
% RDM as 1 - Spearman correlation between condition patterns.
% Pairs that cannot be correlated are assigned maximal dissimilarity (1).
nc = size(patterns, 1);
rdm = zeros(nc);
for i = 1:nc
    for j = 1:nc
        if i == j, continue, end
        r = corr(patterns(i,:)', patterns(j,:)', 'Type', 'Spearman', 'rows', 'complete');
        if isnan(r), rdm(i,j) = 1; else, rdm(i,j) = 1 - r; end
    end
end
end

function w = arial_widths(phr)
% Rendered width of each phrase in Arial, measured with the figure text
% metrics. Only relative width is meaningful, which is what the RDM uses.
f = figure('Visible', 'off');
ax = axes('Parent', f);
w = zeros(1, numel(phr));
for i = 1:numel(phr)
    h = text(ax, 0, 0, phr{i}, 'FontName', 'Arial', 'FontSize', 40, 'Units', 'points');
    e = get(h, 'Extent');
    w(i) = e(3);
    delete(h);
end
close(f);
end
