function s27_source_localise()
%S27_SOURCE_LOCALISE  Project the cleaned overt EEG into source space.
%
%  Multiplies each cleaned 56-channel overt epoch by its subject's sLORETA
%  imaging kernel and averages the resulting source time courses within the
%  148 Destrieux scouts, producing one file per subject in the same layout as
%  the covert source dataset. Downstream analysis code therefore reads the
%  overt data unchanged; only the directory differs.
%
%  ATLAS SELECTION
%    The scout set is selected by atlas name in tess_cortex_pial_low.mat and
%    the script aborts unless the selected atlas contains exactly N_SCOUTS
%    (148) scouts. The order of the Atlas array varies between subjects, so
%    an index-based selection is not safe.
%
%  OUTPUT FORMAT (-v7.3, exactly five variables)
%    condition_data       nTrials x 1 cell; each cell 148 x 500 double
%                         (ROI x time, -0.5 to +1.496 s at 250 Hz)
%    condition_data_type  nTrials x 1 cell of char, e.g. 'O 1_u_1_b_1'
%    roiindex             148 x 2 cell; {i,1} = 1 x Ni Brainstorm vertex
%                         indices, {i,2} = scout label, e.g.
%                         'G_Ins_lg_and_S_cent_ins L'
%    kernel_name          'sLORETA'
%    freq_band            'raw'
%  EEG.data is cast to double before the projection so that the stored cells
%  are float64, matching the covert dataset. Files are named
%  <BrainstormSubjectName>_<kernel_name>_<freq_band>.mat, e.g.
%  Subject09_sLORETA_raw.mat.
%
%  The step is resumable and writes each file atomically (to <name>.part,
%  then renames), so an interrupted run leaves no half-written .mat behind.
%  The pool size is derived from free memory measured at runtime rather than
%  fixed, and is additionally capped by cfg.n_workers.
%
%  ENVIRONMENT (all optional)
%    OVERT_SET_GLOB     input filename pattern, one '%s' = 4-digit subject id
%                       (default 'S%s_*rejectchan56_o1.set')
%    KERNEL_STUDY_GLOB  Brainstorm study folder pattern holding the kernel
%    KERNEL_TYPE        kernel name, default 'sLORETA'
%    FREQ_BAND          label written into the output, default 'raw'
%    ATLAS_NAME         atlas to select, default 'Destrieux'
%    N_SCOUTS           expected scout count, default 148
%    TRIAL_START        first epoch sample, default 1
%    TRIAL_END          last epoch sample, default 500
%    SUBJECTS           comma-separated Brainstorm subject names to restrict to
%    FORCE              '1' recomputes subjects whose output already exists
%    MAX_WORKERS MEM_PER_WORKER_GB MEM_FRACTION MEM_CAP_GB
%
%  INPUT   <cfg.out>/overt_clean/*rejectchan56_o1.set        (from s26)
%          cfg.bst_protocol/{anat,data}                      imaging kernels
%  OUTPUT  <cfg.out>/overt_source/Subject##_sLORETA_raw.mat
%          <cfg.out>/overt_source/logs/                      manifest and per-subject logs

t_all = tic;
stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));

%% ---------------------------------------------------------------- config
here = fileparts(mfilename('fullpath'));
addpath(fullfile(fileparts(fileparts(here)), 'config'));
paths = set_paths();

cfg.clean_dir      = fullfile(paths.out, 'overt_clean');
cfg.set_glob       = env_str('OVERT_SET_GLOB',    'S%s_*rejectchan56_o1.set');
cfg.bst_root       = paths.bst_protocol;                    % holds anat/ and data/
cfg.study_glob     = env_str('KERNEL_STUDY_GLOB', '*_rejectchan_o');
cfg.kernel_type    = env_str('KERNEL_TYPE',       'sLORETA');
cfg.freq_band      = env_str('FREQ_BAND',         'raw');
cfg.atlas_name     = env_str('ATLAS_NAME',        'Destrieux');
cfg.n_scouts       = env_num('N_SCOUTS',          148);
cfg.trial_start    = env_num('TRIAL_START',       1);
cfg.trial_end      = env_num('TRIAL_END',         500);
cfg.out_dir        = fullfile(paths.out, 'overt_source');
cfg.log_dir        = fullfile(cfg.out_dir, 'logs');
cfg.eeglab_path    = paths.eeglab;
cfg.ref_dir        = paths.source;                          % covert dataset, layout reference
cfg.subjects       = env_str('SUBJECTS', '');
cfg.force          = strcmp(env_str('FORCE', '0'), '1');
cfg.max_workers    = env_num('MAX_WORKERS',       paths.n_workers);
cfg.mem_per_worker = env_num('MEM_PER_WORKER_GB', 3);       % MATLAB plus data copies
cfg.mem_fraction   = env_num('MEM_FRACTION',      0.40);    % share of available memory
cfg.mem_cap_gb     = env_num('MEM_CAP_GB',        96);      % absolute budget ceiling

if ~isfolder(cfg.out_dir), mkdir(cfg.out_dir); end
if ~isfolder(cfg.log_dir), mkdir(cfg.log_dir); end
sublog_dir = fullfile(cfg.log_dir, 'subjects');
if ~isfolder(sublog_dir), mkdir(sublog_dir); end

fprintf('===== s27_source_localise  %s =====\n', stamp);
fprintf('cleaned overt   : %s\n', cfg.clean_dir);
fprintf('set glob        : %s\n', cfg.set_glob);
fprintf('brainstorm root : %s\n', cfg.bst_root);
fprintf('kernel study    : %s / results_%s*.mat\n', cfg.study_glob, cfg.kernel_type);
fprintf('atlas           : %s (%d scouts, selected by name)\n', cfg.atlas_name, cfg.n_scouts);
fprintf('epoch samples   : %d:%d\n', cfg.trial_start, cfg.trial_end);
fprintf('out dir         : %s\n', cfg.out_dir);
fprintf('force rebuild   : %d\n', cfg.force);

%% ------------------------------------------------- memory before anything
[availGB, memRaw] = mem_available_gb();
fprintf('\n-- memory before --\n%s\navailable ~ %.1f GB\n', memRaw, availGB);

%% ---------------------------------------------------------- EEGLAB on path
ensure_eeglab(cfg.eeglab_path);
fprintf('pop_loadset     : %s\n', which('pop_loadset'));

%% ---------------------------------- print the covert dataset's layout
report_reference_layout(cfg.ref_dir);

%% ------------------------------------------------------ build subject table
subs = discover_subjects(cfg);
if isempty(subs)
    error('s27:nosubjects', ['No subjects found.\n' ...
        '  brainstorm root : %s\n  study glob      : %s\n  clean dir       : %s\n  set glob        : %s'], ...
        cfg.bst_root, cfg.study_glob, cfg.clean_dir, cfg.set_glob);
end
fprintf('\nsubjects resolved: %d\n', numel(subs));

todo = true(numel(subs), 1);
for i = 1:numel(subs)
    if ~cfg.force && isfile(subs(i).outFile)
        todo(i) = false;
    end
end
fprintf('already done     : %d   to process: %d\n', sum(~todo), sum(todo));

%% ----------------------------------------------- memory-aware parallel pool
n_todo = sum(todo);
nw = plan_workers(cfg, availGB, n_todo);
fprintf(['\nparpool workers  : %d  (available %.1f GB x %.2f / %.1f GB per worker, ' ...
         'capped by cores=%d, subjects=%d)\n'], ...
        nw, availGB, cfg.mem_fraction, cfg.mem_per_worker, feature('numcores'), n_todo);
open_pool(nw);

%% --------------------------------------------------------------- main loop
idx  = find(todo);
res  = cell(numel(idx), 1);
subs_todo = subs(idx);

parfor k = 1:numel(idx)
    s = subs_todo(k);
    r = struct('subject', s.bstName, 'status', 'error', 'msg', '', ...
               'atlas_index', NaN, 'n_scouts', NaN, 'n_trials', NaN, ...
               'n_sources', NaN, 'n_good', NaN, 'bytes', NaN, ...
               'set_file', s.setFile, 'kernel_file', s.kernelFile, ...
               'anat_file', s.anatFile, 'out_file', s.outFile, 'seconds', NaN);
    tsub = tic;
    lf = fullfile(sublog_dir, [s.bstName '.log']);
    try
        r = process_one(s, cfg, r, lf);
        r.status = 'ok';
    catch ME
        r.status = 'error';
        r.msg    = sprintf('%s | %s', ME.identifier, ME.message);
        log_line(lf, sprintf('ERROR %s', r.msg));
        fprintf('  [FAIL] %s : %s\n', s.bstName, r.msg);
    end
    r.seconds = toc(tsub);
    fprintf('  [%s] %-16s trials=%s scouts=%s %6.1f s\n', ...
        upper(r.status), r.subject, num2str(r.n_trials), num2str(r.n_scouts), r.seconds);
    res{k} = r;
end

%% ------------------------------------------------------------- manifest/log
[availGB2, memRaw2] = mem_available_gb();
fprintf('\n-- memory after --\n%s\navailable ~ %.1f GB\n', memRaw2, availGB2);

manifest = fullfile(cfg.log_dir, sprintf('s27_manifest_%s.csv', stamp));
write_manifest(manifest, res, subs(~todo));

nok  = sum(cellfun(@(r) strcmp(r.status, 'ok'),    res));
nbad = sum(cellfun(@(r) strcmp(r.status, 'error'), res));
fprintf('\n===== done in %.1f min =====\n', toc(t_all)/60);
fprintf('processed ok : %d\nfailed       : %d\nskipped      : %d\n', nok, nbad, sum(~todo));
fprintf('manifest     : %s\n', manifest);
fprintf('output       : %s\n', cfg.out_dir);
if nbad > 0
    fprintf('\nFAILURES (rerunning without FORCE retries only these):\n');
    for k = 1:numel(res)
        if strcmp(res{k}.status, 'error')
            fprintf('  %-16s %s\n', res{k}.subject, res{k}.msg);
        end
    end
end
end % ======================================================================


% =========================================================================
% per-subject work
% =========================================================================
function r = process_one(s, cfg, r, lf)
ensure_eeglab(cfg.eeglab_path);
log_line(lf, sprintf('=== %s start | mem_avail %.1f GB', s.bstName, mem_available_gb()));

% ---- 1. Destrieux scouts, selected by name ------------------------------
A = load(s.anatFile, 'Atlas');
if ~isfield(A, 'Atlas') || isempty(A.Atlas)
    error('s27:noatlas', '%s: no Atlas in %s', s.bstName, s.anatFile);
end
allNames = arrayfun(@(a) string(a.Name), A.Atlas(:));
ai = find(strcmpi(allNames, cfg.atlas_name));
if numel(ai) ~= 1
    error('s27:atlasname', '%s: expected exactly 1 atlas named "%s", found %d. Available: %s', ...
        s.bstName, cfg.atlas_name, numel(ai), strjoin(cellstr(allNames)', ', '));
end
atl = A.Atlas(ai);
nsc = numel(atl.Scouts);
if nsc ~= cfg.n_scouts
    error('s27:nscouts', ...
        '%s: atlas "%s" at index %d has %d scouts, expected %d.', ...
        s.bstName, char(atl.Name), ai, nsc, cfg.n_scouts);
end
r.atlas_index = ai;
r.n_scouts    = nsc;
log_line(lf, sprintf('atlas "%s" at index %d, %d scouts (atlases present: %s)', ...
    char(atl.Name), ai, nsc, strjoin(cellstr(allNames)', '|')));

roiindex = cell(nsc, 2);
for rr = 1:nsc
    v = atl.Scouts(rr).Vertices;
    if isempty(v)
        error('s27:emptyscout', '%s: scout %d (%s) has no vertices', ...
            s.bstName, rr, atl.Scouts(rr).Label);
    end
    roiindex{rr,1} = double(v(:))';          % 1 x Ni row vector
    roiindex{rr,2} = char(atl.Scouts(rr).Label);
end

% ---- 2. imaging kernel --------------------------------------------------
K = load(s.kernelFile, 'ImagingKernel', 'GoodChannel');
if ~isfield(K, 'ImagingKernel') || isempty(K.ImagingKernel)
    error('s27:nokernel', '%s: ImagingKernel missing in %s', s.bstName, s.kernelFile);
end
kern = double(K.ImagingKernel);
good = double(K.GoodChannel(:))';
nSrc = size(kern, 1);
nCh  = size(kern, 2);
r.n_sources = nSrc;
r.n_good    = numel(good);
if nCh ~= numel(good)
    error('s27:kernelshape', '%s: ImagingKernel has %d columns but GoodChannel has %d entries', ...
        s.bstName, nCh, numel(good));
end
maxv = max(cellfun(@(v) max(v), roiindex(:,1)));
if maxv > nSrc
    error('s27:vertexrange', '%s: scout vertex %d exceeds the %d kernel rows (anat/kernel mismatch)', ...
        s.bstName, maxv, nSrc);
end
log_line(lf, sprintf('kernel %s : %dx%d, GoodChannel n=%d, max scout vertex %d', ...
    s.kernelFile, nSrc, nCh, numel(good), maxv));

% ---- 3. cleaned overt epochs -------------------------------------------
[fp, fn, fx] = fileparts(s.setFile);
EEG = pop_loadset('filename', [fn fx], 'filepath', fp);
dat = EEG.data;
if ndims(dat) ~= 3
    error('s27:notepoched', '%s: %s is not epoched (size %s)', ...
        s.bstName, s.setFile, mat2str(size(dat)));
end
nTr = size(dat, 3);
if size(dat, 1) < max(good)
    error('s27:channels', '%s: data has %d channels but GoodChannel asks for %d', ...
        s.bstName, size(dat, 1), max(good));
end
if cfg.trial_end > size(dat, 2)
    error('s27:tooshort', '%s: epoch has %d samples, TRIAL_END=%d', ...
        s.bstName, size(dat, 2), cfg.trial_end);
end
labels = trial_labels(EEG, nTr);
n_overt = sum(startsWith(labels, 'O'));
if n_overt == 0
    error('s27:notovert', ['%s: none of the %d trial labels start with "O" (first: "%s"). ' ...
        'This file does not hold overt trials.'], s.bstName, nTr, labels{1});
end
r.n_trials = nTr;
log_line(lf, sprintf('set %s : %d ch x %d samp x %d trials, %d overt-labelled, srate %g', ...
    s.setFile, size(dat, 1), size(dat, 2), nTr, n_overt, EEG.srate));

% ---- 4. project and scout-average --------------------------------------
condition_data      = cell(nTr, 1);
condition_data_type = cell(nTr, 1);
vtx = roiindex(:,1);
for t = 1:nTr
    seg = double(dat(good, cfg.trial_start:cfg.trial_end, t));
    src = kern * seg;                                  % nSrc x nTime
    rm  = zeros(nsc, size(src, 2));
    for rr = 1:nsc
        rm(rr,:) = mean(src(vtx{rr}, :), 1);
    end
    condition_data{t}      = rm;
    condition_data_type{t} = labels{t};
end
dat = []; src = []; EEG = []; seg = []; %#ok<NASGU> release before the -v7.3 write

% ---- 5. atomic save -----------------------------------------------------
ds = struct();
ds.condition_data      = condition_data;
ds.condition_data_type = condition_data_type;
ds.roiindex            = roiindex;
ds.kernel_name         = cfg.kernel_type;
ds.freq_band           = cfg.freq_band;

tmpf = [s.outFile '.part'];
if isfile(tmpf), delete(tmpf); end
save(tmpf, '-fromstruct', ds, '-v7.3');
movefile(tmpf, s.outFile, 'f');

d = dir(s.outFile);
r.bytes = d.bytes;
log_line(lf, sprintf('saved %s (%d bytes)', s.outFile, d.bytes));
end


% =========================================================================
% discovery
% =========================================================================
function subs = discover_subjects(cfg)
subs = struct('bstName', {}, 'sid', {}, 'study', {}, 'kernelFile', {}, ...
              'anatFile', {}, 'setFile', {}, 'outFile', {});
dd = dir(fullfile(cfg.bst_root, 'data', 'Subject*'));
dd = dd([dd.isdir]);

keep = {};
if ~isempty(cfg.subjects)
    keep = strtrim(strsplit(cfg.subjects, ','));
end

for i = 1:numel(dd)
    bstName = dd(i).name;
    if ~isempty(keep) && ~any(strcmpi(bstName, keep)), continue; end

    sdir = fullfile(dd(i).folder, bstName);
    st   = dir(fullfile(sdir, cfg.study_glob));
    st   = st([st.isdir]);

    kernelFile = ''; study = '';
    for j = 1:numel(st)
        kk = dir(fullfile(st(j).folder, st(j).name, ['results_' cfg.kernel_type '*.mat']));
        if ~isempty(kk)
            kernelFile = fullfile(kk(1).folder, kk(1).name);
            study      = st(j).name;
            break
        end
    end
    if isempty(kernelFile)
        fprintf('  skip %-18s : no results_%s*.mat under %s\n', bstName, cfg.kernel_type, cfg.study_glob);
        continue
    end

    tok = regexp(study, '^S(\d{4})', 'tokens', 'once');
    if isempty(tok)
        fprintf('  skip %-18s : cannot parse S#### from study "%s"\n', bstName, study);
        continue
    end
    sid = tok{1};

    anatFile = fullfile(cfg.bst_root, 'anat', bstName, 'tess_cortex_pial_low.mat');
    if ~isfile(anatFile)
        fprintf('  skip %-18s : anat missing %s\n', bstName, anatFile);
        continue
    end

    sg = dir(fullfile(cfg.clean_dir, sprintf(cfg.set_glob, sid)));
    if numel(sg) ~= 1
        fprintf('  skip %-18s : %d matches for %s in %s (need exactly 1)\n', ...
            bstName, numel(sg), sprintf(cfg.set_glob, sid), cfg.clean_dir);
        continue
    end

    subs(end+1) = struct( ...
        'bstName',    bstName, ...
        'sid',        sid, ...
        'study',      study, ...
        'kernelFile', kernelFile, ...
        'anatFile',   anatFile, ...
        'setFile',    fullfile(sg(1).folder, sg(1).name), ...
        'outFile',    fullfile(cfg.out_dir, sprintf('%s_%s_%s.mat', bstName, cfg.kernel_type, cfg.freq_band))); %#ok<AGROW>
end
subs = subs(:);
end


function labels = trial_labels(EEG, nTr)
% One label per epoch. Events and epochs are paired one-to-one when possible,
% otherwise the event closest to time zero within each epoch is taken.
labels = cell(nTr, 1);
if isfield(EEG, 'event') && numel(EEG.event) == nTr
    for t = 1:nTr, labels{t} = EEG.event(t).type; end
elseif isfield(EEG, 'epoch') && numel(EEG.epoch) == nTr
    for t = 1:nTr
        et = EEG.epoch(t).eventtype;
        el = EEG.epoch(t).eventlatency;
        if ~iscell(et), et = {et}; end
        if ~iscell(el), el = {el}; end
        lat = cellfun(@(x) double(x), el);
        [~, k] = min(abs(lat));
        labels{t} = et{k};
    end
else
    nev = 0; nep = 0;
    if isfield(EEG, 'event'), nev = numel(EEG.event); end
    if isfield(EEG, 'epoch'), nep = numel(EEG.epoch); end
    error('s27:labels', 'cannot pair events with epochs: %d trials, %d events, %d epoch entries', ...
        nTr, nev, nep);
end
labels = cellfun(@(x) char(string(x)), labels, 'UniformOutput', false);
end


% =========================================================================
% layout reference
% =========================================================================
function report_reference_layout(refdir)
% Prints the variable layout of one covert source file, which the output of
% this script reproduces field for field. Read-only and optional.
if ~isfolder(refdir), return; end
f = dir(fullfile(refdir, 'Subject*_sLORETA_raw.mat'));
if isempty(f)
    fprintf('\n[ref] no covert reference file in %s - layout check skipped\n', refdir);
    return
end
reffile = fullfile(f(1).folder, f(1).name);
try
    mf = matfile(reffile);
    w  = whos('-file', reffile);
    fprintf('\n[ref] layout of the covert dataset (%s)\n', f(1).name);
    for i = 1:numel(w)
        fprintf('      %-22s %-10s %s\n', w(i).name, mat2str(w(i).size), w(i).class);
    end
    c1 = mf.condition_data(1,1);
    r1 = mf.roiindex(1,1);
    r2 = mf.roiindex(1,2);
    fprintf('      condition_data{1}  : %s %s\n', mat2str(size(c1{1})), class(c1{1}));
    fprintf('      roiindex{1,1}      : %s %s\n', mat2str(size(r1{1})), class(r1{1}));
    fprintf('      roiindex{1,2}      : "%s"\n', r2{1});
catch ME
    fprintf('\n[ref] could not read %s (%s)\n', reffile, ME.message);
end
end


% =========================================================================
% memory and pool
% =========================================================================
function [availGB, raw] = mem_available_gb()
availGB = NaN; raw = '';
try
    if isunix && ~ismac
        [st, out] = system('free -m');
        if st == 0
            raw = strtrim(out);
            tk = regexp(out, 'Mem:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', 'tokens', 'once');
            if numel(tk) == 6
                availGB = str2double(tk{6}) / 1024;      % "available" column
            else
                tk2 = regexp(out, 'Mem:\s+(\d+)\s+(\d+)\s+(\d+)', 'tokens', 'once');
                if numel(tk2) == 3, availGB = str2double(tk2{3}) / 1024; end
            end
        end
    elseif ismac
        [st, out] = system('vm_stat');
        if st == 0
            raw = strtrim(out);
            ps  = regexp(out, 'page size of (\d+) bytes', 'tokens', 'once');
            pf  = regexp(out, 'Pages free:\s+(\d+)',        'tokens', 'once');
            pin = regexp(out, 'Pages inactive:\s+(\d+)',    'tokens', 'once');
            pp  = regexp(out, 'Pages speculative:\s+(\d+)', 'tokens', 'once');
            if ~isempty(ps) && ~isempty(pf)
                pg = str2double(ps{1});
                n  = str2double(pf{1});
                if ~isempty(pin), n = n + str2double(pin{1}); end
                if ~isempty(pp),  n = n + str2double(pp{1});  end
                availGB = n * pg / 2^30;
            end
        end
    end
catch
end
end


function nw = plan_workers(cfg, availGB, n_todo)
%PLAN_WORKERS  Pool size derived from measured free memory.
%   budget  = min(available_GB * MEM_FRACTION, MEM_CAP_GB)
%   workers = floor(budget / MEM_PER_WORKER_GB)
%   then capped by the core count, the number of subjects and MAX_WORKERS.
%   If free memory cannot be read the fallback is 2 workers.
if n_todo <= 1
    nw = 1; return
end
if isnan(availGB) || availGB <= 0
    fprintf('WARNING: could not read free memory - falling back to 2 workers\n');
    nw = 2;
else
    budget = min(availGB * cfg.mem_fraction, cfg.mem_cap_gb);
    nw = floor(budget / cfg.mem_per_worker);
    fprintf('  memory plan: avail %.1f GB x %.2f capped at %.0f GB -> budget %.1f GB / %.1f GB per worker = %d\n', ...
        availGB, cfg.mem_fraction, cfg.mem_cap_gb, budget, cfg.mem_per_worker, nw);
end
nw = max(1, nw);
nw = min([nw, feature('numcores'), n_todo, 24]);
if cfg.max_workers > 0
    nw = min(nw, cfg.max_workers);
end
end


function open_pool(nw)
p = gcp('nocreate');
if ~isempty(p) && p.NumWorkers ~= nw
    delete(p); p = [];
end
if nw > 1 && isempty(p)
    try
        parpool('Processes', nw);
    catch ME
        fprintf('WARNING: parpool(%d) failed (%s) - running serially\n', nw, ME.message);
    end
elseif nw <= 1 && ~isempty(p)
    delete(p);
end
end


% =========================================================================
% small helpers
% =========================================================================
function ensure_eeglab(p)
if exist('pop_loadset', 'file') == 2, return; end
if isempty(p) || ~isfolder(p)
    error('s27:eeglab', 'EEGLAB not found. Set eeglab_dir in config.json (got "%s").', p);
end
addpath(p);
sub = {{'functions','adminfunc'}, {'functions','popfunc'}, {'functions','sigprocfunc'}, ...
       {'functions','guifunc'},   {'functions','miscfunc'}, {'functions','statistics'}};
for k = 1:numel(sub)
    q = fullfile(p, sub{k}{:});
    if isfolder(q), addpath(q); end
end
if exist('pop_loadset', 'file') ~= 2
    error('s27:eeglab', 'EEGLAB found at %s but pop_loadset is still not on the path.', p);
end
end


function v = env_str(name, dflt)
v = getenv(name);
if isempty(v), v = dflt; end
end


function v = env_num(name, dflt)
s = getenv(name);
if isempty(s), v = dflt; return; end
v = str2double(s);
if isnan(v), v = dflt; end
end


function log_line(f, msg)
fid = fopen(f, 'a');
if fid > 0
    fprintf(fid, '%s  %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), msg);
    fclose(fid);
end
end


function write_manifest(fname, res, skipped)
fid = fopen(fname, 'w');
if fid < 0, fprintf('WARNING: cannot write manifest %s\n', fname); return; end
fprintf(fid, ['subject,status,atlas_index,n_scouts,n_trials,n_sources,n_good,bytes,' ...
              'seconds,set_file,kernel_file,anat_file,out_file,msg\n']);
for k = 1:numel(res)
    r = res{k};
    fprintf(fid, '%s,%s,%s,%s,%s,%s,%s,%s,%.1f,%s,%s,%s,%s,"%s"\n', ...
        r.subject, r.status, num2str(r.atlas_index), num2str(r.n_scouts), ...
        num2str(r.n_trials), num2str(r.n_sources), num2str(r.n_good), ...
        num2str(r.bytes), r.seconds, r.set_file, r.kernel_file, r.anat_file, ...
        r.out_file, strrep(r.msg, '"', ''''));
end
for k = 1:numel(skipped)
    s = skipped(k);
    fprintf(fid, '%s,skipped_exists,,,,,,,0.0,%s,%s,%s,%s,""\n', ...
        s.bstName, s.setFile, s.kernelFile, s.anatFile, s.outFile);
end
fclose(fid);
end
