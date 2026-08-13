%% ========================================================================
%  s24_epoch_overt_blocks.m -- Stage 1 of the overt arm
%
%  Each recording session alternated ten blocks, odd = overt speech and
%  even = covert speech, with the EEG running continuously throughout. The
%  main analysis epoched the covert (even) blocks only. This script builds
%  the matching overt half from the same raw BrainVision recordings, using
%  the same import, resampling, filtering and epoching operations as the
%  covert Stage 1. The two arms therefore differ in condition and in
%  nothing else.
%
%  Relative to the covert Stage 1, only the following are different:
%    - the epoched event types are {'O 1'..'O 5'} instead of {'C 1'..'C 5'}
%    - the epoched blocks are [1 3 5 7 9] instead of [2 4 6 8 10]
%    - trials per block are recorded rather than asserted to be 20; overt
%      recordings may legitimately contain fewer
%    - block membership is tested with endsWith('_b_<n>') rather than
%      contains(), because with the odd-block list '_b_1' is a prefix of
%      '_b_10' and a substring test would merge the two blocks
%    - the script runs unattended: any subject that fails a consistency
%      check is logged and skipped instead of interrupting the run
%
%  PROCESSING
%    1. import the BrainVision triplet, resample to 250 Hz,
%       1-100 Hz bandpass, 49-51 Hz notch
%    2. keep the five stimulus markers 'S  1'..'S  5'
%    3. cut the recording into blocks at latency gaps longer than 20 s and
%       require exactly ten blocks
%    4. drop any marker that follows its predecessor by less than 5 s
%    5. label markers by block parity: odd -> 'O', even -> 'C'
%    6. expand each overt marker into '<type>_u_1_b_<block>' and epoch
%       -0.5 to +1.5 s around it
%
%  Only the first utterance of each trial (u_1) is expanded, matching the
%  covert arm. The counts that would follow from expanding u_2..u_5 as
%  synthetic events at +2/+4/+6/+8 s are written to the report instead, so
%  the number of markers involved is on record.
%
%  ENVIRONMENT (optional)
%    SUBJ_LIST   indices into the discovered header list, e.g. "1" or "1:5"
%
%  INPUT   cfg.raw_overt/*.vhdr (+ .eeg/.vmrk)
%  OUTPUT  <cfg.out>/overt_epochs/<SID>_overt_processed_trials.set (+ .fdt)
%          <cfg.out>/overt_epochs/s24_epoch_report.csv   one row per subject
%          <cfg.out>/overt_epochs/s24_epoch_log.txt      full diary
%
%  Report columns:
%    subject, n_blocks_found, n_trials_kept, n_u1_events,
%    n_dropped_lt5s_all, n_dropped_lt5s_odd, n_u2to5_potential,
%    trials_b1, trials_b3, trials_b5, trials_b7, trials_b9, status, note
%% ========================================================================

clearvars
clc

addpath(fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'config'));
cfg = set_paths();

data_dir   = cfg.raw_overt;
output_dir = fullfile(cfg.out, 'overt_epochs');
if isempty(data_dir)
    error('s24:noRawDir', ...
        'raw_overt_dir is not set in config.json; it must point at the raw BrainVision recordings.');
end
if ~isfolder(output_dir), mkdir(output_dir); end

subj_list_str = getenv('SUBJ_LIST');

diary_file = fullfile(output_dir, 's24_epoch_log.txt');
diary(diary_file);
diary on

fprintf('=== s24_epoch_overt_blocks ===\n');
fprintf('started    : %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
fprintf('raw dir    : %s\n', data_dir);
fprintf('output dir : %s\n', output_dir);

%% ---------------- EEGLAB ----------------
if isfolder(cfg.eeglab), addpath(cfg.eeglab); end
if exist('eeglab', 'file') ~= 2
    error('s24:noEEGLAB', 'EEGLAB not on the path (eeglab_dir = "%s").', cfg.eeglab);
end
eeglab('nogui');
if exist('pop_loadbv', 'file') ~= 2
    error('s24:noBVA', 'pop_loadbv not found -- the bva-io plugin is missing from %s.', cfg.eeglab);
end

%% ---------------- parameters ----------------
file_list         = dir(fullfile(data_dir, '*.vhdr'));
event_types       = {'O 1', 'O 2', 'O 3', 'O 4', 'O 5'};
pre_event_time    = 0.5;
post_event_time   = 1.5;
valid_event_types = {'S  1', 'S  2', 'S  3', 'S  4', 'S  5'};
overt_blocks      = [1 3 5 7 9];

if isempty(file_list)
    diary off
    error('s24:noData', 'No *.vhdr found in %s', data_dir);
end

fprintf('found %d .vhdr file(s)\n', numel(file_list));

% The recording set holds 59 header files, of which the published pipeline
% processes 58: the second entry of the alphabetically sorted list is
% excluded. The same exclusion is applied here so that subject indices are
% identical to the covert arm.
if numel(file_list) == 59
    fprintf('excluding header %s, leaving 58 recordings\n', file_list(2).name);
    file_list(2) = [];
else
    fprintf(['WARNING: expected 59 .vhdr files, found %d. No entry excluded; ' ...
             'subject indexing may differ from the covert arm.\n'], numel(file_list));
end

n_files = numel(file_list);
if isempty(subj_list_str)
    subj_list = 1:min(58, n_files);
else
    subj_list = str2num(subj_list_str);  %#ok<ST2NM>  accepts "1", "1:5", "1,7,9"
    subj_list = subj_list(subj_list >= 1 & subj_list <= n_files);
    subj_list = reshape(subj_list, 1, []);
    fprintf('SUBJ_LIST override: %s\n', mat2str(subj_list));
end
fprintf('processing %d of %d recording(s)\n\n', numel(subj_list), n_files);

%% ---------------- report file, written incrementally ----------------
csv_file = fullfile(output_dir, 's24_epoch_report.csv');
fid_csv = fopen(csv_file, 'w');
if fid_csv < 0
    diary off
    error('s24:noCSV', 'Cannot open %s for writing.', csv_file);
end
fprintf(fid_csv, ['subject,n_blocks_found,n_trials_kept,n_u1_events,' ...
                  'n_dropped_lt5s_all,n_dropped_lt5s_odd,n_u2to5_potential,' ...
                  'trials_b1,trials_b3,trials_b5,trials_b7,trials_b9,status,note\n']);

n_sub_total  = numel(subj_list);
rep_subject  = cell(n_sub_total, 1);
rep_status   = cell(n_sub_total, 1);
rep_note     = cell(n_sub_total, 1);
rep_trials   = nan(n_sub_total, 1);
rep_perblock = nan(n_sub_total, numel(overt_blocks));
rep_i        = 0;

%% ======================= main loop =====================================
for subj = subj_list
    rep_i = rep_i + 1;

    sid                = file_list(subj).name(1:end-5);   % strip '.vhdr'
    rep_subject{rep_i} = sid;
    rep_status{rep_i}  = 'skip_error';
    rep_note{rep_i}    = 'unknown';

    n_blocks_found = NaN;
    n_u1_events    = NaN;
    n_dropped_all  = NaN;
    n_dropped_odd  = NaN;
    n_trials_kept  = NaN;
    trial_counts   = nan(1, numel(overt_blocks));

    fprintf('\n----- [%d/%d] %s -----\n', rep_i, n_sub_total, sid);

    try
        %% ---- import, resample, filter ----
        EEG = pop_loadbv(data_dir, file_list(subj).name);
        EEG = pop_resample(EEG, 250);
        EEG = pop_eegfiltnew(EEG, 1, 100, [], 0, [], 0);
        EEG = pop_eegfiltnew(EEG, 49, 51, [], 1, [], 0);

        valid_indices = find(ismember({EEG.event.type}, valid_event_types));
        EEG.event = EEG.event(valid_indices);

        if isempty(EEG.event)
            rep_status{rep_i} = 'skip_noevents';
            rep_note{rep_i}   = 'no S 1..S 5 stimulus markers found';
            fprintf('  SKIP: no valid stimulus markers.\n');
            error('s24:handled', 'handled');   % jump to the report writer
        end

        %% ---- block boundaries from latency gaps longer than 20 s ----
        latencies     = [EEG.event.latency];
        latency_diffs = diff(latencies) / EEG.srate;
        block_breaks  = find(latency_diffs > 20);

        block_numbers = ones(1, numel(EEG.event));
        current_block = 1;
        for i = 1:numel(block_breaks)
            block_numbers(block_breaks(i)+1:end) = current_block + 1;
            current_block = current_block + 1;
        end

        n_blocks_found = max(block_numbers);
        fprintf('  markers=%d  blocks detected=%d\n', numel(EEG.event), n_blocks_found);

        if n_blocks_found ~= 10
            fprintf('  SKIP: %s expected 10 blocks, found %d\n', sid, n_blocks_found);
            rep_status{rep_i} = 'skip_blocks';
            rep_note{rep_i}   = sprintf('expected 10 blocks found %d', n_blocks_found);
            error('s24:handled', 'handled');
        end

        %% ---- attach the block number to every event ----
        event_cell  = struct2cell(EEG.event);
        field_names = fieldnames(EEG.event);
        event_cell(end+1, :) = num2cell(block_numbers); %#ok<SAGROW>
        field_names{end+1}   = 'block_number'; %#ok<SAGROW>
        EEG.event = cell2struct(event_cell, field_names, 1);

        %% ---- drop markers less than 5 s after the previous one ----
        to_remove = [];
        for j = 2:numel(EEG.event)
            if (EEG.event(j).latency - EEG.event(j-1).latency) < 5 * EEG.srate
                to_remove = [to_remove, j]; %#ok<AGROW>
            end
        end
        n_dropped_all = numel(to_remove);
        if n_dropped_all > 0
            dropped_blocks = [EEG.event(to_remove).block_number];
            n_dropped_odd  = sum(mod(dropped_blocks, 2) == 1);
        else
            n_dropped_odd = 0;
        end
        fprintf('  markers dropped by the 5 s rule: %d total, %d in overt blocks\n', ...
                n_dropped_all, n_dropped_odd);
        EEG.event(to_remove) = [];

        if ~isempty(EEG.event) && isequal(EEG.event(1).type, 'boundary')
            EEG.event(1) = [];
        end

        %% ---- label by block parity: odd = overt, even = covert ----
        for j = 1:numel(EEG.event)
            current_block = EEG.event(j).block_number;
            if mod(current_block, 2) == 1
                EEG.event(j).type = ['O' EEG.event(j).type(3:end)];
            else
                EEG.event(j).type = ['C' EEG.event(j).type(3:end)];
            end
            EEG.event(j).block_number = current_block;
        end

        %% ---- the block sequence must alternate O C O C ... ----
        pattern_check = true;
        for blk = 1:10
            block_events = find([EEG.event.block_number] == blk);
            if ~isempty(block_events)
                if mod(blk, 2) == 1 && ~startsWith(EEG.event(block_events(1)).type, 'O')
                    pattern_check = false;
                    break;
                elseif mod(blk, 2) == 0 && ~startsWith(EEG.event(block_events(1)).type, 'C')
                    pattern_check = false;
                    break;
                end
            end
        end

        if ~pattern_check
            fprintf('  SKIP: %s block sequence is not O C O C alternating\n', sid);
            rep_status{rep_i} = 'skip_pattern';
            rep_note{rep_i}   = 'block sequence not O C O C alternating';
            error('s24:handled', 'handled');
        end

        %% ---- expand overt markers into '<type>_u_1_b_<block>' ----
        expanded_events = struct('type', {}, 'latency', {}, 'urevent', {}, 'block_number', {});
        n_u1_events = 0;
        for j = 1:numel(EEG.event)
            current_block = EEG.event(j).block_number;
            if ismember(EEG.event(j).type, event_types)
                for k = 1:1                                  % first utterance only
                    new_event = struct();
                    new_event.type = [EEG.event(j).type '_u_' num2str(k), '_b_', num2str(current_block)];
                    new_event.latency = EEG.event(j).latency + (k-1) * 2 * EEG.srate;
                    new_event.urevent = EEG.event(j).urevent;
                    new_event.block_number = current_block;
                    expanded_events(end+1) = new_event; %#ok<SAGROW>
                    n_u1_events = n_u1_events + 1;
                end
            else
                new_event = struct();
                new_event.type = EEG.event(j).type;
                new_event.latency = EEG.event(j).latency;
                new_event.urevent = EEG.event(j).urevent;
                new_event.block_number = current_block;
                expanded_events(end+1) = new_event; %#ok<SAGROW>
            end
        end
        EEG.event = expanded_events;
        EEG = eeg_checkset(EEG, 'eventconsistency');

        fprintf('  overt u_1 events: %d   (u_2..u_5 not expanded; potential = %d)\n', ...
                n_u1_events, 4 * n_u1_events);

        if n_u1_events == 0
            rep_status{rep_i} = 'skip_noevents';
            rep_note{rep_i}   = 'no overt events after expansion';
            fprintf('  SKIP: no overt events to epoch.\n');
            error('s24:handled', 'handled');
        end

        %% ---- epoch every overt phrase in every overt block ----
        all_event_types = {};
        for j = 1:numel(event_types)
            for k = 1:1
                for blk = overt_blocks
                    all_event_types{end+1} = [event_types{j} '_u_' num2str(k) '_b_' num2str(blk)]; %#ok<SAGROW>
                end
            end
        end

        EEG = pop_epoch(EEG, all_event_types, [-pre_event_time post_event_time], ...
            'newname', [sid '_overt_epochs'], 'epochinfo', 'yes');

        n_trials_kept = EEG.trials;

        %% ---- trials per overt block ----
        % endsWith, not contains: '_b_1' is a prefix of '_b_10'.
        trial_counts = zeros(1, numel(overt_blocks));
        for bi = 1:numel(overt_blocks)
            tag = ['_b_' num2str(overt_blocks(bi))];
            count = 0;
            for i = 1:EEG.trials
                et = EEG.epoch(i).eventtype;
                if ~iscell(et), et = {et}; end
                et = et(cellfun(@ischar, et));
                if ~isempty(et) && any(endsWith(et, tag))
                    count = count + 1;
                end
            end
            trial_counts(bi) = count;
        end

        fprintf('  trials kept = %d   per overt block %s = %s\n', ...
                n_trials_kept, mat2str(overt_blocks), mat2str(trial_counts));

        note = '';
        if ~all(trial_counts == 20)
            fprintf('  NOTE: %s does not have 20 trials in every overt block: %s\n', ...
                    sid, mat2str(trial_counts));
            note = sprintf('per-block counts %s (not all 20)', mat2str(trial_counts));
        end

        %% ---- save ----
        output_filename = [sid '_overt_processed_trials.set'];
        EEG = pop_saveset(EEG, 'filename', output_filename, 'filepath', output_dir);

        rep_status{rep_i} = 'ok';
        rep_note{rep_i}   = note;
        fprintf('  saved: %s\n', fullfile(output_dir, output_filename));

    catch ME
        if ~strcmp(ME.identifier, 's24:handled')
            rep_status{rep_i} = 'skip_error';
            rep_note{rep_i}   = sprintf('%s | %s', ME.identifier, ME.message);
            fprintf('  ERROR (subject skipped): %s\n', ME.message);
            if ~isempty(ME.stack)
                fprintf('    at %s line %d\n', ME.stack(1).name, ME.stack(1).line);
            end
        end
    end

    %% ---- one report row per subject, flushed immediately ----
    rep_trials(rep_i)      = n_trials_kept;
    rep_perblock(rep_i, :) = trial_counts;

    note_csv = strrep(rep_note{rep_i}, '"', '''');
    fprintf(fid_csv, '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,"%s"\n', ...
        sid, num2str(n_blocks_found), num2str(n_trials_kept), num2str(n_u1_events), ...
        num2str(n_dropped_all), num2str(n_dropped_odd), ...
        num2str(4 * n_u1_events), ...
        num2str(trial_counts(1)), num2str(trial_counts(2)), num2str(trial_counts(3)), ...
        num2str(trial_counts(4)), num2str(trial_counts(5)), ...
        rep_status{rep_i}, note_csv);
end

fclose(fid_csv);

%% ======================= summary =======================================
ok_mask = strcmp(rep_status, 'ok');
n_ok    = sum(ok_mask);
n_skip  = numel(rep_status) - n_ok;

fprintf('\n================ SUMMARY ================\n');
fprintf('subjects attempted : %d\n', n_sub_total);
fprintf('succeeded          : %d\n', n_ok);
fprintf('skipped / failed   : %d\n', n_skip);

if n_skip > 0
    fprintf('\n--- failures ---\n');
    for i = 1:numel(rep_status)
        if ~ok_mask(i)
            fprintf('  %-24s %-16s %s\n', rep_subject{i}, rep_status{i}, rep_note{i});
        end
    end
end

if n_ok > 0
    kept = rep_trials(ok_mask);
    fprintf('\n--- trial counts (successful subjects) ---\n');
    fprintf('  min %d  median %g  max %d  mean %.2f  total %d\n', ...
            min(kept), median(kept), max(kept), mean(kept), sum(kept));
    uk = unique(kept);
    for i = 1:numel(uk)
        fprintf('  %4d trials : %d subject(s)\n', uk(i), sum(kept == uk(i)));
    end

    pb = rep_perblock(ok_mask, :);
    fprintf('\n--- per-block trial counts (successful subjects) ---\n');
    for bi = 1:numel(overt_blocks)
        fprintf('  block %2d : min %d  median %g  max %d\n', ...
                overt_blocks(bi), min(pb(:,bi)), median(pb(:,bi)), max(pb(:,bi)));
    end
end

fprintf('\nreport   : %s\n', csv_file);
fprintf('log      : %s\n', diary_file);
fprintf('finished : %s\n', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));

diary off
