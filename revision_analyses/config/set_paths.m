function cfg = set_paths()
%SET_PATHS  Resolve every data and output location used in this repository.
%
%   cfg = SET_PATHS() reads config/config.json (copy config.example.json and
%   edit it) and returns a struct with absolute paths. All scripts obtain
%   their locations from this function; none contain hard-coded paths.
%
%   Fields returned:
%     cfg.data_root      root of the data distribution
%     cfg.eeg            preprocessed EEG (.set/.fdt)
%     cfg.raw_overt      raw BrainVision recordings (.vhdr/.eeg/.vmrk) from
%                        which the overt blocks are epoched
%                        (optional; used only by 10_overt_arm/preprocessing)
%     cfg.source         source-localised covert data (Destrieux, 148 ROIs)
%     cfg.source_overt   source-localised overt data
%     cfg.audio          Hilbert envelopes of the audio recordings
%     cfg.audio_raw      continuous microphone recordings and event markers
%                        (optional; used only by 11_acoustic_monitoring)
%     cfg.bst_protocol   Brainstorm protocol root (contains anat/ and data/)
%     cfg.out            output root for this repository
%     cfg.eeglab         EEGLAB installation
%     cfg.brainstorm     Brainstorm installation
%     cfg.n_workers      parallel workers to request
%
%   cfg.addback(variant) returns the directory of a component add-back
%   dataset, e.g. cfg.addback('lateye').

here = fileparts(mfilename('fullpath'));
cfgFile = fullfile(here, 'config.json');
if ~isfile(cfgFile)
    error('set_paths:noConfig', ...
        'config.json not found. Copy config.example.json to config.json and edit it.');
end

raw = jsondecode(fileread(cfgFile));

root = raw.data_root;
if ~isfolder(root)
    error('set_paths:badRoot', 'data_root does not exist: %s', root);
end

cfg = struct();
cfg.data_root    = root;
cfg.eeg          = fullfile(root, raw.eeg_dir);
if isfield(raw, 'raw_overt_dir'), cfg.raw_overt = fullfile(root, raw.raw_overt_dir); else, cfg.raw_overt = ''; end
cfg.source       = fullfile(root, raw.source_dir);
cfg.source_overt = fullfile(root, raw.source_overt_dir);
cfg.audio        = fullfile(root, raw.audio_dir);
if isfield(raw, 'audio_raw_dir'), cfg.audio_raw = raw.audio_raw_dir; else, cfg.audio_raw = ''; end
cfg.bst_protocol = raw.brainstorm_protocol;
cfg.eeglab       = raw.eeglab_dir;
cfg.brainstorm   = raw.brainstorm_dir;
cfg.n_workers    = raw.n_workers;

outRoot = raw.output_root;
if ~isAbsolutePath(outRoot)
    outRoot = fullfile(fileparts(here), outRoot);
end
cfg.out = outRoot;
if ~isfolder(cfg.out), mkdir(cfg.out); end

pattern = raw.addback_dir_pattern;
cfg.addback = @(variant) fullfile(root, sprintf(pattern, variant));

end


function tf = isAbsolutePath(p)
if ispc
    tf = numel(p) > 1 && (p(2) == ':' || startsWith(p, '\\'));
else
    tf = startsWith(p, '/');
end
end
