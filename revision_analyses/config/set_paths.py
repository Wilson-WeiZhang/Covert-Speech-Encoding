"""Resolve every data and output location used in this repository.

Copy ``config.example.json`` to ``config.json``, edit it, then::

    from config.set_paths import set_paths
    cfg = set_paths()
    env = cfg.audio / "S0009_envelope.mat"

All scripts obtain their locations from :func:`set_paths`; none contain
hard-coded paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    data_root: Path
    eeg: Path
    raw_overt: Path
    source: Path
    source_overt: Path
    audio: Path
    audio_raw: Path
    bst_protocol: Path
    out: Path
    eeglab: Path
    brainstorm: Path
    n_workers: int
    _addback_pattern: str

    def addback(self, variant: str) -> Path:
        """Directory of a component add-back dataset, e.g. ``addback('lateye')``."""
        return self.data_root / (self._addback_pattern % variant)


def set_paths(config_file: Path | None = None) -> Config:
    cfg_path = Path(config_file) if config_file else _HERE / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            "config.json not found. Copy config.example.json to config.json and edit it."
        )

    raw = json.loads(cfg_path.read_text())
    root = Path(raw["data_root"]).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    out = Path(raw["output_root"]).expanduser()
    if not out.is_absolute():
        out = _HERE.parent / out
    out.mkdir(parents=True, exist_ok=True)

    return Config(
        data_root=root,
        eeg=root / raw["eeg_dir"],
        raw_overt=root / raw.get("raw_overt_dir", ""),
        source=root / raw["source_dir"],
        source_overt=root / raw["source_overt_dir"],
        audio=root / raw["audio_dir"],
        audio_raw=Path(raw.get("audio_raw_dir", "")).expanduser(),
        bst_protocol=Path(raw["brainstorm_protocol"]).expanduser(),
        out=out,
        eeglab=Path(raw["eeglab_dir"]).expanduser(),
        brainstorm=Path(raw["brainstorm_dir"]).expanduser(),
        n_workers=int(raw["n_workers"]),
        _addback_pattern=raw["addback_dir_pattern"],
    )
