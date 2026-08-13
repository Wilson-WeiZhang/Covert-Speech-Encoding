#!/usr/bin/env python3
"""Per-participant acoustic envelopes for overt and covert trials.

The microphone records continuously for the whole session, so overt and covert
blocks are captured in a single wav file with the same microphone, gain and
placement. Cutting both conditions out of that one recording makes the overt
trials a sensitivity positive control for the covert trials: the two differ
only in task instruction.

TRIAL GEOMETRY
    Trials are cut 14 s from the trial-onset beep. Within that window the
    phrase appears on screen at 2.0 s and the five utterances follow over the
    next ~10 s. All analysis windows below are expressed relative to PHRASE
    ONSET, which is the event the EEG epochs are also aligned to.

WINDOWS
    baseline   -0.5 to 0.0 s relative to phrase onset. Immediately
               pre-stimulus, and late enough to exclude the decay of the
               trial-onset beep.
    utterance   0.0 to 10.0 s relative to phrase onset, covering all five
               utterances.

MARKER PARSING
    The marker file repeats the same S-marker on consecutive lines, so
    consecutive duplicates are dropped. Markers that are not phrase cues take
    part in that de-duplication chain, so they must be parsed and only then
    filtered out; discarding them first yields the wrong marker count.
    Each participant has 200 phrase markers. Condition follows the block index,
    blocks alternating overt and covert from the first block onwards.

OUT : <out>/11_acoustic_monitoring/acoustic_envelopes.npz
        env_overt, env_covert     (n_subj, n_bins) mean RMS envelope, 100 ms bins
        base_overt, base_covert   (n_subj,) mean RMS over the baseline window
        utt_overt, utt_covert     (n_subj,) mean RMS over the utterance window
        subjects, bin_s, phrase_onset_s
"""
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.set_paths import set_paths

MARKER_SFREQ = 1000.0        # marker file position units per second
TRIAL_S = 14.0               # trial window, from the trial-onset beep
BIN_S = 0.1                  # envelope bin width
PHRASE_ONSET_S = 2.0         # phrase appears on screen this long after the beep
BASELINE_S = (-0.5, 0.0)     # relative to phrase onset
UTTERANCE_S = (0.0, 10.0)    # relative to phrase onset
N_PHRASE_MARKERS = 200


def audio_root(cfg):
    root = Path(cfg.audio_raw)
    if not (root / "Audio_PP").is_dir():
        raise SystemExit(
            f"'audio_raw_dir' does not point at the continuous recordings: {root}\n"
            "It must contain Audio_PP/<subject>.wav and EEG/<subject>_Filters.vmrk."
        )
    return root


def find_subjects(root):
    """Participants that have both a recording and its marker file."""
    subjects = []
    for wav in sorted((root / "Audio_PP").glob("S*.wav")):
        if (root / "EEG" / f"{wav.stem}_Filters.vmrk").is_file():
            subjects.append(wav.stem)
    return subjects


def load_markers(vmrk):
    """Return [(is_overt, onset_seconds)] for the phrase-cue markers."""
    kept, last = [], None
    with open(vmrk, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("Mk") or line.startswith("Mk1="):
                continue
            f = line.split(",")
            if len(f) < 3:
                continue
            desc, pos = f[1], f[2]
            if desc[:1] == "S":
                if desc == last:      # drop consecutive repeats of the same marker
                    continue
                last = desc
            try:
                kept.append((desc, int(pos) / MARKER_SFREQ))
            except ValueError:
                pass
    phrase = [(d, o) for d, o in kept if d.startswith("S  ")]
    # 20 trials per block; blocks alternate, starting overt.
    return [(i // 20 in (0, 2, 4, 6, 8), o) for i, (d, o) in enumerate(phrase)]


def main():
    cfg = set_paths()
    root = audio_root(cfg)
    outdir = cfg.out / "11_acoustic_monitoring"
    outdir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(root)
    print(f"found {len(subjects)} participants with audio and markers")

    E = {"overt": [], "covert": []}
    W = {"overt": [], "covert": []}
    used = []
    for sid in subjects:
        rows = load_markers(root / "EEG" / f"{sid}_Filters.vmrk")
        if len(rows) != N_PHRASE_MARKERS:
            print(f"skip {sid}: {len(rows)} phrase markers "
                  f"(expected {N_PHRASE_MARKERS})")
            continue

        sr, data = wavfile.read(root / "Audio_PP" / f"{sid}.wav", mmap=True)
        if data.ndim > 1:
            data = data[:, 0]

        t0 = rows[0][1]                       # first phrase onset anchors the session
        width, need = int(sr * BIN_S), int(sr * TRIAL_S)
        nbin = need // width

        acc = {"overt": [], "covert": []}
        for is_overt, onset in rows:
            s = int(sr * (onset - t0))
            seg = data[s:s + need]
            if seg.shape[0] < need:
                continue
            x = seg[:nbin * width].astype(np.float64).reshape(nbin, width)
            acc["overt" if is_overt else "covert"].append(np.sqrt((x * x).mean(axis=1)))

        if not acc["overt"] or not acc["covert"]:
            print(f"skip {sid}: no usable trials in one condition")
            continue

        b0 = int((PHRASE_ONSET_S + BASELINE_S[0]) / BIN_S)
        b1 = int((PHRASE_ONSET_S + BASELINE_S[1]) / BIN_S)
        u0 = int((PHRASE_ONSET_S + UTTERANCE_S[0]) / BIN_S)
        u1 = int((PHRASE_ONSET_S + UTTERANCE_S[1]) / BIN_S)
        for c in ("overt", "covert"):
            m = np.array(acc[c]).mean(axis=0)          # bins, averaged over trials
            E[c].append(m)
            W[c].append((m[b0:b1].mean(), m[u0:u1].mean()))
        used.append(sid)
        print(f"{sid} ok  overt {len(acc['overt'])}  covert {len(acc['covert'])}", flush=True)

    out = outdir / "acoustic_envelopes.npz"
    np.savez(out,
             subjects=np.array(used),
             env_overt=np.array(E["overt"]), env_covert=np.array(E["covert"]),
             base_overt=np.array([a for a, _ in W["overt"]]),
             utt_overt=np.array([b for _, b in W["overt"]]),
             base_covert=np.array([a for a, _ in W["covert"]]),
             utt_covert=np.array([b for _, b in W["covert"]]),
             bin_s=BIN_S, phrase_onset_s=PHRASE_ONSET_S)
    print(f"wrote {out} ({len(used)} participants)")


if __name__ == "__main__":
    main()
