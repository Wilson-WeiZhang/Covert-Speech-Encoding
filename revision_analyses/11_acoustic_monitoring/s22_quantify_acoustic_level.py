#!/usr/bin/env python3
"""Trial-by-trial screen for audible vocalisation during covert trials.

Where s21 averages trials into a per-participant envelope, this script keeps
every trial, so that individual covert trials can be compared against the
participant's own overt distribution. The question it answers is not "is the
covert mean elevated" but "did any single covert trial reach the acoustic level
of overt speech".

Overt trials from the SAME continuous recording are the sensitivity positive
control: same microphone, gain, session and window convention, differing only
in task instruction.

TRIAL GEOMETRY
    Trials are cut 14 s from the trial-onset beep; the phrase appears on screen
    at 2.0 s and the five utterances follow over the next ~10 s. Windows are
    expressed relative to PHRASE ONSET:
        baseline   -0.5 to 0.0 s   immediately pre-stimulus, past the beep decay
        utterance   0.0 to 10.0 s  covering all five utterances

MARKER PARSING
    The marker file repeats the same S-marker on consecutive lines, so
    consecutive duplicates are dropped. Markers that are not phrase cues take
    part in that de-duplication chain, so they must be parsed and only then
    filtered out; discarding them first yields the wrong marker count.
    Each participant has 200 phrase markers. Condition follows the block index,
    blocks alternating overt and covert from the first block onwards.

Usage:  python s22_quantify_acoustic_level.py [n_participants]

OUT : <out>/11_acoustic_monitoring/acoustic_level_summary.json
        per participant: trial counts, median baseline and utterance RMS,
        utterance/baseline ratios, and the number of covert trials reaching the
        5th, 25th and 50th percentile of that participant's overt trials
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.set_paths import set_paths

MARKER_SFREQ = 1000.0        # marker file position units per second
TRIAL_S = 14.0               # trial window, from the trial-onset beep
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
    """Return [(phrase_id, condition, onset_seconds)] for the phrase-cue markers."""
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
                continue
    phrase = [(d, o) for d, o in kept if d.startswith("S  ")]
    rows = []
    for i, (desc, onset) in enumerate(phrase):
        # 20 trials per block; blocks alternate, starting overt.
        cond = "overt" if (i // 20) in (0, 2, 4, 6, 8) else "covert"
        rows.append((int(desc[-1]) - 1, cond, onset))
    return rows


def rms(x):
    x = x.astype(np.float64)
    return float(np.sqrt((x * x).mean())) if x.size else float("nan")


def do_subject(root, sid):
    rows = load_markers(root / "EEG" / f"{sid}_Filters.vmrk")
    if len(rows) != N_PHRASE_MARKERS:
        return {"subject": sid,
                "error": f"{len(rows)} phrase markers (expected {N_PHRASE_MARKERS})"}

    sr, data = wavfile.read(root / "Audio_PP" / f"{sid}.wav", mmap=True)
    if data.ndim > 1:
        data = data[:, 0]

    t0 = rows[0][2]                       # first phrase onset anchors the session
    nb = (int(sr * (PHRASE_ONSET_S + BASELINE_S[0])),
          int(sr * (PHRASE_ONSET_S + BASELINE_S[1])))
    nu = (int(sr * (PHRASE_ONSET_S + UTTERANCE_S[0])),
          int(sr * (PHRASE_ONSET_S + UTTERANCE_S[1])))
    need = int(sr * TRIAL_S)

    rec = {"overt": [], "covert": []}
    for lab, cond, onset in rows:
        s = int(sr * (onset - t0))
        seg = data[s:s + need]
        if seg.shape[0] < nu[1]:
            continue
        rec[cond].append((rms(seg[nb[0]:nb[1]]), rms(seg[nu[0]:nu[1]]), lab))

    res = {"subject": sid, "sr": int(sr), "dur_min": round(data.shape[0] / sr / 60, 1)}
    for c in ("overt", "covert"):
        a = np.array([[b, u] for b, u, _ in rec[c]], dtype=float)
        ratio = a[:, 1] / np.maximum(a[:, 0], 1e-9)
        res[c] = {
            "n": len(a),
            "base_med": round(float(np.median(a[:, 0])), 2),
            "utt_med": round(float(np.median(a[:, 1])), 2),
            "ratio_med": round(float(np.median(ratio)), 3),
            "ratio_p95": round(float(np.percentile(ratio, 95)), 3),
            "utt_max": round(float(a[:, 1].max()), 2),
        }

    # How many covert trials reach the participant's own overt speech range?
    co = np.array([u for _, u, _ in rec["covert"]], dtype=float)
    ov = np.array([u for _, u, _ in rec["overt"]], dtype=float)
    for q in (5, 25, 50):
        res[f"covert_ge_overt_p{q}"] = int((co >= np.percentile(ov, q)).sum())
    res["covert_n"] = int(co.size)
    res["covert_over_overt_med"] = round(float(np.median(co) / np.median(ov)), 4)
    return res


def main():
    cfg = set_paths()
    root = audio_root(cfg)
    outdir = cfg.out / "11_acoustic_monitoring"
    outdir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(root)
    if len(sys.argv) > 1:
        subjects = subjects[:int(sys.argv[1])]

    allres = []
    for sid in subjects:
        try:
            r = do_subject(root, sid)
        except Exception as e:
            r = {"subject": sid, "error": f"{type(e).__name__}: {e}"}
        allres.append(r)
        print(json.dumps(r), flush=True)

    out = outdir / "acoustic_level_summary.json"
    with open(out, "w") as fh:
        json.dump(allres, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
