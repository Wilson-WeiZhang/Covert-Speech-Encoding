#!/usr/bin/env python3
"""Acoustic level in each analysis window, for both instructions.

s23 draws the figure and reports the single 0-2 s window that spans the EEG
epoch. This script reports the same quantity in each of the windows the
manuscript uses, so every acoustic value quoted in the text and in the
Supplementary Note can be recomputed from the released pipeline.

INPUT
    <out>/11_acoustic_monitoring/acoustic_envelopes.npz, written by s21.

MEASURE
    Per participant and condition, the mean RMS envelope over the window is
    divided by that participant's own mean RMS over the baseline window and
    converted once to decibels:

        20 * log10( mean(envelope over window) / mean(envelope over baseline) )

    Averaging the envelope before the logarithm, rather than averaging
    per-bin decibel values, keeps the scalar on the same footing as the
    energy ratio it describes; the two differ because the logarithm is
    concave.

BASELINE
    -0.5 to 0.0 s relative to phrase onset, computed separately for each
    condition, so a participant is always compared with their own silence.

WINDOWS
    0-600 ms      speech planning period
    200-400 ms    the window in which phrase discrimination peaks
    600-1200 ms   execution period
    0-2 s         the full EEG epoch (the window s23 plots)

TEST
    One-sided one-sample t-test against the participant's own baseline. The
    direction is fixed in advance by physics: audible production can only add
    acoustic energy, so only an elevation is of interest. The 95% confidence
    interval reported beside it is two-sided.

OUT
    <out>/11_acoustic_monitoring/acoustic_window_levels.csv   per participant
    <out>/11_acoustic_monitoring/acoustic_window_stats.csv    per window

Usage:  python s24_acoustic_windows.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.set_paths import set_paths

BASELINE_S = (-0.5, 0.0)          # relative to phrase onset
WINDOWS = (
    ("0-600 ms", (0.0, 0.6)),
    ("200-400 ms", (0.2, 0.4)),
    ("600-1200 ms", (0.6, 1.2)),
    ("0-2 s", (0.0, 2.0)),
)


def levels(env, t, window):
    """Per-participant dB of `window` against the baseline window."""
    base = (t >= BASELINE_S[0]) & (t < BASELINE_S[1])
    win = (t >= window[0]) & (t < window[1])
    return 20 * np.log10(env[:, win].mean(1) / env[:, base].mean(1))


def summarise(v):
    """Mean, two-sided 95% CI, one-sided p for an elevation above baseline."""
    t_stat = stats.ttest_1samp(v, 0).statistic
    lo, hi = stats.t.interval(0.95, v.size - 1, loc=v.mean(), scale=stats.sem(v))
    return {
        "n": int(v.size),
        "mean_dB": float(v.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "t": float(t_stat),
        "p_one_sided": float(1 - stats.t.cdf(t_stat, v.size - 1)),
        "n_above_zero": int((v > 0).sum()),
        "min_dB": float(v.min()),
        "max_dB": float(v.max()),
    }


def main():
    cfg = set_paths()
    D = cfg.out / "11_acoustic_monitoring"
    d = np.load(D / "acoustic_envelopes.npz", allow_pickle=True)
    subjects = [str(s) for s in d["subjects"]]
    env = {"overt": d["env_overt"], "covert": d["env_covert"]}
    t = (np.arange(env["overt"].shape[1]) * float(d["bin_s"])
         - float(d["phrase_onset_s"]))

    per_subject, rows = {}, []
    for label, window in WINDOWS:
        for cond, e in env.items():
            v = levels(e, t, window)
            per_subject[(label, cond)] = v
            rows.append(dict(window=label, condition=cond, **summarise(v)))

    with open(D / "acoustic_window_stats.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    with open(D / "acoustic_window_levels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        keys = list(per_subject)
        w.writerow(["subject"] + [f"{lab}_{cond}_dB" for lab, cond in keys])
        for i, s in enumerate(subjects):
            w.writerow([s] + [f"{per_subject[k][i]:.4f}" for k in keys])

    print(f"N = {len(subjects)}")
    for r in rows:
        print(f"  {r['condition']:6s} {r['window']:12s} {r['mean_dB']:+7.2f} dB "
              f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  "
              f"one-sided p = {r['p_one_sided']:.3f}  "
              f"above 0 dB: {r['n_above_zero']}/{r['n']}  "
              f"min {r['min_dB']:+.2f}  max {r['max_dB']:+.2f}")
    print(f"  wrote {D / 'acoustic_window_stats.csv'} "
          f"and {D / 'acoustic_window_levels.csv'}")


if __name__ == "__main__":
    main()
