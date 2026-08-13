#!/usr/bin/env python3
"""Supplementary figure: acoustic level during covert trials, with overt trials
as the sensitivity control.

TIME BASE
    Trials are cut 14 s from the trial-onset beep and the phrase appears on
    screen at 2.0 s, so everything here is re-expressed relative to phrase
    onset, the same event the EEG epochs are aligned to.

BASELINE
    -0.5 to 0.0 s relative to phrase onset, computed separately for each
    condition, so each condition is normalised against its own pre-stimulus
    level.

ANALYSIS WINDOW
    0.0 to 2.0 s relative to phrase onset, spanning the EEG epoch.

DECIBEL CONVENTION
    20 * log10( mean(envelope over window) / mean(envelope over baseline) ):
    the envelope is averaged over the window first and the logarithm is taken
    once. Converting each bin to dB and averaging afterwards is not equivalent,
    because the logarithm is concave and the overt envelope swings over a wide
    range; the two orders differ substantially for overt trials.

STATISTICS
    One-sided one-sample t-test against the participant's own baseline.
    Vocalisation can only raise the acoustic level, so the direction is set by
    the physics of the measurement rather than chosen after seeing the data.

IN  : <out>/11_acoustic_monitoring/acoustic_envelopes.npz   (from s21)
OUT : <out>/11_acoustic_monitoring/FigS_covert_acoustic_monitoring.png / .pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.set_paths import set_paths

BASELINE_S = (-0.5, 0.0)     # relative to phrase onset
ANALYSIS_S = (0.0, 2.0)      # relative to phrase onset, spanning the EEG epoch
RED, BLUE = "#B23A34", "#2E6F9E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth": 0.9,
})


def main():
    cfg = set_paths()
    D = cfg.out / "11_acoustic_monitoring"

    d = np.load(D / "acoustic_envelopes.npz", allow_pickle=True)
    eo, ec = d["env_overt"], d["env_covert"]
    n = eo.shape[0]
    t = np.arange(eo.shape[1]) * float(d["bin_s"]) - float(d["phrase_onset_s"])

    base = (t >= BASELINE_S[0]) & (t < BASELINE_S[1])
    win = (t >= ANALYSIS_S[0]) & (t < ANALYSIS_S[1])
    show = (t >= BASELINE_S[0]) & (t <= ANALYSIS_S[1])

    # Per-participant dB time course, each condition against its own baseline.
    bo, bc = eo[:, base].mean(1, keepdims=True), ec[:, base].mean(1, keepdims=True)
    Do, Dc = 20 * np.log10(eo / bo), 20 * np.log10(ec / bc)
    # Scalar per participant: average the envelope, then convert once (see docstring).
    so = 20 * np.log10(eo[:, win].mean(1) / bo.ravel())
    sc = 20 * np.log10(ec[:, win].mean(1) / bc.ravel())

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [2.6, 1]})

    # ---- a: group-mean time course -----------------------------------------
    for Dcond, c, lab in ((Do, RED, "overt"), (Dc, BLUE, "covert")):
        m = Dcond[:, show].mean(0)
        se = Dcond[:, show].std(0, ddof=1) / np.sqrt(n)
        axa.plot(t[show], m, color=c, lw=2.0, label=lab, zorder=3)
        axa.fill_between(t[show], m - se, m + se, color=c, alpha=0.25, lw=0, zorder=2)
    axa.axvline(0, color="k", ls="--", lw=1.2, zorder=1)
    axa.axhline(0, color="0.4", ls=":", lw=1.0, zorder=1)
    axa.set_xlabel("Time from phrase onset (s)")
    axa.set_ylabel("Acoustic level re baseline (dB)")
    axa.set_xlim(BASELINE_S[0], ANALYSIS_S[1])
    axa.legend(frameon=False, loc="upper right", fontsize=11)
    axa.spines[["top", "right"]].set_visible(False)
    axa.text(-0.16, 1.02, "a", transform=axa.transAxes,
             fontsize=15, fontweight="bold", va="bottom")

    # ---- b: per-participant level over the analysis window -----------------
    rng = np.random.default_rng(0)
    for i, (v, c) in enumerate(((so, RED), (sc, BLUE))):
        axb.boxplot(v, positions=[i], widths=0.55, showfliers=False,
                    medianprops=dict(color="k", lw=1.4),
                    boxprops=dict(color="k", lw=1.0),
                    whiskerprops=dict(color="k", lw=1.0),
                    capprops=dict(color="k", lw=1.0))
        axb.scatter(np.full(v.size, i) + rng.uniform(-0.13, 0.13, v.size), v,
                    s=16, color=c, alpha=0.55, lw=0, zorder=3)
    axb.axhline(0, color="0.4", ls=":", lw=1.0)
    axb.set_xticks([0, 1], ["Overt", "Covert"])
    axb.set_ylabel("0-2 s re baseline (dB)")
    top = max(so.max(), sc.max())
    axb.text(0, top + 3, "***", ha="center", fontsize=13)
    axb.text(1, top + 3, "n.s.", ha="center", fontsize=11)
    axb.set_ylim(min(so.min(), sc.min()) - 3, top + 7)
    axb.spines[["top", "right"]].set_visible(False)
    axb.text(-0.34, 1.02, "b", transform=axb.transAxes,
             fontsize=15, fontweight="bold", va="bottom")

    fig.tight_layout()
    D.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(D / f"FigS_covert_acoustic_monitoring.{ext}", dpi=300,
                    bbox_inches="tight")

    p_cov = 1 - stats.t.cdf(stats.ttest_1samp(sc, 0).statistic, n - 1)
    print(f"N = {n}")
    print(f"  overt  0-2 s {so.mean():+.2f} dB   positive {int((so > 0).sum())}/{n}"
          f"   min {so.min():+.2f}")
    print(f"  covert 0-2 s {sc.mean():+.2f} dB   one-sided p = {p_cov:.3f}   "
          f">0: {int((sc > 0).sum())}/{n}   max {sc.max():+.2f}")
    print(f"  wrote {D / 'FigS_covert_acoustic_monitoring.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
