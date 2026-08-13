#!/usr/bin/env python3
"""Supplementary figure for the stimulus-attribute RSA: collinearity of the
attribute models, and the null with its two references.

The point of the figure is to make "the attribute models explain nothing"
readable as a bounded result rather than as a missing analysis. Two panels:

  a  collinearity is shown rather than asserted: the six attribute RDMs
     correlated against each other. Several pairs are strongly related, and
     with five conditions they cannot be separated.
  b  the null with its two references. Observed RDM correlation per window for
     each family composite, the exhaustive permutation threshold, and the Nili
     et al. 2014 noise-ceiling band. The gap between the ceiling and the
     threshold is the power of the test, made visible rather than argued.

The figure deliberately does not show a variance partition of the neural data:
at five conditions that partition is dominated by degrees of freedom (see the
s19 docstring).

IN  : <out>/09_stimulus_attribute_rsa/rsa_figure_data.npz     (from s19)
      <out>/09_stimulus_attribute_rsa/rsa_VISUAL_combined.csv (from s16)
      <out>/09_stimulus_attribute_rsa/rsa_PHON_ARTIC_combined.csv
OUT : <out>/09_stimulus_attribute_rsa/FigS_rsa_noise_ceiling.png / .pdf
"""
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.set_paths import set_paths

cfg = set_paths()
D = cfg.out / "09_stimulus_attribute_rsa"

FOCUS = (200.0, 400.0)
C_VIS, C_PHO = "#4C72B0", "#C44E52"
C_CEIL, C_NULL = "#BFBFBF", "#7F7F7F"

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42,
})

d = np.load(D / "rsa_figure_data.npz", allow_pickle=True)
RV, RP = d["RV"], d["RP"]
null_rv, null_rp = d["null_rv"], d["null_rp"]
nc, wins = d["nc"], d["wins"]
attr, names = d["attr"], [str(x) for x in d["attr_names"]]
vis, pho = d["vis"], d["pho"]
n = RV.shape[0]


def fwe(fn):
    """Family-wise permutation p per window, read from the CSVs written by s16.

    The CSVs carry 15 windows (to 1500 ms) while the figure data carry 14 (to
    1400 ms), so align on window start rather than assuming equal length.
    """
    with open(D / fn) as fh:
        rows = list(csv.DictReader(fh))
    by_start = {float(r["win_start_ms"]): float(r["p_perm_fwe"]) for r in rows}
    return np.array([by_start[w0] for w0 in wins[:, 0]])


pv_fwe = fwe("rsa_VISUAL_combined.csv")
pp_fwe = fwe("rsa_PHON_ARTIC_combined.csv")

# Shared rank variance between the two family composites, reported so the
# legend can quote it.
r_fam = spearmanr(vis, pho).statistic
shared = r_fam ** 2
print(f"  FWE p: visual min {pv_fwe.min():.3f}, phon./artic. min {pp_fwe.min():.3f}"
      f"  -> significant windows: {(pv_fwe < .05).sum()} / {(pp_fwe < .05).sum()} of {len(wins)}")


def z(v):
    """Rank-standardise with MIDRANKS for ties.

    word_count, onset_place and onset_manner are heavily tied. Breaking ties
    arbitrarily, as argsort-based ranking does, silently changes the composite
    correlation, so ties must be averaged here exactly as they are in s19.
    """
    r = rankdata(v)
    return (r - r.mean()) / r.std()


fig = plt.figure(figsize=(7.2, 2.55))
gs = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.72], wspace=0.42,
                      left=0.075, right=0.985, top=0.86, bottom=0.19)

# ---------------------------------------------------------------- panel a ---
ax = fig.add_subplot(gs[0])
K = len(names)
M = np.array([[np.corrcoef(z(attr[i]), z(attr[j]))[0, 1] for j in range(K)] for i in range(K)])
Mplot = M.copy()
np.fill_diagonal(Mplot, np.nan)          # the diagonal is 1 by construction, not a finding
cmap = mpl.colormaps["RdBu_r"].copy()
cmap.set_bad("#F0F0F0")
im = ax.imshow(Mplot, cmap=cmap, vmin=-1, vmax=1)
short = ["letters", "words", "extent", "syllables", "place", "manner"]
ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels(short, rotation=45, ha="right")
ax.set_yticklabels(short)
for i in range(K):
    for j in range(K):
        if i == j:
            continue
        ax.text(j, i, f"{M[i, j]:+.2f}".replace("+0.", ".").replace("-0.", "-."),
                ha="center", va="center", fontsize=5.4,
                color="white" if abs(M[i, j]) > 0.62 else "black")
# outline the two attribute families
ax.plot([-0.5, 2.5, 2.5, -0.5, -0.5], [-0.5, -0.5, 2.5, 2.5, -0.5], color=C_VIS, lw=1.1)
ax.plot([2.5, 5.5, 5.5, 2.5, 2.5], [2.5, 2.5, 5.5, 5.5, 2.5], color=C_PHO, lw=1.1)
ax.set_xlim(-0.5, K - 0.5)
ax.set_ylim(K - 0.5, -0.5)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, ticks=[-1, 0, 1])
cb.outline.set_linewidth(0.6)
cb.ax.tick_params(width=0.6, length=2)
cb.set_label("Spearman $r$", labelpad=1)
ax.text(-0.30, 1.14, "a", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

# ---------------------------------------------------------------- panel b ---
ax = fig.add_subplot(gs[1])
x = wins.mean(1)
ax.axvspan(FOCUS[0], FOCUS[1], color="#F2C14E", alpha=0.20, lw=0, zorder=0)
ax.fill_between(x, nc[:, 0], nc[:, 1], color=C_CEIL, alpha=0.85, lw=0, zorder=1,
                label="noise ceiling")
ax.plot(x, np.percentile(null_rv, 95, axis=0), color=C_NULL, ls=(0, (3, 2)), lw=0.9,
        zorder=2, label="permutation 95th pct")
ax.plot(x, np.percentile(null_rp, 95, axis=0), color=C_NULL, ls=(0, (3, 2)), lw=0.9, zorder=2)
for arr, c, lab in ((RV, C_VIS, "visual"), (RP, C_PHO, "phon./artic.")):
    m = arr.mean(0)
    se = arr.std(0, ddof=1) / np.sqrt(n)
    ax.fill_between(x, m - se, m + se, color=c, alpha=0.20, lw=0, zorder=3)
    ax.plot(x, m, color=c, lw=1.3, zorder=4, label=lab)
ax.axhline(0, color="black", lw=0.5, zorder=1)

# Significance strip: one marker per window per family, filled where the
# permutation FWE p < .05. Showing every tested window explicitly is what makes
# an empty strip a result rather than an omission.
for arr_p, c, y in ((pv_fwe, C_VIS, -0.175), (pp_fwe, C_PHO, -0.205)):
    sig = arr_p < 0.05
    ax.scatter(x[~sig], np.full((~sig).sum(), y), s=5, facecolors="none",
               edgecolors=c, linewidths=0.5, zorder=5, clip_on=False)
    if sig.any():
        ax.scatter(x[sig], np.full(sig.sum(), y), s=5, c=c, zorder=5, clip_on=False)

ax.set_xlabel("Window centre (ms)", labelpad=2)
ax.set_ylabel("RDM correlation ($r$)", labelpad=2)
ax.set_xlim(0, 1400)
ax.set_xticks(np.arange(0, 1401, 200))
ax.set_ylim(-0.235, 0.305)
ax.set_yticks(np.arange(-0.10, 0.21, 0.05))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=5.8, ncol=2, loc="upper left",
          bbox_to_anchor=(-0.015, 1.03), handlelength=1.4, borderaxespad=0.0,
          labelspacing=0.22, columnspacing=1.0)
ax.text(-0.105, 1.12, "b", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")

for ext in ("png", "pdf"):
    fig.savefig(D / f"FigS_rsa_noise_ceiling.{ext}", dpi=400,
                bbox_inches="tight", facecolor="white")
print(f"wrote {D / 'FigS_rsa_noise_ceiling.png'} (+ .pdf)")

fk = [k for k, (a, b) in enumerate(wins) if a >= FOCUS[0] and b <= FOCUS[1]]
print(f"  focus {FOCUS[0]:.0f}-{FOCUS[1]:.0f} ms: "
      f"vis r = {RV[:, fk].mean():+.3f}, pho r = {RP[:, fk].mean():+.3f}, "
      f"ceiling {nc[fk, 0].mean():+.3f} to {nc[fk, 1].mean():+.3f}, "
      f"shared {shared * 100:.1f}%")
