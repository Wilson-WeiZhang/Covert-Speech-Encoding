#!/usr/bin/env python3
"""Statistics for the stimulus-attribute RSA, on the neural RDMs from s16/s17.

Three properties of this design shape the analysis:

1. WINDOW RANGE. The epochs end at 1496 ms, so the 1400-1500 ms window is a
   filter edge and is excluded. Windows run from 0 to 1400 ms.

2. COLLINEARITY. With only five conditions the attribute RDMs cannot be
   orthogonalised: letter_count and syllables correlate rho = +0.60 across the
   10 phrase pairs, onset_manner and syllables -0.57. A simple correlation
   therefore cannot attribute an effect to the visual rather than the
   phonological family. The main test is a PARTIAL correlation of each
   family's composite RDM controlling for the other's.

3. CONFIRMATORY VS EXPLORATORY. The prediction is specified in advance: visual
   attributes should dominate early (100-200 ms) and phonological/articulatory
   attributes should dominate in the discrimination window (200-400 ms). That
   is a confirmatory 2x2 (family x window) test on two pre-specified windows,
   reported alongside the exploratory sweep over all windows.

Permutations shuffle the phrase labels of the attribute RDM, which is the
correct exchangeability unit because the neural RDMs are held fixed. The
maximum over windows in each permutation gives family-wise error control.

IN  : <out>/09_stimulus_attribute_rsa/neural_rdm.csv
      <out>/09_stimulus_attribute_rsa/attribute_visual_extent.csv
      <out>/09_stimulus_attribute_rsa/rsa_letter_count.csv  (pair-order check)
OUT : <out>/09_stimulus_attribute_rsa/rsa_statistics.txt
      <out>/09_stimulus_attribute_rsa/rsa_partial_correlations.csv
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.set_paths import set_paths

cfg = set_paths()
D = cfg.out / "09_stimulus_attribute_rsa"

# Pair order. MATLAB's find(triu(true(5),1)) is COLUMN-major:
#   1-2, 1-3, 2-3, 1-4, 2-4, 3-4, 1-5, 2-5, 3-5, 4-5
# A row-major list would silently pair the wrong phrases with the wrong
# dissimilarities and still produce plausible numbers, so the order is
# reproduced exactly and verified below against the MATLAB output.
iu = [(i, j) for j in range(5) for i in range(j)]

EDGE_LIMIT = 1400.0

# ---- neural RDMs ------------------------------------------------------------
with open(D / "neural_rdm.csv") as fh:
    neural_rows = list(csv.DictReader(fh))
subs = sorted({r["subject"] for r in neural_rows})
edge_starts = sorted({float(r["win_start_ms"]) for r in neural_rows})
step = edge_starts[1] - edge_starts[0]
n, nwin = len(subs), len(edge_starts)
si = {s: k for k, s in enumerate(subs)}
wi = {w: k for k, w in enumerate(edge_starts)}
NV = np.full((n, nwin, len(iu)), np.nan)
for r in neural_rows:
    NV[si[r["subject"]], wi[float(r["win_start_ms"])]] = \
        [float(r[f"p{t + 1}"]) for t in range(len(iu))]
edges = np.array(edge_starts + [edge_starts[-1] + step])

keep = edges[:-1] < EDGE_LIMIT
wins = [(edges[k], edges[k + 1]) for k in range(nwin) if keep[k]]
kidx = np.where(keep)[0]

# ---- attribute RDMs, rebuilt in the SAME pair order -------------------------
letters = [7, 14, 8, 11, 9]
words = [2, 2, 2, 2, 1]
syll = [2, 4, 3, 3, 3]
place = ["velar", "alveolar", "labiodental", "vowel", "alveolar"]
manner = ["plosive", "plosive", "fricative", "vowel", "plosive"]


def ad(v):
    """Dissimilarity of a numeric attribute: absolute difference."""
    return np.array([abs(v[i] - v[j]) for i, j in iu], float)


def cd(v):
    """Dissimilarity of a categorical attribute: 0 if equal, 1 if not."""
    return np.array([0.0 if v[i] == v[j] else 1.0 for i, j in iu])


with open(D / "attribute_visual_extent.csv") as fh:
    extent = np.array([float(r["value"]) for r in csv.DictReader(fh)])

ATTR = {"letter_count": ad(letters), "word_count": ad(words), "visual_extent": extent,
        "syllables": ad(syll), "onset_place": cd(place), "onset_manner": cd(manner)}


def z(v):
    """Rank-standardise, using midranks so that ties are averaged."""
    r = rankdata(v)
    return (r - r.mean()) / r.std()


VIS = (z(ATTR["letter_count"]) + z(ATTR["word_count"]) + z(ATTR["visual_extent"])) / 3
PHO = (z(ATTR["syllables"]) + z(ATTR["onset_place"]) + z(ATTR["onset_manner"])) / 3

out = []


def say(s=""):
    print(s)
    out.append(s)


# ---- verify the pair order against the MATLAB output ------------------------
with open(D / "rsa_letter_count.csv") as fh:
    chk = list(csv.DictReader(fh))
lc = ad(letters)
mine = [np.nanmean([spearmanr(NV[s, k], lc)[0] for s in range(n)]) for k in range(nwin)]
theirs = [float(r["mean_rho"]) for r in chk]
dmax = max(abs(a - b) for a, b in zip(mine, theirs))
if dmax > 1e-6:
    raise SystemExit(f"pair order mismatch: max |python - matlab| = {dmax:.3g} "
                     f"on letter_count. Fix `iu` before trusting anything below.")

say("Stimulus-attribute RSA -- statistics")
say(f"(pair order verified against the MATLAB output, max diff {dmax:.1e})")
say(f"N = {n} participants, {len(wins)} windows of {step:.0f} ms over "
    f"0-{EDGE_LIMIT:.0f} ms (1400-1500 ms dropped: epochs end at 1496 ms)")
say()
say("COLLINEARITY of the attribute RDMs (Spearman over the 10 phrase pairs):")
ks = list(ATTR)
say("%-15s" % "" + "".join("%13s" % k[:12] for k in ks))
for a in ks:
    say("%-15s" % a + "".join("%13s" % ("%+.2f" % spearmanr(ATTR[a], ATTR[b])[0]) for b in ks))
r_fam = spearmanr(VIS, PHO)[0]
say(f"\nVISUAL composite vs PHON/ARTIC composite: rho = {r_fam:+.3f}")
say("With five conditions these cannot be orthogonalised, so single-attribute")
say("correlations cannot be read as attributions. Partial correlations below.")


# ---- partial correlation, each family controlling for the other -------------
def partial(nv, x, y):
    """Spearman partial correlation of nv with x, controlling for y.

    All three vectors are rank-transformed, then nv and x are residualised on y
    and the residuals correlated.
    """
    rn, rx, ry = rankdata(nv), rankdata(x), rankdata(y)

    def res(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]

    a, b = res(rn, ry), res(rx, ry)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


P_vis = np.array([[partial(NV[s, k], VIS, PHO) for k in kidx] for s in range(n)])
P_pho = np.array([[partial(NV[s, k], PHO, VIS) for k in kidx] for s in range(n)])

say()
say("PARTIAL correlations (mean over participants; * uncorrected p < .05)")
say("%-12s %10s %10s" % ("window(ms)", "VISUAL|P", "PHON|V"))
csv_rows = []
for i, (a, b) in enumerate(wins):
    mv, mp = np.nanmean(P_vis[:, i]), np.nanmean(P_pho[:, i])
    pv = ttest_1samp(P_vis[:, i], 0, nan_policy="omit").pvalue
    pp = ttest_1samp(P_pho[:, i], 0, nan_policy="omit").pvalue
    say("%-12s %9.3f%s %9.3f%s" % (f"{a:.0f}-{b:.0f}", mv, "*" if pv < .05 else " ",
                                   mp, "*" if pp < .05 else " "))
    csv_rows.append((a, b, mv, pv, mp, pp))


# ---- pre-specified 2x2: family x window -------------------------------------
def widx(lo, hi):
    return [i for i, (a, b) in enumerate(wins) if a >= lo and b <= hi]


EARLY, LATE = widx(100, 200), widx(200, 400)
say()
say("PRE-SPECIFIED TEST: visual attributes should dominate early (100-200 ms), "
    "phonological/articulatory attributes in the discrimination window (200-400 ms)")
ve, vl = P_vis[:, EARLY].mean(1), P_vis[:, LATE].mean(1)
pe, pl = P_pho[:, EARLY].mean(1), P_pho[:, LATE].mean(1)
for nm, e, l in (("VISUAL|PHON", ve, vl), ("PHON/ARTIC|VISUAL", pe, pl)):
    te, tl = ttest_1samp(e, 0), ttest_1samp(l, 0)
    say(f"  {nm:<20s} early {e.mean():+.3f} (t={te.statistic:+.2f}, p={te.pvalue:.3f})   "
        f"late {l.mean():+.3f} (t={tl.statistic:+.2f}, p={tl.pvalue:.3f})")
inter = (pl - pe) - (vl - ve)
ti = ttest_1samp(inter, 0)
say(f"  interaction (PHON late-early) - (VISUAL late-early) = {inter.mean():+.3f}, "
    f"t({len(inter) - 1}) = {ti.statistic:+.2f}, p = {ti.pvalue:.3f}")
say("  the prediction requires a positive, significant interaction")

# ---- permutation, max over windows (FWE) ------------------------------------
rng = np.random.default_rng(0)
NPERM = 1000


def null_max(comp, ctrl):
    """Null distribution of the maximum partial correlation across windows."""
    C = np.zeros((5, 5))
    K = np.zeros((5, 5))
    for t, (i, j) in enumerate(iu):
        C[i, j] = C[j, i] = comp[t]
        K[i, j] = K[j, i] = ctrl[t]
    mx = np.empty(NPERM)
    for q in range(NPERM):
        o = rng.permutation(5)
        cperm = np.array([C[o[i], o[j]] for i, j in iu])
        kperm = np.array([K[o[i], o[j]] for i, j in iu])
        vals = [np.nanmean([partial(NV[s, k], cperm, kperm) for s in range(n)]) for k in kidx]
        mx[q] = np.nanmax(vals)
    return mx


say()
say(f"permutation test, {NPERM} shuffles of the phrase labels, max over windows (FWE):")
for nm, comp, ctrl, obs in (("VISUAL|PHON", VIS, PHO, np.nanmean(P_vis, 0)),
                            ("PHON/ARTIC|VISUAL", PHO, VIS, np.nanmean(P_pho, 0))):
    nulls = null_max(comp, ctrl)
    p = [(1 + (nulls >= o).sum()) / (NPERM + 1) for o in obs]
    sig = [(wins[i], obs[i], p[i]) for i in range(len(wins)) if p[i] < .05]
    if sig:
        say(f"  {nm}: " + "; ".join(f"{a:.0f}-{b:.0f} ms rho={r:+.3f} p={q:.4f}"
                                    for (a, b), r, q in sig))
    else:
        say(f"  {nm}: no window survives FWE")

with open(D / "rsa_statistics.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
with open(D / "rsa_partial_correlations.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["win_start_ms", "win_end_ms", "visual_partial", "p_visual",
                "phon_partial", "p_phon"])
    w.writerows(csv_rows)
print("\nwrote rsa_statistics.txt, rsa_partial_correlations.csv")
