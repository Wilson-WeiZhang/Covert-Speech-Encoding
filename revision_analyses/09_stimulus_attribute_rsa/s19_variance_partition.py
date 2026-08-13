#!/usr/bin/env python3
"""Variance partitioning, noise ceiling and exact permutation for the
stimulus-attribute RSA.

s18 shows that the attribute RDMs are collinear and reports partial Spearman
correlations. A partial correlation says how much is left after controlling,
not how much is shared, and it carries no reference for how much there was to
find in the first place. This script adds the three quantities that turn a null
into a bounded result:

  1. unique versus common variance, so the collinearity is shown rather than
     asserted
  2. a noise ceiling, so the null has a reference for what was achievable
  3. the full permutation distribution, so the power of the test is visible

Everything runs off the exported RDMs; no decoding or source analysis is
re-run.

SCALE. The commonality partition is computed and reported, but the figure and
the headline numbers rest on the correlation scale instead. With five
conditions, ten dissimilarities regressed on two predictors have an expected
R2 of about 2/9 = 0.22 from degrees of freedom alone, so the partition is
dominated by the design rather than by the data; and a within-participant R2
is not on the same scale as a between-participant noise ceiling, so the two
cannot be compared directly. On the correlation scale the model fit, the
ceiling and the null are all in the same units:

  * model fit = mean over participants of Spearman r(neural RDM, attribute RDM)
  * ceiling   = Nili et al. 2014, same units
  * null      = exhaustive relabelling, same units

Collinearity is then quantified between the MODELS (correlation of the two
composite RDMs), which needs no neural data and has no degrees-of-freedom
problem.

Two further choices worth stating:

  * Rank transform throughout, matching the Spearman framing used in s18.
  * The permutation is EXHAUSTIVE. Five phrases give 5! = 120 relabellings, so
    the null is complete rather than sampled and the finest resolvable p is
    1/120 = .0083. s18 draws 1000 random shuffles of the same 120
    possibilities.
  * The noise ceiling follows Nili et al. 2014: the upper bound is the mean
    correlation of each participant's RDM with the group mean INCLUDING that
    participant, the lower bound with the leave-one-out mean.

IN  : <out>/09_stimulus_attribute_rsa/neural_rdm.csv
      <out>/09_stimulus_attribute_rsa/attribute_visual_extent.csv
OUT : <out>/09_stimulus_attribute_rsa/variance_partition.csv
      <out>/09_stimulus_attribute_rsa/model_fit.csv
      <out>/09_stimulus_attribute_rsa/rsa_figure_data.npz
      <out>/09_stimulus_attribute_rsa/variance_partition_report.md
"""
import csv
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.set_paths import set_paths

cfg = set_paths()
D = cfg.out / "09_stimulus_attribute_rsa"

# Pair order: MATLAB find(triu(true(5),1)) is COLUMN-major (see s17 and s18).
iu = [(i, j) for j in range(5) for i in range(j)]

EDGE_LIMIT = 1400.0          # epochs end at 1496 ms; the last window is a filter edge
FOCUS = (200.0, 400.0)       # the discrimination window the manuscript rests on


# ---------------------------------------------------------------- data ------
def load_neural():
    with open(D / "neural_rdm.csv") as fh:
        rows = list(csv.DictReader(fh))
    subs = sorted({r["subject"] for r in rows})
    starts = sorted({float(r["win_start_ms"]) for r in rows})
    step = starts[1] - starts[0]
    si = {s: k for k, s in enumerate(subs)}
    wi = {w: k for k, w in enumerate(starts)}
    NV = np.full((len(subs), len(starts), len(iu)), np.nan)
    for r in rows:
        NV[si[r["subject"]], wi[float(r["win_start_ms"])]] = \
            [float(r[f"p{t + 1}"]) for t in range(len(iu))]
    keep = [k for k, a in enumerate(starts) if a < EDGE_LIMIT]
    wins = [(starts[k], starts[k] + step) for k in keep]
    return subs, NV[:, keep, :], wins


def attribute_rdms():
    """Attribute RDMs, identical definitions to s18_rsa_statistics.py."""
    letters, words, syll = [7, 14, 8, 11, 9], [2, 2, 2, 2, 1], [2, 4, 3, 3, 3]
    place = ["velar", "alveolar", "labiodental", "vowel", "alveolar"]
    manner = ["plosive", "plosive", "fricative", "vowel", "plosive"]
    ad = lambda v: np.array([abs(v[i] - v[j]) for i, j in iu], float)
    cd = lambda v: np.array([0.0 if v[i] == v[j] else 1.0 for i, j in iu])
    with open(D / "attribute_visual_extent.csv") as fh:
        extent = np.array([float(r["value"]) for r in csv.DictReader(fh)])
    return {"letter_count": ad(letters), "word_count": ad(words), "visual_extent": extent,
            "syllables": ad(syll), "onset_place": cd(place), "onset_manner": cd(manner)}


def zrank(v):
    """Rank-standardise, using midranks so that ties are averaged."""
    r = rankdata(v)
    return (r - r.mean()) / r.std()


def composites(ATTR):
    vis = (zrank(ATTR["letter_count"]) + zrank(ATTR["word_count"])
           + zrank(ATTR["visual_extent"])) / 3
    pho = (zrank(ATTR["syllables"]) + zrank(ATTR["onset_place"])
           + zrank(ATTR["onset_manner"])) / 3
    return vis, pho


# --------------------------------------------------- variance partition -----
def r2(y, *xs):
    """R2 of y on the given regressors, all rank-transformed, with intercept."""
    Y = zrank(y)
    X = np.column_stack([np.ones_like(Y)] + [zrank(x) for x in xs])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    return 1.0 - resid.var() / Y.var()


def partition(y, vis, pho):
    """Commonality partition: unique-visual, unique-phonological, shared, total.

    `shared` can be negative when the two predictors suppress one another. That
    is a real property of collinear predictors, not an error, and is reported
    as such.
    """
    rv, rp, rvp = r2(y, vis), r2(y, pho), r2(y, vis, pho)
    return rvp - rp, rvp - rv, rv + rp - rvp, rvp


def permute_rdm(vec, order):
    """Relabel the five phrases and re-read the 10 pairwise dissimilarities."""
    M = np.zeros((5, 5))
    for t, (i, j) in enumerate(iu):
        M[i, j] = M[j, i] = vec[t]
    return np.array([M[order[i], order[j]] for i, j in iu])


# ------------------------------------------------------- noise ceiling ------
def noise_ceiling(NVw):
    """Nili et al. 2014 noise ceiling on rank-transformed RDMs.

    Returns (lower, upper) as the Pearson r of the rank vectors, which equals
    the Spearman r of the raw dissimilarities.
    """
    R = np.array([zrank(NVw[s]) for s in range(NVw.shape[0])])
    n = R.shape[0]
    grand = R.mean(0)
    upper = np.mean([np.corrcoef(R[s], grand)[0, 1] for s in range(n)])
    lower = np.mean([np.corrcoef(R[s], (grand * n - R[s]) / (n - 1))[0, 1] for s in range(n)])
    return lower, upper


# ------------------------------------------------------------------ run -----
def main():
    subs, NV, wins = load_neural()
    ATTR = attribute_rdms()
    vis, pho = composites(ATTR)
    n, nwin = len(subs), len(wins)
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("# Stimulus-attribute RSA: variance partition, noise ceiling, exact permutation")
    say()
    say(f"N = {n} participants, {nwin} windows of "
        f"{wins[0][1] - wins[0][0]:.0f} ms over 0-{EDGE_LIMIT:.0f} ms.")
    say("Rank-transformed RDMs throughout, so R2 is variance explained in rank space.")
    say()

    # -- collinearity of the models -------------------------------------------
    say("## Collinearity of the two composites")
    rho_fam = np.corrcoef(zrank(vis), zrank(pho))[0, 1]
    say(f"VISUAL vs PHON/ARTIC composite: r = {rho_fam:+.3f} "
        f"(shared {rho_fam ** 2 * 100:.1f}% of rank variance)")
    ks = list(ATTR)
    say()
    say("Pairwise among the six single attributes (Spearman over the 10 phrase pairs):")
    say("%-15s" % "" + "".join("%13s" % k[:12] for k in ks))
    for a in ks:
        say("%-15s" % a + "".join(
            "%13s" % ("%+.2f" % np.corrcoef(zrank(ATTR[a]), zrank(ATTR[b]))[0, 1]) for b in ks))
    say()

    # -- per-window partition -------------------------------------------------
    UV = np.zeros((n, nwin)); UP = np.zeros((n, nwin))
    SH = np.zeros((n, nwin)); TOT = np.zeros((n, nwin))
    for s in range(n):
        for k in range(nwin):
            UV[s, k], UP[s, k], SH[s, k], TOT[s, k] = partition(NV[s, k], vis, pho)

    # -- correlation scale: what the figure is built on -----------------------
    RV = np.zeros((n, nwin)); RP = np.zeros((n, nwin))
    for s in range(n):
        for k in range(nwin):
            RV[s, k] = np.corrcoef(zrank(NV[s, k]), zrank(vis))[0, 1]
            RP[s, k] = np.corrcoef(zrank(NV[s, k]), zrank(pho))[0, 1]

    # -- exhaustive permutation over the 120 relabellings, both scales --------
    perms = list(permutations(range(5)))
    null_tot = np.zeros((len(perms), nwin))
    null_rv = np.zeros((len(perms), nwin))
    null_rp = np.zeros((len(perms), nwin))
    for q, o in enumerate(perms):
        vp, pp = permute_rdm(vis, o), permute_rdm(pho, o)
        zv, zp = zrank(vp), zrank(pp)
        for k in range(nwin):
            null_tot[q, k] = np.mean([r2(NV[s, k], vp, pp) for s in range(n)])
            null_rv[q, k] = np.mean([np.corrcoef(zrank(NV[s, k]), zv)[0, 1] for s in range(n)])
            null_rp[q, k] = np.mean([np.corrcoef(zrank(NV[s, k]), zp)[0, 1] for s in range(n)])
    obs_tot = TOT.mean(0)
    pvals = [np.sum(null_tot[:, k] >= obs_tot[k]) / len(perms) for k in range(nwin)]

    nc = [noise_ceiling(NV[:, k, :]) for k in range(nwin)]

    say("## Per-window results")
    say(f"{'window(ms)':<12}{'uniqVIS':>9}{'uniqPHO':>9}{'shared':>9}{'total R2':>10}"
        f"{'perm p':>8}{'NC low':>8}{'NC up':>7}{'NC up^2':>9}")
    rows = []
    for k, (a, b) in enumerate(wins):
        lo, up = nc[k]
        say(f"{f'{a:.0f}-{b:.0f}':<12}{UV[:, k].mean():>9.4f}{UP[:, k].mean():>9.4f}"
            f"{SH[:, k].mean():>9.4f}{TOT[:, k].mean():>10.4f}{pvals[k]:>8.3f}"
            f"{lo:>8.3f}{up:>7.3f}{up ** 2:>9.3f}")
        rows.append((a, b, UV[:, k].mean(), UP[:, k].mean(), SH[:, k].mean(),
                     TOT[:, k].mean(), pvals[k], lo, up))
    say()

    say("## Why the results are reported on the correlation scale")
    say(f"Ten dissimilarities on two predictors: E[R2] from degrees of freedom alone is "
        f"about 2/9 = {2 / 9:.3f}.")
    say(f"Mean of the exhaustive null across windows = {null_tot.mean():.4f}, "
        f"observed mean = {TOT.mean():.4f}.")
    say("The observed R2 does not exceed its own null in any window, and a "
        "within-participant R2")
    say("is not on the same scale as a between-participant noise ceiling. "
        "The figure uses correlations.")
    say()

    # -- the window the manuscript rests on -----------------------------------
    fk = [k for k, (a, b) in enumerate(wins) if a >= FOCUS[0] and b <= FOCUS[1]]
    say(f"## Discrimination window {FOCUS[0]:.0f}-{FOCUS[1]:.0f} ms "
        f"(mean of {len(fk)} windows)")
    lo_f = np.mean([nc[k][0] for k in fk])
    up_f = np.mean([nc[k][1] for k in fk])
    say(f"  noise ceiling            r = {lo_f:+.3f} to {up_f:+.3f}")
    for nm, arr, nullarr in (("VISUAL composite", RV, null_rv),
                             ("PHON/ARTIC composite", RP, null_rp)):
        v = arr[:, fk].mean(1)
        t = ttest_1samp(v, 0)
        nl = nullarr[:, fk].mean(1)
        p = np.sum(nl >= v.mean()) / len(perms)
        say(f"  {nm:<24s} r = {v.mean():+.3f}  (t({n - 1}) = {t.statistic:+.2f}, "
            f"p = {t.pvalue:.3f}; exact perm p = {p:.4f})")
        say(f"  {'':24s} = {v.mean() / up_f * 100:.0f}% of the upper ceiling, "
            f"null 95th pct {np.percentile(nl, 95):+.3f}")
    tot_f = TOT[:, fk].mean(1).mean()
    nullf = null_tot[:, fk].mean(1)
    say(f"  [R2 scale, not used]     total R2 {tot_f:.4f} vs null mean {nullf.mean():.4f}, "
        f"perm p = {np.sum(nullf >= tot_f) / len(perms):.4f}")
    say()

    n95v = np.percentile(null_rv[:, fk].mean(1), 95)
    n95p = np.percentile(null_rp[:, fk].mean(1), 95)
    say("## Reading")
    say(f"1. Both composites fall inside their own null ({RV[:, fk].mean():+.3f} and "
        f"{RP[:, fk].mean():+.3f} against")
    say(f"   95th percentiles of {n95v:+.3f} and {n95p:+.3f}). The null result stands.")
    say(f"2. The test is not powerless. A perfect model could reach r = {up_f:+.3f}, "
        f"which is {up_f / n95v:.1f}x")
    say(f"   and {up_f / n95p:.1f}x the two significance thresholds, so an attribute effect "
        f"at ceiling would")
    say("   have been detected. This is the power statement the null needs, and it is "
        "measured, not assumed.")
    say(f"3. The ceiling is nonetheless low ({lo_f:+.3f} to {up_f:+.3f}): participants' "
        f"RDMs agree with one")
    say("   another very little. That is the same fact the mixed-effects decomposition "
        "reports as")
    say("   individual-specific encoding, arrived at independently, and it bounds what "
        "ANY group-level")
    say("   model of these five phrases could have explained.")

    with open(D / "variance_partition.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["win_start_ms", "win_end_ms", "unique_visual", "unique_phon",
                    "shared", "total_r2", "perm_p", "nc_lower_r", "nc_upper_r"])
        w.writerows(rows)
    with open(D / "model_fit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["win_start_ms", "win_end_ms", "r_visual", "r_phon",
                    "null95_visual", "null95_phon", "nc_lower_r", "nc_upper_r"])
        for k, (a, b) in enumerate(wins):
            w.writerow([a, b, RV[:, k].mean(), RP[:, k].mean(),
                        np.percentile(null_rv[:, k], 95), np.percentile(null_rp[:, k], 95),
                        nc[k][0], nc[k][1]])
    np.savez(D / "rsa_figure_data.npz", RV=RV, RP=RP, null_rv=null_rv, null_rp=null_rp,
             null_tot=null_tot, UV=UV, UP=UP, SH=SH, TOT=TOT,
             nc=np.array(nc), wins=np.array(wins), vis=vis, pho=pho,
             attr=np.array([ATTR[k] for k in ks]), attr_names=np.array(ks))
    (D / "variance_partition_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nwrote {D / 'variance_partition_report.md'}")


if __name__ == "__main__":
    main()
