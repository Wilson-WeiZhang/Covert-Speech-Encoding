#!/usr/bin/env python3
"""Evidence-of-absence add-on for the stimulus-attribute RSA (Supplementary Note S10).

s18/a4b report the partial correlations as a null (no window survives correction).
A null test cannot by itself say the effect is small; three additions turn it into a
bounded, positive statement, all on the per-participant values already computed:

  1. bootstrap 95% CI of the mean partial correlation (10 000 resamples)
  2. Bayes factor BF01 for the one-sample test against zero (JZS default prior,
     Cauchy scale r = 0.707; Rouder et al. 2009), per window and for the pooled
     200-400 ms discrimination window and the pooled 100-200 ms early window
  3. model fit vs. noise ceiling: paired test of each participant's leave-one-out
     ceiling (Nili et al. 2014, lower bound) minus that participant's plain Spearman
     correlation with each composite model, in the pooled windows -- i.e. whether the
     attribute models fall significantly short of what a perfect model could reach.

in : out/analysis/a4_neural_rdm.csv, a4_attr_extent.csv
out: out/analysis/a4e_rsa_evidence.txt, a4e_rsa_evidence_windows.csv
"""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr, ttest_1samp, t as tdist
from scipy.integrate import quad

D = Path(__file__).resolve().parent / 'out' / 'analysis'
iu = [(i, j) for j in range(5) for i in range(j)]

rows = list(csv.DictReader(open(D / 'a4_neural_rdm.csv')))
subs = sorted({r['subject'] for r in rows})
edge_starts = sorted({float(r['win_start_ms']) for r in rows})
step = edge_starts[1] - edge_starts[0]
n, nwin = len(subs), len(edge_starts)
si = {s: k for k, s in enumerate(subs)}; wi = {w: k for k, w in enumerate(edge_starts)}
NV = np.full((n, nwin, len(iu)), np.nan)
for r in rows:
    NV[si[r['subject']], wi[float(r['win_start_ms'])]] = [float(r[f'p{t+1}']) for t in range(len(iu))]
edges = np.array(edge_starts + [edge_starts[-1] + step])
kidx = np.where(edges[:-1] < 1500.0)[0]
wins = [(edges[k], edges[k + 1]) for k in kidx]

letters = [7, 14, 8, 11, 9]; words = [2, 2, 2, 2, 1]; syll = [2, 4, 3, 3, 3]
place = ['velar', 'alveolar', 'labiodental', 'vowel', 'alveolar']
manner = ['plosive', 'plosive', 'fricative', 'vowel', 'plosive']
ad = lambda v: np.array([abs(v[i] - v[j]) for i, j in iu], float)
cd = lambda v: np.array([0.0 if v[i] == v[j] else 1.0 for i, j in iu])
extent = np.array([float(r['value']) for r in csv.DictReader(open(D / 'a4_attr_extent.csv'))])
z = lambda v: (rankdata(v) - rankdata(v).mean()) / rankdata(v).std()
VIS = (z(ad(letters)) + z(ad(words)) + z(extent)) / 3
PHO = (z(ad(syll)) + z(cd(place)) + z(cd(manner))) / 3

def partial(nv, x, y):
    rn, rx, ry = rankdata(nv), rankdata(x), rankdata(y)
    def res(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    a, b = res(rn, ry), res(rx, ry)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan

P_vis = np.array([[partial(NV[s, k], VIS, PHO) for k in kidx] for s in range(n)])
P_pho = np.array([[partial(NV[s, k], PHO, VIS) for k in kidx] for s in range(n)])
R_vis = np.array([[spearmanr(NV[s, k], VIS)[0] for k in kidx] for s in range(n)])
R_pho = np.array([[spearmanr(NV[s, k], PHO)[0] for k in kidx] for s in range(n)])

# ---- JZS Bayes factor, one-sample t (Rouder et al. 2009, r = 0.707) -------------
def bf10_jzs(tval, N, r=0.707):
    v = N - 1
    def like(g):
        return (1 + N * g * r ** 2) ** -0.5 * (1 + tval ** 2 / ((1 + N * g * r ** 2) * v)) ** (-(v + 1) / 2) \
            * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g))
    num, _ = quad(like, 0, np.inf, limit=200)
    den = (1 + tval ** 2 / v) ** (-(v + 1) / 2)
    return num / den

def boot_ci(x, B=10000, seed=0):
    rng = np.random.default_rng(seed); x = x[~np.isnan(x)]
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(B)])
    return np.percentile(m, [2.5, 97.5])

def ceiling_lower(NVw):
    """Nili 2014 lower bound: each participant's Spearman with the leave-one-out mean RDM."""
    R = np.array([rankdata(v) for v in NVw])
    out = np.empty(len(R))
    for s in range(len(R)):
        loo = np.delete(R, s, 0).mean(0)
        out[s] = spearmanr(R[s], loo)[0]
    return out

out = []
def say(s=''):
    print(s); out.append(s)

say('S10 evidence-of-absence add-on (a4e)')
say(f'N = {n}, {len(wins)} windows of {step:.0f} ms')
say()
say('Per-window BF01 (JZS, r = .707) for the partial correlation against zero:')
say('%-12s %8s %8s %8s %8s' % ('window', 'VIS|P', 'BF01', 'PHO|V', 'BF01'))
wrows = []
for i, (a, b) in enumerate(wins):
    tv = ttest_1samp(P_vis[:, i], 0, nan_policy='omit'); tp = ttest_1samp(P_pho[:, i], 0, nan_policy='omit')
    bv = 1 / bf10_jzs(tv.statistic, n); bp = 1 / bf10_jzs(tp.statistic, n)
    say('%-12s %+8.3f %8.2f %+8.3f %8.2f' % (f'{a:.0f}-{b:.0f}', np.nanmean(P_vis[:, i]), bv, np.nanmean(P_pho[:, i]), bp))
    wrows.append((a, b, np.nanmean(P_vis[:, i]), tv.pvalue, bv, np.nanmean(P_pho[:, i]), tp.pvalue, bp))
bf_v = [r[4] for r in wrows]; bf_p = [r[7] for r in wrows]
say(f'BF01 range: visual {min(bf_v):.2f}-{max(bf_v):.2f}; phonological {min(bf_p):.2f}-{max(bf_p):.2f}')
say(f'windows with BF01 > 3 (moderate evidence for null): visual {sum(b > 3 for b in bf_v)}/{len(bf_v)}, '
    f'phonological {sum(b > 3 for b in bf_p)}/{len(bf_p)}')

def widx(lo, hi): return [i for i, (a, b) in enumerate(wins) if a >= lo and b <= hi]
for label, idx in (('POOLED 200-400 ms (discrimination window)', widx(200, 400)),
                   ('POOLED 100-200 ms (early window)', widx(100, 200))):
    say(); say(label)
    kk = [kidx[i] for i in idx]
    NVpool = NV[:, kk, :].mean(1)               # average RDM over the pooled windows
    ceil_lo = ceiling_lower(NVpool)
    for nm, P, R, comp in (('VISUAL|PHON', P_vis, R_vis, VIS), ('PHON/ARTIC|VISUAL', P_pho, R_pho, PHO)):
        pm = P[:, idx].mean(1); tt = ttest_1samp(pm, 0); lo, hi = boot_ci(pm)
        bf01 = 1 / bf10_jzs(tt.statistic, n)
        rm = np.array([spearmanr(NVpool[s], comp)[0] for s in range(n)])
        gap = ceil_lo - rm; tg = ttest_1samp(gap, 0)
        say(f'  {nm:<18s} partial rho = {pm.mean():+.3f}, bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], '
            f't({n-1}) = {tt.statistic:+.2f}, p = {tt.pvalue:.3f}, BF01 = {bf01:.2f}')
        say(f'  {"":<18s} plain rho vs ceiling: model {rm.mean():+.3f}, ceiling lower bound {ceil_lo.mean():+.3f}, '
            f'gap {gap.mean():+.3f}, t({n-1}) = {tg.statistic:+.2f}, p = {tg.pvalue:.4f} (one-sample, gap > 0)')

(D / 'a4e_rsa_evidence.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')
with open(D / 'a4e_rsa_evidence_windows.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['win_start_ms', 'win_end_ms', 'visual_partial', 'p_visual', 'BF01_visual',
                                   'phon_partial', 'p_phon', 'BF01_phon']); w.writerows(wrows)
print('\nwrote a4e_rsa_evidence.txt, a4e_rsa_evidence_windows.csv')
