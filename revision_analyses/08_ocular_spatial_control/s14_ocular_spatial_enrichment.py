#!/usr/bin/env python3
"""Spatial test: are the discriminative parcels where ocular leakage would land?

Ocular artefact projects maximally onto frontopolar and orbital cortex. If the
phrase discrimination observed in source space were residual ocular leakage,
the parcels carrying a significant phrase effect should be concentrated in that
territory. This script turns that expectation into a number.

TEST. The group-level rmANOVA maps (148 parcels x 30 time windows) are taken as
given, and the question asked of them is whether the FDR-significant
parcel-window pairs are enriched in an anatomically pre-specified
ocular-proximal parcel set relative to that set's share of the atlas. Fisher
exact tests are reported one-sided in both directions (enrichment and
depletion).

Two levels of analysis are printed. The pair-level Fisher test treats all
148 x 30 = 4440 cells as independent observations, which they are not: the 30
adjacent 50 ms windows of one parcel are strongly autocorrelated, and homologous
left/right parcels are not independent either, so the pair-level p value is
inflated by pseudo-replication. The parcel-level test counts each parcel once
("does it show a significant effect in any window?"), which removes the
pseudo-replication; the resulting hypergeometric test is exact and needs no
independence approximation. The parcel-level result is the one to report.

PARCEL SET. Every member is written out and matched on its exact base name,
never by substring. The CORE set is the orbital and frontopolar surface, i.e.
the cortex directly above the orbits (9 bilateral families = 18 parcels). The
BROAD set additionally takes every remaining frontal parcel (superior, middle
and inferior frontal gyri and sulci) as a sensitivity analysis, so the
conclusion cannot depend on how tightly the set was drawn.

Both maps are tested: the theta-band map behind the spatial claim of Fig. 3 and
the broadband map behind the window shown in Fig. 2b.

The decoding counterpart of this test is s15_ocular_roi_decoding.m, which asks
the same question with the classifier instead of the significance map.

INPUT
  <repo>/config/EEG_ROI_LABELS.csv   atlas label table (ROI_LABEL_FILE overrides)
  cfg.out/s1_rmanova_theta.mat       theta-band rmANOVA map, variable q_values
  cfg.out/c3_2_rmanova_results.mat   broadband rmANOVA map, variable q_values
                                     (RMANOVA_THETA / RMANOVA_BROADBAND override
                                     the two map locations)

OUTPUT
  a report on stdout; no files are written

Run:  python s14_ocular_spatial_enrichment.py
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.set_paths import set_paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

N_PARCELS = 148
ALPHA = 0.05

OCULAR_CORE = [
    'G_and_S_frontomargin', 'G_and_S_transv_frontopol', 'G_front_inf-Orbital',
    'G_orbital', 'G_rectus', 'S_orbital-H_Shaped', 'S_orbital_lateral',
    'S_orbital_med-olfact', 'S_suborbital',
]
FRONTAL_EXTRA = [
    'G_front_inf-Opercular', 'G_front_inf-Triangul', 'G_front_middle', 'G_front_sup',
    'S_front_inf', 'S_front_middle', 'S_front_sup',
]


def resolve(env_var: str, default: Path) -> Path:
    """Path from an environment override, else the default; must exist."""
    path = Path(os.environ[env_var]).expanduser() if os.environ.get(env_var) else default
    if not path.is_file():
        raise FileNotFoundError(f'{path} not found (set {env_var} to override)')
    return path


def load_labels(label_file: Path) -> list[str]:
    """Parcel names ordered by the source-data index eeg_idx = 1..148."""
    with open(label_file, encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == N_PARCELS, len(rows)
    names: list[str | None] = [None] * N_PARCELS
    for r in rows:
        names[int(r['eeg_idx']) - 1] = r['eeg_name']
    assert all(names), 'gap in eeg_idx 1..148'
    return names


def build_set(names: list[str], bases: list[str]) -> list[int]:
    """Exact base-name match on 'BASE L' / 'BASE R'; every base must hit exactly 2."""
    idx: list[int] = []
    for b in bases:
        hit = [i for i, n in enumerate(names) if n in (f'{b} L', f'{b} R')]
        assert len(hit) == 2, f'{b} matched {len(hit)} parcels, expected 2 (L+R)'
        idx += hit
    return sorted(idx)


def report(q: np.ndarray, parcel_set: list[int], label: str) -> None:
    """Print the pair-level and parcel-level enrichment tests for one set."""
    sig = q < ALPHA
    n_pair = sig.sum()
    parcels_sig = np.unique(np.where(sig)[0])
    in_set = np.array([i in parcel_set for i in range(N_PARCELS)])

    # Pair-level enrichment over parcel-window pairs (pseudo-replicated, see the
    # module docstring; reported for completeness only).
    sig_in = sig[in_set].sum()
    sig_out = sig[~in_set].sum()
    ns_in = (~sig[in_set]).sum()
    ns_out = (~sig[~in_set]).sum()
    table = [[sig_in, sig_out], [ns_in, ns_out]]
    odds, p_over = stats.fisher_exact(table, alternative='greater')
    _, p_under = stats.fisher_exact(table, alternative='less')

    expected_pairs = n_pair * len(parcel_set) / N_PARCELS
    share = 100 * len(parcel_set) / N_PARCELS
    n_in_sig = sum(1 for i in parcels_sig if i in parcel_set)

    print(f'\n  {label}: {len(parcel_set)}/{N_PARCELS} parcels ({share:.1f}% of the atlas)')
    print(f'    significant parcel-window pairs inside the set : {sig_in}'
          f'  (expected {expected_pairs:.1f} if uniform)')
    print(f'    significant pairs outside                      : {sig_out}')
    print(f'    odds ratio {odds:.3f}   enrichment p = {p_over:.4f}'
          f'   depletion p = {p_under:.4f}')
    print(f'    distinct significant parcels inside the set    : {n_in_sig} / {len(parcels_sig)}')

    # Parcel-level test: each parcel counted once, exact hypergeometric.
    n_sig = len(parcels_sig)
    expected_parcels = n_sig * len(parcel_set) / N_PARCELS
    p_parcel_under = stats.hypergeom.cdf(n_in_sig, N_PARCELS, len(parcel_set), n_sig)
    p_parcel_over = stats.hypergeom.sf(n_in_sig - 1, N_PARCELS, len(parcel_set), n_sig)
    print(f'    PARCEL-LEVEL (primary test)                    : '
          f'{n_in_sig} / {n_sig} in set, expected {expected_parcels:.1f}')
    print(f'      hypergeometric enrichment p = {p_parcel_over:.4f}   '
          f'depletion p = {p_parcel_under:.4f}')


def main() -> None:
    cfg = set_paths()
    label_file = resolve('ROI_LABEL_FILE', REPO / 'config' / 'EEG_ROI_LABELS.csv')
    maps = [
        ('theta (Fig. 3)', resolve('RMANOVA_THETA', cfg.out / 's1_rmanova_theta.mat')),
        ('broadband (Fig. 2b)',
         resolve('RMANOVA_BROADBAND', cfg.out / 'c3_2_rmanova_results.mat')),
    ]

    names = load_labels(label_file)
    core = build_set(names, OCULAR_CORE)
    broad = sorted(set(core) | set(build_set(names, FRONTAL_EXTRA)))
    print(f'ocular-proximal CORE  : {len(core)} parcels')
    print('   ' + ', '.join(names[i] for i in core))
    print(f'ocular-proximal BROAD : {len(broad)} parcels (core + all remaining frontal)')

    for tag, path in maps:
        with h5py.File(path, 'r') as f:
            q = np.array(f['q_values']).T
        assert q.shape == (N_PARCELS, 30), q.shape
        print(f'\n{"=" * 74}\n{tag}   [{path.name}]   '
              f'{int((q < ALPHA).sum())} FDR-significant pairs')
        report(q, core, 'CORE (orbital + frontopolar)')
        report(q, broad, 'BROAD (core + all frontal)')

        # Where the significant parcels actually are.
        parcels_sig = np.unique(np.where(q < ALPHA)[0])
        counts = [(names[i], int((q[i] < ALPHA).sum())) for i in parcels_sig]
        counts.sort(key=lambda t: -t[1])
        print('\n    top significant parcels (by number of significant windows):')
        for name, k in counts[:8]:
            print(f'      {k:2d} windows  {name}')


if __name__ == '__main__':
    main()
