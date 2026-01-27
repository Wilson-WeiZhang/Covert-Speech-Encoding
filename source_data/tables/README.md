# Source Data Tables

Machine-learning friendly format: **each row = one sample, each column = one feature**

Generated: 2025-01-27

---

## Figure 2: Overt Speech Timing

### `Fig2_OnsetOffset_BySubjectWord.csv` (265 rows)
| Column | Description |
|--------|-------------|
| subject_idx | Subject index (1-53) |
| word_id | Word ID (1-5) |
| onset_ms | Speech onset time (ms) |
| offset_ms | Speech offset time (ms) |
| duration_ms | Speech duration (ms) |

**Reproducible results**: onset 594±117ms, offset 1230±164ms, offset F(4,260)=79.89

---

## Figure 3: F-test Statistics (Theta Band)

### `Fig3_Fvalues_ROIxWindow.csv` (148 rows x 31 cols)
- **Rows**: 148 ROIs (Destrieux atlas)
- **Columns**: ROI_idx + 30 time windows (win01_0-50ms ... win30_1450-1500ms)
- **Values**: F-statistic from rmANOVA (5 words)
- **Band**: Theta (4-8 Hz)

### `Fig3_Pvalues_ROIxWindow.csv` (148 rows x 31 cols)
- Same structure as F-values
- **Values**: Uncorrected p-values

### `Fig3_Qvalues_FDR_ROIxWindow.csv` (148 rows x 31 cols)
- Same structure as F-values
- **Values**: FDR-corrected q-values (Benjamini-Hochberg)

**Reproducible results**: 114 FDR-significant pairs, peak F=12.25 at 250-300ms

---

## Figure 4: LME Variance Decomposition

### `Fig4_LME_R2_ByPair.csv` (348 rows)
| Column | Description |
|--------|-------------|
| pair_idx | ROI-window pair index |
| ROI_idx | ROI index (1-148) |
| window_idx | Window index (1-12) |
| window_start_ms | Window start (ms) |
| window_end_ms | Window end (ms) |
| R2_marginal_fixed | R² (Fixed effects only) |
| R2_cond_RI | R² (Random Intercept model) |
| R2_cond_RS | R² (Random Slope model) |
| Delta_RI | R² increase from Fixed to RI |
| Delta_RS | R² increase from RI to RS |
| best_model | Best model by AIC |

**Reproducible results**: R²_fixed=0.28%, Δ_RI=7.56%, Δ_RS=9.72%, ratio=62x

### Classification Feature Data (Large Files)
> **Note**: Classification feature matrices (5700 trials × 22200 features) are too large for this repository.
> Available at: https://github.com/Wilson-WeiZhang/Covert-Speech-Encoding

---

## Figure 5: fMRI-EEG RSA

### `Fig5_RSA_BySubjectCluster.csv` (294 rows)
| Column | Description |
|--------|-------------|
| subject_idx | Subject index (1-49) |
| stage | prep (0-600ms) or exec (600-1200ms) |
| stage_ms | Time window in milliseconds |
| cluster_id | 1=Left_SM, 2=Right_SM, 3=Occipital |
| cluster_name | Cluster name |
| rsa_r | Subject-level RSA correlation |

**Reproducible results**:
| Stage | Cluster | mean_r | Paper |
|-------|---------|--------|-------|
| prep | Left_SM | 0.094 | ✓ matches |
| prep | Right_SM | 0.079 | ✓ matches |
| prep | Occipital | 0.031 | ✓ matches |
| exec | Left_SM | 0.041 | ✓ matches |
| exec | Right_SM | -0.029 | ✓ matches |
| exec | Occipital | -0.079 | ✓ matches |

> **Note**: p-values from permutation test (1000 iterations) - Prep Left_SM p=0.043*

---

## Figure 6: Connectivity

### `Fig6_wPLI_BySubjectWord.csv` (285 rows)
| Column | Description |
|--------|-------------|
| subject_idx | Subject index (1-57) |
| word_id | Word ID (1-5) |
| mean_wpli_prep | Mean wPLI (prep stage) |

**Reproducible results**: Word effect F(4,280)=5.02, p=0.0006

### `Fig6_NodeStrength_ByROI.csv` (148 rows)
| Column | Description |
|--------|-------------|
| ROI_idx | ROI index (1-148) |
| mean_node_strength_prep | Mean node strength (prep stage, Delta band) |

**Reproducible results**: ROI 55 (G_postcentral L) strength = 0.652

### `Fig6_dPLI_Direction_ByROI.csv` (34 rows)
| Column | Description |
|--------|-------------|
| ROI_idx | ROI index of connected node |
| stage | prep or exec |
| mean_dpli | Mean dPLI value (>0.5 = outflow from ROI 55) |
| tstat | t-statistic vs 0.5 |
| p_value | Uncorrected p-value |
| q_value_fdr | FDR-corrected q-value |

- **ROI 55** = G_postcentral L (Left Postcentral Gyrus)
- **Prep stage**: 30 connections, ROI 37 significant outflow (q=0.014)
- **Exec stage**: 4 connections

---

## Notes

1. **ROI Index**: 1-148 corresponds to Destrieux atlas (74 left + 74 right hemisphere)
2. **Time Windows**: 50ms bins, 0-1500ms range
3. **Word IDs**: 1-5 corresponding to the 5 command phrases
4. **Baseline**: All neural data baseline-corrected using -500 to 0 ms pre-stimulus
5. **Frequency Bands**: Figure 3 uses theta band (4-8 Hz); Figure 6 uses delta band (1-4 Hz)

## Usage Example (Python)

```python
import pandas as pd

# Load LME R² data
lme = pd.read_csv('Fig4_LME_R2_ByPair.csv')
print(f"R²_fixed = {lme['R2_marginal_fixed'].mean()*100:.2f}%")
print(f"Δ_RI = {lme['Delta_RI'].mean()*100:.2f}%")
print(f"Δ_RS = {lme['Delta_RS'].mean()*100:.2f}%")

# Load RSA data
rsa = pd.read_csv('Fig5_RSA_ByCluster.csv')
prep_left = rsa[(rsa['stage'] == 'prep') & (rsa['cluster_id'] == 1)]
print(f"Left SM: r={prep_left['mean_r'].values[0]:.3f}, p={prep_left['p_perm'].values[0]:.3f}")
```

---

## Data Provenance

| Table | Source MAT File | Processing Script |
|-------|-----------------|-------------------|
| Fig2_OnsetOffset | `READAUDIO/subject_stats2.mat` | `s7_2_speech_onset_offset.m` |
| Fig3_Fvalues/Pvalues/Qvalues | `s5_2_rmanova_theta.mat` | `s5_2_rmanova_per_band.m` |
| Fig4_LME_R2 | `lme_results_rmanova.mat` | `c4_3b_lme_fit_models_rmanova.m` |
| Fig5_RSA | `fmri_eeg_rsa/fig5_rsa_data_v2.mat` | `c5_5_rsa_correlation_permutation.m` |
| Fig6_wPLI | `c6_connectivity_results/roi55_wpli_by_word.mat` | `c6_84_wpli_twoperiod_compute.m` |
| Fig6_NodeStrength | `c6_connectivity_results/wpli_rmanova_raw.mat` | `c6_8c_node_strength_ftest.m` |
| Fig6_dPLI | `c6_connectivity_results/roi55_dpli_direction_fdr.mat` | `c6_17_dpli_direction_fdr.m` |
