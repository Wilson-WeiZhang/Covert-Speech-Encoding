# Source data

`source_data.xlsx` is identical to Supplementary Data 1 of the article: the numerical values
underlying the graphs and charts in Figures 2 to 6, one worksheet per figure panel or analysis,
at the participant or trial level where applicable. Participants are identified by anonymous
index only.

| Worksheet | Figure | Content |
|---|---|---|
| `Fig2_Raw_OnsetOffset` | 2a | Overt speech onset, offset and duration (ms) per participant and phrase |
| `Fig2_Summary` | 2a | Group mean and SD of onset and offset, with the repeated-measures ANOVA across phrases |
| `Fig2b_Raw_Timecourse` | 2b | Group-mean z-scored source activity of the left postcentral gyrus per phrase and time point, with s.e.m. |
| `Fig3_Raw_Fvalues`, `Fig3_Raw_Pvalues`, `Fig3_Raw_Qvalues_FDR` | 3a | Repeated-measures ANOVA across phrases for each of 148 ROIs and 30 windows of 50 ms (theta band): F, uncorrected p and Benjamini-Hochberg q |
| `Fig3_Summary` | 3 | Number of tests, significant ROI-window pairs and distinct ROIs |
| `Fig3b_HMP_ByWindow` | 3b | Harmonic mean p value per time window |
| `Fig4a_Confusion_Matrix` | 4a | Five-class confusion matrix of the phrase classifier |
| `Fig4_Raw_Classification`, `Fig4_Summary_Classification` | 4b | Classification accuracy per participant and its group summary |
| `Fig4b_Classification_Null` | 4b | Permutation null of the classification accuracy per participant |
| `Fig4_Raw_LME`, `Fig4_Summary_LME_Variance` | 4d, 4e | Marginal and conditional R2 of the linear mixed-effects models per ROI-window pair, and their summary |
| `Fig5_Raw_RSA`, `Fig5_Summary_RSA` | 5c-e | EEG-fMRI representational similarity (Spearman rho) per participant, cluster and period, and the permutation test per cluster |
| `Fig6_Raw_NodeStrength`, `Fig6_Summary_Hub` | 6b | Repeated-measures ANOVA of node strength across phrases per ROI and frequency band, and the summary for the left postcentral gyrus |
| `Fig6c_NodeStrength_ByPhrase` | 6c | Node strength of the left postcentral gyrus per participant and phrase |
| `Fig6_Raw_Connections` | 6d | Repeated-measures ANOVA across phrases for each connection of the left postcentral gyrus |
| `Fig6_Raw_dPLI`, `Fig6_Summary_dPLI` | 6e, 6f | Directed phase lag index of the connections of the left postcentral gyrus |
| `Fig6_Raw_DurationCorr`, `Fig6_Summary_Duration` | 6g, 6h | Spoken duration and node strength per participant and phrase, and the correlation summary per period |
| `Sup_Classification` | Supplementary | Accuracy of the alternative classification schemes |

The raw recordings are available under controlled access; see the Data Availability section of
the repository README.
