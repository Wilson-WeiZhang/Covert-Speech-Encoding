# Covert Speech EEG-fMRI Analysis

This repository contains the analysis code for the research paper:

**Mapping the Spatiotemporal Encoding Signatures of Covert Speech with Source-based EEG-fMRI**

Wei Zhang<sup>1,2</sup>, Muyun Jiang<sup>1</sup>, Kok Ann Colin Teo<sup>2,3,4,5</sup>, Raghavan Bhuvanakantham<sup>2</sup>, Zhiwei Guo<sup>1</sup>, Shuailei Zhang<sup>1</sup>, Chuan Huat Vince Foo<sup>6</sup>, Victoria Leong<sup>7</sup>, Jia Lu<sup>6</sup>, Balazs Gulyas<sup>2,8,9</sup>, & Cuntai Guan<sup>1,10</sup>

<sup>1</sup> College of Computing and Data Science, Nanyang Technological University, Singapore
<sup>2</sup> Cognitive Neuroimaging Centre, Nanyang Technological University, Singapore
<sup>3</sup> Lee Kong Chian School of Medicine, Nanyang Technological University, Singapore
<sup>4</sup> IGP-Neuroscience, Interdisciplinary Graduate Programme, Nanyang Technological University, Singapore
<sup>5</sup> Division of Neurosurgery, National University Health System, Singapore
<sup>6</sup> DSO National Laboratories, Singapore
<sup>7</sup> Early Mental Potential and Wellbeing Research (EMPOWER) Centre, Nanyang Technological University, Singapore
<sup>8</sup> Hungarian Research Network (HUN-REN), Hungary
<sup>9</sup> Department of Clinical Neuroscience, Karolinska Institutet, Stockholm, Sweden
<sup>10</sup> Center for AI in Medicine (C-AIM), Nanyang Technological University, Singapore

## Overview

This repository contains the analytical code used to investigate the spatiotemporal neural encoding of covert (inner) speech. The analysis combines source-localized EEG with independent fMRI validation to map phrase-discriminative brain activity and individual variability in neural encoding strategies.

## Citation

If you use this code in your research, please cite our paper:

```
Zhang, W., Jiang, M., Teo, K.A.C., Bhuvanakantham, R., Guo, Z., Zhang, S., Foo, C.H.V.,
Leong, V., Lu, J., Gulyas, B., & Guan, C. (2025). Mapping the Spatiotemporal Encoding
Signatures of Covert Speech with Source-based EEG-fMRI.
```

## Data Availability

All datasets have been made publicly available through Nanyang Technological University (NTU)'s data repository (DR-NTU Data https://researchdata.ntu.edu.sg/) and can be accessed according to NTU's open access policy:

1. **Source-localized EEG data**: https://doi.org/[pending] (sLORETA, Destrieux 148 ROIs, N=57)
2. **Preprocessed EEG data**: https://doi.org/[pending] (EEGLAB format, N=57)
3. **fMRI contrast maps**: https://doi.org/[pending] (SPM first-level, N=49)

This code repository demonstrates the analytical methodology used in our study. The scripts are designed to work with the publicly available data listed above.

## Analysis Pipeline

The analysis consists of 30 sequential steps:

### Preprocessing (Steps 01-03)
1. **Step 01**: Raw EEG preprocessing (filtering, resampling, epoching)
2. **Step 02**: ICA decomposition (AMICA algorithm)
3. **Step 03**: Artifact rejection (ICLabel classification)

### Source Localization (Steps 04-05)
4. **Step 04**: sLORETA source reconstruction (Destrieux atlas, 148 ROIs)
5. **Step 05**: Prepare channel-level data for comparison analyses

### Statistical Analysis (Steps 06-10)
6. **Step 06**: F-test for phrase discrimination (ROI × time window)
7. **Step 07**: Repeated measures ANOVA with permutation testing
8. **Step 08**: Generate Figure 3 (statistical evidence)
9. **Step 09**: Frequency band filtering (delta, theta, alpha, beta)
10. **Step 10**: Cross-band comparison analysis

### Classification & LME (Steps 11-15)
11. **Step 11**: 5-class SVM classification (leave-one-block-out)
12. **Step 12**: LME data preparation
13. **Step 13**: LME variance decomposition (Nakagawa & Schielzeth method)
14. **Step 14**: Bootstrap confidence intervals
15. **Step 15**: Generate Figure 4 (individual differences)

### fMRI-EEG Validation (Steps 16-19)
16. **Step 16**: Extract fMRI F-map clusters
17. **Step 17**: Load EEG activity patterns
18. **Step 18**: Calculate representational dissimilarity matrices
19. **Step 19**: RSA permutation test (N=49)

### Connectivity Analysis (Steps 20-25)
20. **Step 20**: wPLI connectivity computation
21. **Step 21**: dPLI directional connectivity
22. **Step 22**: Hub node identification (ROI 55)
23. **Step 23**: Information flow analysis
24. **Step 24**: Duration-connectivity correlation
25. **Step 25**: Generate Figure 6 (connectivity)

### Supplementary Analyses (Steps 26-30)
26. **Step 26**: Generate Supplementary Table S1
27. **Step 27**: Overt speech onset/offset timing
28. **Step 28**: Source localization reproducibility
29. **Step 29**: Leave-one-block-out validation
30. **Step 30**: Channel-level SVM comparison

## Requirements

The code in this repository requires the following major dependencies:
- MATLAB R2024b (https://www.mathworks.com)
- EEGLAB 2024.2.1 (https://eeglab.org/)
- Brainstorm (https://neuroimage.usc.edu/brainstorm/)
- FieldTrip (https://www.fieldtriptoolbox.org/)
- SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)

## Acknowledgements

This research is supported by DSO National Laboratories, Singapore. We thank the Cognitive Neuroimaging Centre, Nanyang Technological University, Singapore for computational resources used in data analysis.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

Copyright (c) 2025 Zhang et al.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the material without restriction, including the rights to use, copy, modify, remix, transform, and build upon the material, provided that:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

NonCommercial — You may not use the material for commercial purposes.

No additional restrictions — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

The full license text is available at: https://creativecommons.org/licenses/by-nc/4.0/

## Contact

For any questions regarding this code repository, please contact:

PI: Prof. Cuntai Guan
Centre for Brain-Computing Research (CBCR)
Nanyang Technological University, Singapore
Email: ctguan@ntu.edu.sg

or the author: Dr. Wei Zhang
College of Computing and Data Science
Nanyang Technological University, Singapore
Email: wilson.zhangwei@ntu.edu.sg
