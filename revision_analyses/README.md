# Revision-stage analyses

Analysis code for the revision of *Spatiotemporal Encoding Signatures of Covert
Speech Revealed by Source-localized EEG and fMRI* (in revision). This
repository section contains the analyses added during revision; the primary analysis pipeline is in the main repository
([Covert-Speech-Encoding](https://github.com/Wilson-WeiZhang/Covert-Speech-Encoding)).

## Module index

| Module | Analysis | Supplementary Note |
|---|---|---|
| `01_ocular_component_sets` | Construction of the ocular component sets (manual and union) and their scalp topographies | S8, S12 |
| `02_source_export` | Source reconstruction with artefact components reinstated (add-back datasets) | S12 |
| `03_pipeline_verification` | Verification that the reconstruction pipeline reproduces the published source data and classifier output | S12 |
| `04_classification_variants` | Phrase classification across the six component-set variants | S12 |
| `05_empirical_chance` | Permutation-based empirical chance level for the phrase classifier | S9 |
| `06_artefact_decoding` | Phrase decoding from removed artefact components; cohort-level null | S12 |
| `07_visual_roi_removal` | Phrase classification after removal of visual and occipitotemporal ROIs | S11 |
| `08_ocular_spatial_control` | Spatial distribution of the ocular add-back effect | S12 |
| `09_stimulus_attribute_rsa` | Representational similarity analysis of stimulus attributes | S10 |
| `10_overt_arm` | Overt-speech arm: preprocessing, source export, classification, spatiotemporal mapping, muscle/ocular component time courses | S13 |
| `11_acoustic_monitoring` | Acoustic level monitoring of covert and overt blocks | S14 |

## Requirements

- MATLAB R2023a or later with EEGLAB 2024.2 and Brainstorm
- Python 3.12 with `numpy`, `scipy`, `pandas`, `matplotlib`, `h5py`

## Configuration

Copy `config/config.example.json` to `config/config.json` and set the data
locations. All scripts resolve paths through `config/set_paths.m` (MATLAB) or
`config/set_paths.py` (Python); no paths are hard-coded in the analysis
scripts. Outputs are written under `results/`.

EEG and audio data are not part of this repository; see the Data Availability
statement of the paper for the data repository.

## Running order

Within each module, scripts are numbered in execution order (`sNN_`).
Cross-module dependencies: `01` before `02`; `02` before `03` and `04`;
`10_overt_arm/preprocessing` before the other `10_overt_arm` scripts.

## License

CC BY-NC 4.0, as for the main repository.
