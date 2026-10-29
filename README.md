# RCSGF

Selected implementation materials for **Reciprocal-Space Continuity-Guided Self-Supervised Gap Filling (RCSGF)** for synchrotron SAXS/WAXD/USAXS detector-gap filling.

RCSGF constructs self-supervised training pairs directly from experimentally observed regions by introducing artificial masks only within valid observations and using the corresponding pre-mask intensities as proxy supervision. The gap-filling network is based on the Two-Stage Inpainting Network (TSINet).

## Repository Structure

```text
RCSGF-main/
├── README.md
├── requirements.txt
├── generate_mask10_general.py
├── get_gap.py
└── Network/
    ├── TSINet.py
    ├── FPMAF.py
    ├── Gated.py
    ├── FCR_demo.ipynb
    ├── SCAM_demo.ipynb
    └── XBSAM_demo.ipynb
```

## Scope of This Release

This repository is a **selected methodological code release** accompanying the RCSGF study. It includes:

- the generalized Mask 10 generator used to construct the Input–PGT self-supervised proxy task;
- detector-gap mask utilities and the SGALoss implementation in `get_gap.py`;
- source files showing the TSINet, FPMAF, and gated-convolution architecture;
- selected implementation excerpts for the FCR, SCAM, and XBSAM components in Jupyter Notebook format.

`FCR_demo.ipynb`, `SCAM_demo.ipynb`, and `XBSAM_demo.ipynb` are intended to present selected code corresponding to the principal mechanisms described in the associated TSINet paper and Supporting Information. They are **not complete standalone module implementations** and are not intended to be imported as drop-in replacements for the original Python modules.

Accordingly, this repository is not a complete end-to-end training or reproduction package. The full experiment-management pipeline, pretrained weights, and all scripts used for training, validation, metric calculation, radial integration, and figure generation are not included.

## Main Files

### `generate_mask10_general.py`

Generalized implementation of the artificial masking configuration referred to as **Mask 10** in the RCSGF study.

The script:

- accepts two-dimensional TIFF images of arbitrary size;
- detects horizontal and vertical detector gaps from the input image rather than assuming a fixed number of gaps;
- measures detector-gap widths from each image;
- generates gap-adjacent strip masks according to the detected gap geometry;
- adds 1 × 1-pixel point masks within the remaining valid candidate regions;
- excludes physical detector-gap pixels from artificial masking;
- preserves the original TIFF image size and dtype;
- supports reproducible per-image randomization, run manifests, resume mode, and optional mask export.

### `get_gap.py`

Provides detector-gap mask utilities for two-dimensional images and batches, together with an SGALoss implementation.

### `Network/TSINet.py`

Provides the two-stage TSINet architecture used as the gap-filling network in RCSGF.

### `Network/FPMAF.py`

Provides the Feature Pyramid Mask-Aware Fusion (FPMAF) architecture used in the TSINet refinement stage.

### `Network/Gated.py`

Provides the gated-convolution layer used in TSINet.

### `Network/FCR_demo.ipynb`

Selected implementation excerpts illustrating the main components of the **Feature Correlation Refiner (FCR)**, including Q/K/V projection, local-window processing, spatial restoration, and gated feature fusion.

### `Network/SCAM_demo.ipynb`

Selected implementation excerpts illustrating the principal **Semantic Coherence Attention Module (SCAM)** operations, including known/unknown patch partitioning, cosine-similarity matching, Top-K retrieval, and softmax-weighted fusion.

### `Network/XBSAM_demo.ipynb`

Selected implementation excerpts illustrating the principal **X-Ray Based Self-Attention Module (XBSAM)** operations, including gradient-guided feature modulation, predefined periodic-pattern kernels, SimAM, patch-wise similarity responses, and feature mapping.

## Requirements

The released code is written in Python and PyTorch. The environment information used for the released source files is recorded in `requirements.txt`.

For `generate_mask10_general.py`, ensure that the following packages are available:

```text
numpy
tifffile
tqdm
```

For the network-related source files, PyTorch is required. Other dependencies used by the released files are listed in `requirements.txt`.

## Input Data and Mask Convention

The current Mask 10 generator operates on single-channel two-dimensional TIFF patterns.

The input to `generate_mask10_general.py` should be a **PGT image that already contains the detector gaps**. The physical detector-gap pixels are retained as invalid regions and are never selected as artificial-mask pixels.

For network masks:

```text
mask = 1  -> artificial masked region / position to be predicted
mask = 0  -> observed region
```

During self-supervised training, artificial masks are introduced only within valid observed regions. The corresponding pre-mask PGT intensities therefore remain available as proxy supervision, whereas the physical detector-gap regions themselves do not provide training targets.

## Generating Mask 10

The generalized generator provides the four dataset presets used in the RCSGF study:

```text
SAXS:    0.65–0.70
WAXD:    0.75–0.80
AgBiO3:  0.45–0.55
USAXS:   0.65–0.70
```

Example:

```bash
python generate_mask10_general.py \
    --input-dir /path/to/PGT \
    --output-dir /path/to/Input \
    --preset saxs
```

The `--preset` option can be set to `saxs`, `waxd`, `agbio3`, or `usaxs` for the corresponding dataset.

Run the following command to view all available options:

```bash
python generate_mask10_general.py --help
```

The generator writes the masked TIFF images to the selected output directory and records the generation parameters and mask statistics in a CSV manifest.

## Dataset Organization in the RCSGF Study

For the SAXS, WAXD, and AgBiO3 datasets, complete experimental patterns were used to construct PGT images containing synthetic detector gaps. Mask 10 was then applied only to the valid observed regions of these PGT images to produce the self-supervised Inputs.

For the USAXS dataset, the experimentally acquired patterns already contain physical detector-module gaps and are therefore used directly as PGT images before artificial masking.

## Public Datasets

The SAXS, WAXD, and AgBiO3 datasets are publicly available from the Science Data Bank:

**DOI:** https://doi.org/10.57760/sciencedb.34110

The USAXS dataset is publicly available from the Science Data Bank:

**DOI:** https://doi.org/10.57760/sciencedb.017rq

## Associated Publications

### RCSGF

**Self-Supervised Gap Filling of Synchrotron SAXS/WAXD Data Based on Reciprocal-Space Continuity**

Bibliographic information and the final DOI can be added after publication.

### TSINet

The TSINet architecture and the FCR, FPMAF, XBSAM, SCAM, and SGALoss components were originally described in:

**A deep learning approach to gap filling for synchrotron SAXS and WAXD patterns**  
*Journal of Applied Crystallography* (2026), 59.  
**DOI:** https://doi.org/10.1107/S1600576726009350

## Citation

If you use the RCSGF methodology, Mask 10 generator, or the released implementation materials in academic work, please cite the associated RCSGF paper once its final bibliographic information is available. When using or discussing the TSINet architecture and its component modules, please also cite the corresponding *Journal of Applied Crystallography* article listed above.

## Contact

For questions regarding the released implementation materials or associated datasets, please contact the authors of the corresponding publication.
