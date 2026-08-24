RCSGF
Core implementation code for Reciprocal-Space Continuity-Guided Self-Supervised Gap Filling (RCSGF) for synchrotron small-angle X-ray scattering (SAXS) and wide-angle X-ray diffraction (WAXD) patterns.
RCSGF constructs self-supervised training targets from experimentally observed regions and uses the Two-Stage Inpainting Network (TSINet) to predict locally missing intensities caused by detector gaps.
Repository Structure
```text
RCSGF_main/
├── README.md
├── requirements.txt
├── generate_mask10.py
├── get_gap.py
└── Network/
    ├── TSINet.py
    ├── FCR.py
    ├── FPMAF.py
    ├── Gated.py
    ├── SCAM.py
    └── XBSAM.py
```
Main files
`generate_mask10.py`  
Generates the artificial masking configuration referred to as Mask 10 in the associated study.
`get_gap.py`  
Provides detector-gap mask utilities and loss-function implementations, including SGALoss.
`Network/TSINet.py`  
Implements the Two-Stage Inpainting Network (TSINet).
`Network/FCR.py`  
Implements the Feature Correlation Refiner (FCR).
`Network/FPMAF.py`  
Implements the Feature Pyramid Mask-Aware Fusion (FPMAF) module.
`Network/SCAM.py` and `Network/XBSAM.py`  
Implement the two feature-processing branches used in FPMAF.
`Network/Gated.py`  
Implements the gated convolution layer used throughout TSINet.
Scope of This Code Package
This repository provides the core network, mask-generation, detector-gap processing, and loss-function implementations used in RCSGF.
The current public package does not include the complete experiment-management pipeline, pretrained weights, or all scripts used for training, validation, metric calculation, and figure generation. Experimental settings and evaluation procedures are described in the associated paper.
Requirements
The code is implemented in Python using PyTorch. The main third-party packages used by the current source files are:
PyTorch
NumPy
OpenCV
imageio
tqdm
Install the dependencies with:
```bash
pip install -r requirements.txt
```
The versions recorded in `requirements.txt` should correspond to the environment used for the released code.
Input Data
The current implementation is designed for single-channel two-dimensional X-ray scattering/diffraction patterns stored as TIFF images.
The Mask 10 generation script is configured for images of:
```text
512 × 512 pixels
```
Detector-gap rows and columns are identified from invalid-value regions according to the criteria implemented in `generate_gap_mask_2d()`. If the intensity encoding or preprocessing of another dataset differs, the gap-detection criteria should be adjusted accordingly.
Artificial Mask Convention
For network input masks:
```text
mask = 1  -> position to be predicted / masked region
mask = 0  -> observed region
```
During self-supervised training, artificial masking is applied only to valid observed regions so that the corresponding pre-mask measurements remain available as proxy supervision.
Generating Mask 10
`generate_mask10.py` constructs Mask 10 using fixed strip-shaped masks together with discrete 1 × 1-pixel masks near the detector gaps.
Before running the script, set:
```python
input_dir = r'PATH_TO_INPUT_TIFFS'
output_dir = r'PATH_TO_OUTPUT_TIFFS'
```
The target coverage ratio of the discrete masks is controlled by:
```python
TARGET_RATIO = (lower_bound, upper_bound)
```
The settings used for the three data types in the associated study are:
```text
SAXS:    0.65–0.70
WAXD:    0.75–0.80
AgBiO3:  0.45–0.55
```
The current script should therefore be configured with the corresponding `TARGET_RATIO` before generating masks for each data type.
Run:
```bash
python generate_mask10.py
```
The processed TIFF images will be written to `output_dir`.
Using TSINet
A minimal network initialization is:
```python
from Network.TSINet import TSINet

model = TSINet(
    img_ch=1,
    output_ch=1,
    cnum=48
)
```
The forward call is:
```python
stage1_out, stage2_out = model(images, masks)
```
where:
```text
images : [B, 1, H, W]
masks  : [B, 1, H, W]
```
`stage1_out` and `stage2_out` are the outputs of the coarse-restoration and refinement stages, respectively.
SGALoss
`SGALoss` is implemented in `get_gap.py` and applies region-weighted losses to the outputs of both TSINet stages. For the self-supervised task, only positions for which observed PGT values are available are used as supervision; detector-gap positions themselves do not provide training targets.
The exact loss weights and experimental settings should be chosen consistently with the associated paper.
Associated Dataset
The SAXS/WAXD dataset used in the study is publicly available from Science Data Bank:
DOI: https://doi.org/10.57760/sciencedb.34110
Associated Publication
Self-Supervised Gap Filling of Synchrotron SAXS/WAXD Data Based on Reciprocal-Space Continuity
Publication information and the final article DOI can be added here after publication.
Citation
If this code is used in academic work, please cite the associated RCSGF paper and the corresponding Science Data Bank code record after their bibliographic information becomes available.
Contact
For questions regarding the implementation or associated data, please contact the authors of the corresponding publication.
