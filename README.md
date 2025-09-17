<div align="center">
  <p align="center">
    <h1 align="center">Multi-Modal Angiography Segmentation with nnU-Net</h1>
  </p>

  <p align="center">
    <img src="assets/segmentation_dsa.png" alt="Coronary DSA segmentation" width="30%">
    &nbsp;
    <img src="assets/segmentation_gae.png" alt="Cerebral DSA segmentation" width="30%">
    &nbsp;
    <img src="assets/segmentation_mra.png" alt="MRA segmentation" width="30%">
    <br>
    <em>Example vessel segmentations: cerebral DSA, genicular DSA, and MRA.</em>
  </p>
</div>

---

**Multi-Modal Angiography Segmentation** provides a single, lightweight Python interface for running nnU-Net v2 inference across different vascular imaging modalities - from 2D DSA sequences and MIPs to full 3D MRA and CTA volumes. We also provide model weights for these modalities.

- Unified CLI for DSA, MRA, and CTA
- Automatic CUDA / Apple MPS detection
- Plug-and-play custom trainers and loss functions
- png, dicom (only DSA), and nifti (MRA, CTA, DSA) input file formats are currently supported
- You can input a folder for batch inferencing (same format) or single images. No special naming is needed compared to nnUNet's built-in inferencing

---

## Quickstart

Note: The code was tested on macOS and Linux using Python 3.11. Other versions of Python may lead to dependency or compatibility issues.

1. Install PyTorch

```
pip install torch
```

2. Install project requirements

```
pip install -r requirements.txt
```

3. Set your nnUNet_results env variable

```
export nnUNet_results="/path/to/nnUNet_results"
```

4. Run inference (-m allows for modality selection (DSA, MRA, CTA))

```
python run_inference.py -i dicom_data/1_SMG/Post/DSA.dcm -o outputs -m DSA -md nnUNet_results/Dataset113_XFSCAD/nnUNetTrainer_CE_DC_CBDC__nnUNetPlans__2d -f 0
```

## Model Weights

- [Coronary DSA Google Drive](https://drive.google.com/drive/folders/1RkPjdNm0_bmUbHNVJUFkoI8q8nHvxhKo?usp=sharing)
- [Cerebral DSA Google Drive](https://drive.google.com/drive/folders/1KJj5i3SDC9vjTS98Wjnhgk7m7NO5EqpV?usp=sharing)
- [MRA Google Drive]()
- [CTA Google Drive]()

## Citation

nnUNetTrainer and loss function implementations are from the [cbDice Paper Implementation](https://github.com/PengchengShi1220/cbDice) which is licensed under the Apache License 2.0.

If you use our code in your research, please cite:

```
citation placeholder
```
