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

## Datasets

The released models were trained on publicly available datasets - please cite these works if you use the pretrained models:

| Dataset | Modality          | License / Access | Citation                                                                                                                                                     |
| ------- | ----------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DSCA    | 2D DSA (cerebral) | CC BY 4.0        | [Zhang et al., 2025](https://doi.org/10.5281/zenodo.11255024)                                                                                                |
| XCAD    | 2D DSA (coronary) | No license found | [Ma et al., 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Ma_Self-Supervised_Vessel_Segmentation_via_Adversarial_Learning_ICCV_2021_paper.pdf) |
| FS-CAD  | 2D DSA (coronary) | No license found | [Zeng et al., 2024](https://www.nature.com/articles/s41598-024-71063-5#data-availability)                                                                    |

## Citation

nnUNetTrainer and loss function implementations are from [cbDice](https://github.com/PengchengShi1220/cbDice) which is licensed under the Apache License 2.0.

If you use this code or the pretrained models in your research, please cite:

```
@inproceedings{shi2024centerline,
  title={Centerline Boundary Dice Loss for Vascular Segmentation},
  author={Shi, Pengcheng and Hu, Jiesi and Yang, Yanwu and Gao, Zilve and Liu, Wei and Ma, Ting},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={46--56},
  year={2024},
  organization={Springer}
}
```

and

```
citation placeholder
```
