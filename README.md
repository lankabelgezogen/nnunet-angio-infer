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

- png, dicom (DSA), and nifti (MRA, CTA) file formats are currently supported
- The script automatically uses CUDA or Apple MPS if available.
- You can input a folder for batch inferencing (same format) or single images. No special naming is needed compared to nnUNet's built-in inferencing.
- You can add your custom trainers + loss functions by simply pasting them into the respective directories. We monkeypatch nnUNet's class finder to include them without any manual environment configurations.

## Model Weights

- [DSA Google Drive](https://drive.google.com/drive/folders/1ZlnhJurHPzOPndgY1RdaRFrTJHGgjoYY?usp=sharing)
- [MRA Google Drive]()
- [CTA Google Drive]()

## Citation

nnUNetTrainer and loss function implementations are from the [cbDice Paper Implementation](https://github.com/PengchengShi1220/cbDice) which is licensed under the Apache License 2.0.

If you use our code in your research, please cite:

```
citation placeholder
```
