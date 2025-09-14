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

4. Run inference (DICOM or PNG input)

```
python run_inference.py \
  -i dicom_data/1_SMG/Post/DSA.dcm \
  -o outputs \
  -m nnUNet_results/Dataset113_XFSCAD/nnUNetTrainer_CE_DC_CBDC__nnUNetPlans__2d \
  -f 0
```

- The script automatically uses CUDA or Apple MPS if available.
- nnUNetTrainer and loss function implementations are from [cbDice Paper Implementation](https://github.com/PengchengShi1220/cbDice)
- Model weights can be found on [Google Drive](https://drive.google.com/drive/folders/1ZlnhJurHPzOPndgY1RdaRFrTJHGgjoYY?usp=sharing)

## Citation

Please cite:

```
citation placeholder
```
