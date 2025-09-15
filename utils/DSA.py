import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
import os
import pydicom


def preprocess_image(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".png":
        imgio = NaturalImage2DIO()
        img, props = imgio.read_images((input_path,))
        return img[0]
    elif ext == ".dcm":
        ds = pydicom.dcmread(input_path)
        img = ds.pixel_array
        return np.min(img, axis=0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def run_inference_DSA(
    image: str,
    target_size: tuple[int, int] = (512, 512),
    predictor: nnUNetPredictor = None,
) -> np.ndarray:
    if isinstance(image, str):
        image = preprocess_image(image)

    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    img_torch = (
        torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, H, W)

    # Z-score normalization (per image)
    mean = img_torch.mean()
    std = img_torch.std()
    img_torch = (img_torch - mean) / (std + 1e-8)

    # Predict logits; expected shape: (num_classes, H, W)
    pred = predictor.predict_logits_from_preprocessed_data(img_torch)[0]

    prob_map = torch.sigmoid(pred[0:1])  # keep channel dim: (1, H, W)

    mask = (prob_map.squeeze(0) < 0.5).cpu().numpy().astype(np.uint8) * 255

    return mask
