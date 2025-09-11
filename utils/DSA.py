import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
import pydicom


def preprocess_image(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".png":
        img = Image.open(input_path)
        img = img.convert("L")
        img = np.array(img)
        return img
    elif ext == ".dcm":
        ds = pydicom.dcmread(input_path)
        img = ds.pixel_array
        minip = np.min(img, axis=0)
        return minip
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def run_inference_DSA(
    image_np: np.ndarray | str,
    target_size: tuple[int, int] = (512, 512),
    predictor: nnUNetPredictor = None,
) -> np.ndarray:
    if isinstance(image_np, str):
        image_np = preprocess_image(image_np)

    if image_np.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image_np.shape}")

    original_h, original_w = image_np.shape

    # Resize to training size
    img_torch = (
        torch.from_numpy(image_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, H, W)
    img_torch = F.interpolate(
        img_torch, size=target_size, mode="bilinear", align_corners=False
    )

    # Z-score normalization (per image)
    mean = img_torch.mean()
    std = img_torch.std()
    img_torch = (img_torch - mean) / (std + 1e-8)

    # Predict logits; expected shape: (num_classes, H, W)
    pred = predictor.predict_logits_from_preprocessed_data(img_torch)[0]

    num_classes = pred.shape[0]

    if num_classes == 1:
        prob_map = torch.sigmoid(pred[0:1])  # keep channel dim: (1, H, W)
    else:
        probs = torch.softmax(pred, dim=0)  # (C, H, W)
        if num_classes == 2:
            prob_map = probs[1:2]  # (1, H, W)
        else:
            prob_map = 1.0 - probs[0:1]  # (1, H, W)

    if (target_size[0], target_size[1]) != (original_h, original_w):
        prob_map = F.interpolate(
            prob_map.unsqueeze(0),  # (1, 1, H, W)
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    mask = (prob_map.squeeze(0) < 0.5).cpu().numpy().astype(np.uint8) * 255

    return mask
