import numpy as np
import torch
from PIL import Image
import pydicom
import os

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


def preprocess_image(input_path: str) -> np.ndarray:
    if input_path.endswith(".png"):
        imgio = NaturalImage2DIO()
        img, props = imgio.read_images((input_path,))
        print(type(img[0]))
        return img[0]
    elif input_path.endswith(".dcm"):
        ds = pydicom.dcmread(input_path)
        img = ds.pixel_array
        mip = np.min(img, axis=0)
        return np.expand_dims(mip, axis=0)
    elif input_path.endswith(".nii.gz"):
        imgio = SimpleITKIO()
        img, props = imgio.read_images((input_path,))
        return img[0]
    else:
        raise ValueError(f"Unsupported file type: {input_path}")


def run_DSA_inference_on_image(
    image_path: str,
    predictor: nnUNetPredictor = None,
    output_path: str = None,
    output_binary: bool = False,
) -> None:
    image = preprocess_image(image_path)

    img_torch = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # (1, 1, H, W)

    # Z-score normalization (per image)
    mean = img_torch.mean()
    std = img_torch.std()
    img_torch = (img_torch - mean) / (std + 1e-8)

    # Predict logits; expected shape: (num_classes, H, W)
    pred = predictor.predict_logits_from_preprocessed_data(img_torch)[0]

    prob_map = torch.sigmoid(pred[0:1])  # keep channel dim: (1, H, W)

    mask = (prob_map.squeeze(0) < 0.5).cpu().numpy().astype(np.uint8)
    if not output_binary:
        mask = mask * 255

    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(
        output_path, f"{image_path.split('/')[-1].split('.')[0]}.png"
    )

    Image.fromarray(mask).save(output_path)


def run_DSA_inference_on_folder(
    folder: str,
    predictor: nnUNetPredictor = None,
    output_path: str = None,
    output_binary: bool = False,
) -> None:
    first_file = os.listdir(folder)[0]

    if first_file.endswith(".dcm"):
        for file in os.listdir(folder):
            run_DSA_inference_on_image(
                os.path.join(folder, file), predictor, output_path, output_binary
            )
    else:
        predictor.predict_from_files(folder, output_path, output_binary=output_binary)


def run_DSA_inference(
    image_path_or_folder: str,
    predictor: nnUNetPredictor = None,
    output_path: str = None,
    output_binary: bool = False,
) -> None:
    if os.path.isdir(image_path_or_folder):
        return run_DSA_inference_on_folder(
            image_path_or_folder, predictor, output_path, output_binary
        )
    else:
        return run_DSA_inference_on_image(
            image_path_or_folder, predictor, output_path, output_binary
        )
