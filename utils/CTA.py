import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
import os

# Import required for NibabelIOWithReorient functionality
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
import nibabel


def preprocess_image(input_path: str) -> np.ndarray:
    """
    Use NibabelIOWithReorient to ensure consistency with TopCoW dataset configuration.
    The TopCoW dataset.json specifies "overwrite_image_reader_writer": "NibabelIOWithReorient"
    """
    if input_path.endswith(".nii.gz"):
        imgio = NibabelIOWithReorient()
        img, props = imgio.read_images((input_path,))
        return img, props
    else:
        raise ValueError(f"Unsupported file type: {input_path.split('.')[-1]}")


def run_CTA_inference_on_image(
    image_path: str,
    predictor: nnUNetPredictor = None,
    output_path: str = None,
) -> None:
    image, props = preprocess_image(image_path)
    print(image.shape)  # (1, X, Y, Z)
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(
        output_path, f"{image_path.split('/')[-1].split('.')[0]}"
    )

    predictor.predict_single_npy_array(
        image.astype(np.float32), props, output_file_truncated=output_path
    )


def run_CTA_inference_on_folder(
    folder: str,
    predictor: nnUNetPredictor = None,
    output_path: str = None,
) -> None:
    for file in os.listdir(folder):
        if file.endswith(".nii.gz"):
            run_CTA_inference_on_image(
                os.path.join(folder, file), predictor, output_path
            )


def run_CTA_inference(
    image_path_or_folder: str,
    predictor: nnUNetPredictor = None,
    output_path: str = None,
) -> None:
    if os.path.isdir(image_path_or_folder):
        return run_CTA_inference_on_folder(image_path_or_folder, predictor, output_path)
    else:
        return run_CTA_inference_on_image(image_path_or_folder, predictor, output_path)
