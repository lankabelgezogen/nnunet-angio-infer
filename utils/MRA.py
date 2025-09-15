import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import os


def preprocess_image(input_path: str) -> np.ndarray:
    if input_path.endswith(".nii.gz"):
        imgio = SimpleITKIO()
        img, props = imgio.read_images((input_path,))
        return img, props
    else:
        raise ValueError(f"Unsupported file type: {input_path.split('.')[-1]}")


def run_MRA_inference_on_image(
    image_path: str,
    predictor: nnUNetPredictor = None,
    output_path: str = None,
    output_binary: bool = False,
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


def run_MRA_inference_on_folder(
    folder: str,
    predictor: nnUNetPredictor = None,
    output_path: str = None,
    output_binary: bool = False,
) -> None:
    for file in os.listdir(folder):
        if file.endswith(".nii.gz"):
            run_MRA_inference_on_image(
                os.path.join(folder, file), predictor, output_path
            )
