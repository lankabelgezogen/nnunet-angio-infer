import argparse
import os

import pydicom
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from utils.simple_predictor import SimplePNGPredictor, SimpleNiftiPredictor
from utils.DSA import run_DSA_inference
from utils.MRA import run_MRA_inference

# Monkeypatch nnU-Net's class finder to search local repo trainers first to avoid
# importing optional heavy deps from installed nnunetv2 (e.g., primus -> timm -> torchvision)
try:
    from nnunetv2.utilities import find_class_by_name as _fcbn
    import nnunetv2.inference.predict_from_raw_data as _pfrd

    _orig_recursive_find_python_class = _fcbn.recursive_find_python_class

    def _recursive_find_python_class_repo_first(
        folder: str, class_name: str, current_module: str
    ):
        try:
            local_pkg_dir = os.path.join(os.path.dirname(__file__), "nnUNetTrainer")
            if os.path.isdir(local_pkg_dir):
                tr = _orig_recursive_find_python_class(
                    local_pkg_dir, class_name, "nnUNetTrainer"
                )
                if tr is not None:
                    return tr
        except Exception:
            pass

        return _orig_recursive_find_python_class(folder, class_name, current_module)

    _fcbn.recursive_find_python_class = _recursive_find_python_class_repo_first
    if hasattr(_pfrd, "recursive_find_python_class"):
        _pfrd.recursive_find_python_class = _recursive_find_python_class_repo_first
except Exception:
    pass


class NNUNetV2Wrapper:
    def __init__(
        self,
        model_dir: str,
        folds: Tuple[int, ...] = (0,),
        device: Optional[torch.device] = None,
        mode: str = "DSA",
    ):
        """
        model_dir: path to the trained nnUNet v2 model folder (the one containing 'plans.json', 'fold_X', etc.)
        folds: which folds to use for prediction. By default, only fold 0 is used.
        """

        self.device = device
        if mode == "DSA":
            self.predictor = SimplePNGPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=device,
                verbose=False,
                verbose_preprocessing=False,
            )
        elif mode == "MRA":
            self.predictor = SimpleNiftiPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=device,
                verbose=False,
                verbose_preprocessing=False,
            )

        self.predictor.initialize_from_trained_model_folder(
            model_dir,
            use_folds=folds,
            checkpoint_name="checkpoint_best.pth",
        )

    def predict_array(
        self,
        image_path_or_folder: str,
        mode: str = None,
        output_path: str = None,
        output_binary: bool = False,
    ) -> None:
        """
        image_path_or_folder: path to an image file or folder
        mode: str, either "DSA", "MRA", "CTA"
        output_path: path to the output directory
        output_binary: whether to convert the output to a binary mask (0/1) instead of 0/255
        """

        if mode not in ["DSA", "MRA", "CTA"]:
            raise ValueError(f"Invalid mode: {mode}")

        if mode == "DSA":
            return run_DSA_inference(
                image_path_or_folder,
                self.predictor,
                output_path,
                output_binary,
            )

        elif mode == "MRA":
            return run_MRA_inference(image_path_or_folder, self.predictor, output_path)

        """ elif mode == "CTA":
            return run_CTA_inference_on_image(image_path_or_folder, self.predictor) """


def main():
    parser = argparse.ArgumentParser(
        description="Run trained nnUNet model on PNG/DICOM/NIFTI input."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input file or folder",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        help="Mode to use for inference",
    )
    parser.add_argument(
        "-md",
        "--model_dir",
        required=True,
        help="Path to nnUNet model directory",
    )
    parser.add_argument(
        "-f",
        "--fold",
        default="0",
        help="nnUNet fold",
    )
    parser.add_argument(
        "-b",
        "--binary",
        default=False,
        help="Whether to convert the output to a binary mask (0/1) instead of 0/255 (only for DSA)",
    )
    args = parser.parse_args()

    if args.mode == "MRA":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    nnunet_wrapper = NNUNetV2Wrapper(
        model_dir=args.model_dir,
        folds=args.fold,
        device=device,
        mode=args.mode,
    )

    nnunet_wrapper.predict_array(
        image_path_or_folder=args.input,
        mode=args.mode,
        output_path=args.output,
        output_binary=args.binary,
    )


if __name__ == "__main__":
    main()
