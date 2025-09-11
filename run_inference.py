import argparse
import os

import pydicom
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from utils.DSA import run_inference_DSA

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
    ):
        """
        model_dir: path to the trained nnUNet v2 model folder (the one containing 'plans.json', 'fold_X', etc.)
        folds: which folds to use for prediction. By default, only fold 0 is used.
        """
        if device is None:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        self.device = device
        self.predictor = nnUNetPredictor(
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
        image: np.ndarray | str,
        target_size: tuple[int, int] = (512, 512),
        mode: str = "DSA",
    ) -> np.ndarray:
        """
        image: numpy array or path to an image file
        target_size: tuple[int, int] = (512, 512)
        mode: str, either "DSA", "MRA", "CTA"

        Returns: binary mask numpy array (uint8 with values 0 or 255)
        """

        if mode not in ["DSA", "MRA", "CTA"]:
            raise ValueError(f"Invalid mode: {mode}")

        if mode == "DSA":
            return run_inference_DSA(image, target_size, self.predictor)

        """ elif mode == "MRA":
            return run_inference_MRA(image, target_size, self.predictor)

        elif mode == "CTA":
            return run_inference_CTA(image, target_size, self.predictor) """


def main():
    parser = argparse.ArgumentParser(
        description="Run trained nnUNet model on PNG/DICOM input."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input file",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output directory"
    )
    parser.add_argument(
        "-m",
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
    args = parser.parse_args()

    nnunet_wrapper = NNUNetV2Wrapper(
        model_dir=args.model_dir,
        folds=args.fold,
        device=torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )

    mask = nnunet_wrapper.predict_array(image=args.input, mode="DSA")

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(
        out_path, f"{args.input.split('/')[-1].split('.')[0]}_prediction.png"
    )

    Image.fromarray(mask).save(out_path)


if __name__ == "__main__":
    main()
