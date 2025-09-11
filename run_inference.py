import argparse
import os

import pydicom
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

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
        self, image_np: np.ndarray | str, target_size=(512, 512)
    ) -> np.ndarray:
        """
        image_np: 2D grayscale numpy array (H, W) or path to an image file
        Returns: binary mask numpy array (H, W), uint8 with values 0 or 255
        """
        if isinstance(image_np, str):
            img = Image.open(image_np).convert("L")
            image_np = np.array(img)

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
        pred = self.predictor.predict_logits_from_preprocessed_data(img_torch)[0]

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


def main():
    parser = argparse.ArgumentParser(
        description="Run trained nnUNet model on PNG/DICOM input."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to input file (PNG/DICOM)",
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

    processed_path = preprocess_image(args.input)
    nnunet_wrapper = NNUNetV2Wrapper(
        model_dir=args.model_dir,
        folds=args.fold,
        device=torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )

    mask = nnunet_wrapper.predict_array(processed_path)

    out_path = args.output
    os.makedirs(out_path, exist_ok=True)
    out_path = os.path.join(
        out_path, f"{args.input.split('/')[-1].split('.')[0]}_prediction.png"
    )

    Image.fromarray(mask).save(out_path)


if __name__ == "__main__":
    main()
