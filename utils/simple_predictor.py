import os, pathlib
from typing import Union, List

import torch
import numpy as np
import multiprocessing
from time import sleep

from batchgenerators.utilities.file_and_folder_operations import isfile, join
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.file_path_utilities import (
    check_workers_alive_and_busy,
)
from nnunetv2.utilities.helpers import empty_cache


class SimplePNGPredictor(nnUNetPredictor):
    """Single-channel PNG inference without requiring `_0000` suffix."""

    def _scan_png_folder(self, folder: str) -> List[List[str]]:
        return [
            [str(p)]
            for p in sorted(pathlib.Path(folder).iterdir())
            if p.is_file() and p.suffix.lower() == ".png"
        ]

    def _caseids_from_lists(self, lol: List[List[str]]) -> List[str]:
        return [pathlib.Path(chlist[0]).stem for chlist in lol if chlist]

    def _manage_input_and_output_lists(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[None, str, List[str]],
        folder_with_segs_from_prev_stage: str = None,
        overwrite: bool = True,
        part_id: int = 0,
        num_parts: int = 1,
        save_probabilities: bool = False,
    ):
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = self._scan_png_folder(
                list_of_lists_or_source_folder
            )

        print(
            f"There are {len(list_of_lists_or_source_folder)} cases in the source folder"
        )
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[
            part_id::num_parts
        ]
        caseids = self._caseids_from_lists(list_of_lists_or_source_folder)
        print(f"I am processing {part_id} of {num_parts} parts")
        print(f"There are {len(caseids)} cases that I would like to predict")

        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_filename_truncated = [
                join(output_folder_or_list_of_truncated_output_files, i)
                for i in caseids
            ]
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_filename_truncated = output_folder_or_list_of_truncated_output_files[
                part_id::num_parts
            ]
        else:
            output_filename_truncated = None

        seg_from_prev_stage_files = [None] * len(caseids)

        if not overwrite and output_filename_truncated is not None:
            fe = self.dataset_json.get("file_ending", ".png")
            tmp = [isfile(i + fe) for i in output_filename_truncated]
            if save_probabilities:
                tmp2 = [isfile(i + ".npz") for i in output_filename_truncated]
                tmp = [a and b for a, b in zip(tmp, tmp2)]
            keep = [k for k, done in enumerate(tmp) if not done]
            output_filename_truncated = [output_filename_truncated[k] for k in keep]
            list_of_lists_or_source_folder = [
                list_of_lists_or_source_folder[k] for k in keep
            ]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[k] for k in keep]
            print(f"overwrite=False: working on {len(keep)} new cases.")

        return (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
        )

    def predict_from_files(
        self,
        list_of_lists_or_source_folder: Union[str, List[List[str]]],
        output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
        save_probabilities: bool = False,
        overwrite: bool = True,
        num_processes_preprocessing: int = default_num_processes,
        num_processes_segmentation_export: int = default_num_processes,
        folder_with_segs_from_prev_stage: str = None,
        num_parts: int = 1,
        part_id: int = 0,
        output_binary: bool = False,
    ):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        assert part_id <= num_parts, (
            "Part ID must be smaller than num_parts. Remember that we start counting with 0. "
            "So if there are 3 parts then valid part IDs are 0, 1, 2"
        )
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(
                output_folder_or_list_of_truncated_output_files[0]
            )
        else:
            output_folder = None

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, (
                f"The requested configuration is a cascaded network. It requires the segmentations of the previous "
                f"stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where"
                f" they are located via folder_with_segs_from_prev_stage"
            )

        # sort out input and output filenames
        (
            list_of_lists_or_source_folder,
            output_filename_truncated,
            seg_from_prev_stage_files,
        ) = self._manage_input_and_output_lists(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            folder_with_segs_from_prev_stage,
            overwrite,
            part_id,
            num_parts,
            save_probabilities,
        )
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(
            list_of_lists_or_source_folder,
            seg_from_prev_stage_files,
            output_filename_truncated,
            num_processes_preprocessing,
        )

        return self.predict_from_data_iterator(
            data_iterator,
            save_probabilities,
            num_processes_segmentation_export,
            output_binary,
        )

    def predict_from_data_iterator(
        self,
        data_iterator,
        save_probabilities: bool = False,
        num_processes_segmentation_export: int = default_num_processes,
        output_binary: bool = False,
    ):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        expected_png_paths = []
        with multiprocessing.get_context("spawn").Pool(
            num_processes_segmentation_export
        ) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:
                data = preprocessed["data"]
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed["ofile"]
                if ofile is not None:
                    print(f"\nPredicting {os.path.basename(ofile)}:")
                else:
                    print(f"\nPredicting image of shape {data.shape}:")

                print(
                    f"perform_everything_on_device: {self.perform_everything_on_device}"
                )

                properties = preprocessed["data_properties"]

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to be swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(
                    export_pool, worker_list, r, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        export_pool, worker_list, r, allowed_num_queued=2
                    )

                # convert to numpy to prevent uncatchable memory alignment errors from multiprocessing serialization of torch tensors
                prediction = (
                    self.predict_logits_from_preprocessed_data(data)
                    .cpu()
                    .detach()
                    .numpy()
                )

                if ofile is not None:
                    print(
                        "sending off prediction to background worker for resampling and export"
                    )
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            (
                                (
                                    prediction,
                                    properties,
                                    self.configuration_manager,
                                    self.plans_manager,
                                    self.dataset_json,
                                    ofile,
                                    save_probabilities,
                                ),
                            ),
                        )
                    )
                    fe = self.dataset_json.get("file_ending", ".png")
                    expected_png_paths.append(ofile + fe)
                else:
                    print("sending off prediction to background worker for resampling")
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape,
                            (
                                (
                                    prediction,
                                    self.plans_manager,
                                    self.configuration_manager,
                                    self.label_manager,
                                    properties,
                                    save_probabilities,
                                ),
                            ),
                        )
                    )
                if ofile is not None:
                    print(f"done with {os.path.basename(ofile)}")
                else:
                    print(f"\nDone with image of shape {data.shape}:")
            _ = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        if not output_binary:
            try:
                from PIL import Image

                for png_path in expected_png_paths:
                    if not (
                        isinstance(png_path, str) and png_path.lower().endswith(".png")
                    ):
                        continue
                    if not os.path.isfile(png_path):
                        continue

                    img = Image.open(png_path)
                    arr = np.array(img)

                    vis = (arr > 0).astype(np.uint8) * 255
                    Image.fromarray(vis, mode="L").save(png_path)
            except Exception as e:
                print(f"[warn] PNG 0/255 postprocess failed: {e}")

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return expected_png_paths
