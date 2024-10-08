from pathlib import Path

import threading
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from postprocess_probability_maps import postprocess_single_probability_map

RESOURCE_PATH = Path("resources")


class FetalAbdomenSegmentation(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # Path to nnUNet model directory
        self.nnunet_model_dir = RESOURCE_PATH / "nnUNet_results"

        # Initialize the predictor
        self.predictor = self.initialize_predictor()

    def initialize_predictor(self, task="Dataset001_US",
                             network="2d", checkpoint="checkpoint_final.pth", folds=(0,1,2,3)):
        """
        Initializes the nnUNet predictor
        """
        
        # instantiates the predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(self.nnunet_model_dir,
                 f'{task}/nnUNetTrainerV2_MedNeXt_B_kernel5_100epochs__nnUNetPlans__{network}'),
            use_folds=folds,
            checkpoint_name=checkpoint,
        )
        predictor.dataset_json['file_ending'] = '.mha'

        return predictor

    def predict(self, input_img_path, save_probabilities=True):
        """
        Use trained nnUNet network to generate segmentation masks
        """
        # ideally we would like to use predictor.predict_from_files but this docker container will be called
        # for each individual test case so that this doesn't make sense
        image_np, properties = SimpleITKIO().read_images([input_img_path])
        _, probabilities = self.predictor.predict_single_npy_array(
            image_np, properties, None, None, save_probabilities)

        return probabilities

    def postprocess(self, probability_map):
        """
        Postprocess the nnUNet output to generate the final AC segmentation mask
        """
        # Define the postprocessing configurations
        configs = {
            "soft_threshold": 0.5,
        }

        # Postprocess the probability map
        mask_postprocessed = postprocess_single_probability_map(
            probability_map, configs)
        print('Postprocessing done')
        return mask_postprocessed


def select_fetal_abdomen_mask_and_frame(segmentation_masks: np.ndarray, result: dict) -> None:
    """
    Select the fetal abdomen mask and the corresponding frame number from the segmentation masks.
    Updates the result dictionary with the best selection found before timeout.
    """
    largest_area = 0
    selected_image = None
    fetal_abdomen_frame_number = -1

    for frame in range(segmentation_masks.shape[0]):
        current_frame = segmentation_masks[frame]

        # Check if both classes are present in the current frame
        if np.any(current_frame == 1) and np.any(current_frame == 2):
            area_class_1 = np.sum(current_frame == 1)
            area_class_2 = np.sum(current_frame == 2)
            combined_area = area_class_1 + area_class_2
        else:
            area_class_1 = np.sum(current_frame == 1)
            area_class_2 = np.sum(current_frame == 2)
            combined_area = max(area_class_1, area_class_2)

        if combined_area > largest_area:
            largest_area = combined_area
            selected_image = current_frame
            fetal_abdomen_frame_number = frame
            
            # Update the result dictionary with the latest best values
            result['largest_area'] = largest_area
            result['selected_image'] = selected_image
            result['fetal_abdomen_frame_number'] = fetal_abdomen_frame_number

    if selected_image is None:
        selected_image = np.zeros_like(segmentation_masks[0])
    
    selected_image = (selected_image > 0).astype(np.uint8)
    result['selected_image'] = selected_image
    result['fetal_abdomen_frame_number'] = fetal_abdomen_frame_number


def run_with_timeout(func, args=(), kwargs=None, timeout=60):
    if kwargs is None:
        kwargs = {}

    result = {'largest_area': 0, 'selected_image': None, 'fetal_abdomen_frame_number': -1}
    thread = threading.Thread(target=func, args=args + (result,), kwargs=kwargs)
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        print(f"Function exceeded {timeout / 60} minutes timeout. Returning the best result achieved so far.")
        thread.join()  # Ensure the thread has finished or forcefully terminate (Python doesn’t have direct termination)
    
    # If the function times out, it returns the best result found so far
    return result['selected_image'], result['fetal_abdomen_frame_number']
