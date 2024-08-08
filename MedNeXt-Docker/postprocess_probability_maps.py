import numpy as np
import SimpleITK as sitk
import math
import cv2
import json
import os
from glob import glob
from pathlib import Path
import numpy as np
from nnunetv2.postprocessing.remove_connected_components import (
    remove_all_but_largest_component_from_segmentation,
)


def get_binary_masks(softmax_array, thres, class_label):
    # get all binary segmentation maps - threshold operation
    binary_masks = softmax_array[class_label].copy()

    binary_masks[binary_masks >= thres] = 1
    binary_masks[binary_masks < thres] = 0

    return binary_masks.astype(np.uint8)


def get_positive_frames(mask):
    frames_positive = []
    for fr in range(len(mask)):
        if np.any(mask[fr] != 0):
            frames_positive.append(fr)
    return frames_positive


def merge_annotations(existing_labels, new_labels, priority_label=None):
    # Check if the arrays are 2D (single frame) or 3D (multiple frames)
    if len(existing_labels.shape) == 2:
        # Convert to 3D for consistent handling
        existing_labels = existing_labels[np.newaxis, ...]
        new_labels = new_labels[np.newaxis, ...]

    # Create a mask for the overlapping regions
    overlap_mask = (existing_labels != 0) & (new_labels != 0)

    # Initialize the merged labels with the existing labels
    merged_labels = existing_labels.copy()
    # Handle the non-overlapping regions
    print('Merging non-overlapping regions')
    merged_labels[new_labels != 0] = new_labels[new_labels != 0]

    if np.any(overlap_mask):
        print('Found overlapping labels. Merging...')
        # Handle the overlapping regions
        if priority_label is not None:
            # If a priority label is specified, use it for the overlapping regions
            merged_labels[overlap_mask] = priority_label
        else:
            # If no priority label is specified, choose the label with more pixels
            existing_pixels = np.sum(
                existing_labels == existing_labels[overlap_mask])
            new_pixels = np.sum(new_labels == new_labels[overlap_mask])
            merged_labels[overlap_mask] = np.where(
                existing_pixels >= new_pixels, existing_labels[overlap_mask], new_labels[overlap_mask])
    # If the input was 2D, return the 2D result
    if len(existing_labels.shape) == 2:
        return merged_labels[0, ...]
    return merged_labels


def postprocess_single_probability_map(softmax_prob_map, configs):
    # Define fetal structures labels
    labels_dict = dict(optimal=1, suboptimal=2)

    # Copy the input probability map
    softmax_maps = softmax_prob_map.copy()
    # Apply threshold
    softmax_maps[softmax_maps < configs["soft_threshold"]] = 0

    # Find the class with the maximum probability at each pixel across all channels
    # This will have shape [n_frames, H, W]
    masks = np.argmax(softmax_maps, axis=0)
    masks = masks.astype(np.uint8)

    # keep the largest connected component for each class
    masks_postprocessed = remove_all_but_largest_component_from_segmentation(
        masks, labels_or_regions=labels_dict.values())
    return masks_postprocessed


RESOURCE_PATH = Path("resources")


# Load MASK_FOV from the provided file path
MASK_FOV = sitk.GetArrayFromImage(sitk.ReadImage("resources/fov_mask/fov_mask.mha"))

def fit_ellipses(binary_mask, thickness=1):
    binary_mask = binary_mask.copy()
    binary_mask = np.where(binary_mask != 0, 255, 0).astype(np.uint8)
    mask_fov = np.where(MASK_FOV != 0, 255, 0).astype(np.uint8)

    binary_mask_padded = zero_pad_image(binary_mask, pad_width=200)
    mask_fov_padded = zero_pad_image(mask_fov, pad_width=200)

    binary_mask_fov = (binary_mask_padded * mask_fov_padded).astype(np.uint8)

    _, contours_non_truncated = get_non_truncated_ellipse_contours(
        binary_mask_fov, mask_fov_padded)

    if len(contours_non_truncated) > 5:
        ellipse = cv2.fitEllipse(contours_non_truncated)
        if not any(math.isnan(param) for param in ellipse[1]):
            a = ellipse[1][0] / 2
            b = ellipse[1][1] / 2

            circumference = ellipse_circumference(a, b)

            fitted_ellipse_mask = np.zeros_like(binary_mask_fov)
            cv2.ellipse(fitted_ellipse_mask, ellipse, 1, thickness)
            fitted_ellipse_mask = fitted_ellipse_mask[200:-200, 200:-200]

            # Create a filled ellipse mask aligned with the original mask
            filled_ellipse_mask = create_filled_ellipse(binary_mask.shape, ellipse)

            # Apply the FOV mask
            filled_ellipse_mask_fov = apply_fov_mask(filled_ellipse_mask, mask_fov)

            return ellipse, circumference, fitted_ellipse_mask, filled_ellipse_mask_fov
    print("No ellipse found")
    return None, None, None, None

def create_filled_ellipse(shape, ellipse):
    """
    Create a binary mask with a filled ellipse based on the ellipse parameters.
    """
    center = (int(ellipse[0][0]), int(ellipse[0][1]))
    axes = (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2))
    angle = int(ellipse[2])

    filled_ellipse_mask = np.zeros((shape[0] + 400, shape[1] + 400), dtype=np.uint8)
    cv2.ellipse(filled_ellipse_mask, center, axes, angle, 0, 360, 255, -1)
    
    # Remove the padding to align with the original mask shape
    filled_ellipse_mask = filled_ellipse_mask[200:-200, 200:-200]
    
    return filled_ellipse_mask

def apply_fov_mask(mask, fov_mask):
    """
    Apply the FOV mask to the segmentation mask.
    """
    return np.where(fov_mask != 0, mask, 0).astype(np.uint8)

def ellipse_circumference(a, b):
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))

def zero_pad_image(image, pad_width):
    if isinstance(pad_width, int):
        pad_width = image.ndim * [pad_width]

    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    return padded_image

def get_contour_points(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        print("No contours found in the binary mask.")
        return []
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour

def get_non_truncated_ellipse_contours(ellipse_mask, fov_mask):
    fov_mask_binary = np.where(fov_mask != 0, 255, 0).astype(np.uint8)
    ellipse_mask = np.where(ellipse_mask != 0, 255, 0).astype(np.uint8)

    contours_fov, _ = cv2.findContours(fov_mask_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_binary, _ = cv2.findContours(ellipse_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image_fov = np.zeros_like(fov_mask)
    contour_mask = np.zeros_like(fov_mask)
    cv2.drawContours(contour_image_fov, contours_fov, -1, 255, 1)
    cv2.drawContours(contour_mask, contours_binary, -1, 255, 1)

    non_truncated_contour_mask = np.logical_and(contour_mask, np.logical_not(contour_image_fov))
    non_truncated_contour_mask = np.where(non_truncated_contour_mask != 0, 255, 0).astype(np.uint8)
    contours_non_truncated = get_contour_points(non_truncated_contour_mask)
    return non_truncated_contour_mask, contours_non_truncated