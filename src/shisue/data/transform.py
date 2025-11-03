from typing import Callable, Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

from shisue.utils.config import DataConfig
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


def get_training_transform(config: Optional[DataConfig] = None) -> Callable:
    '''
    Get training transformation pipeline with augmentations.

    Applies:
    - Resize to target size (224x224)
    - Horizontal flip
    - Vertical flip
    - Rotation (±rotation_limit degrees)
    - Normalization (per-image z-score or ImageNet stats)
    - Convert to PyTorch tensor

    Args:
        config: DataConfig object with augmentation parameters

    Returns:
        Albumentations Compose transform
    '''
    config = config or DataConfig()

    transforms_list = [
        # Resize to target size (512x512 -> 224x224)
        A.Resize(
            height=config.image_size,
            width=config.image_size,
            interpolation=cv2.INTER_CUBIC,
            p=1.0
        )
    ]

    # Add augmentations if enabled
    if config.use_augmentation:
        transforms_list.extend([
            # Horizontal flip
            A.HorizontalFlip(p=config.horizontal_flip_prob),

            # Vertical flip
            A.VerticalFlip(p=config.vertical_flip_prob),

            # Rotation (±rotation_limit degrees)
            A.Rotate(
                limit=config.rotation_limit,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                p=config.rotation_prob
            )
        ])

    # Normalization
    if config.normalize:
        if config.mean is not None and config.std is not None:
            # Use provided normalization statistics
            transforms_list.append(
                A.Normalize(
                    mean=config.mean,
                    std=config.std,
                    max_pixel_value=255.0,
                )
            )
        else:
            # Use ImageNet statistics (standard for ResNet backbone)
            transforms_list.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                )
            )

    # Convert to PyTorch tensor (C, H, W) format
    transforms_list.append(ToTensorV2())

    transform = A.Compose(transforms_list)

    logger.info(f"Created training transform: resize={config.image_size}, augment={config.use_augmentation}, normalize={config.normalize}")

    return transform


def get_validation_transform(config: Optional[DataConfig] = None) -> Callable:
    '''
    Get validation/test transformation pipeline without augmentations.

    Applies:
    - Resize to target size (224x224)
    - Normalization (per-image z-score or ImageNet stats)
    - Convert to PyTorch tensor

    Args:
        config: DataConfig object with transformation parameters

    Returns:
        Albumentations Compose transform
    '''
    config = config or DataConfig()

    transforms_list = [
        # Resize to target size (512x512 -> 224x224)
        A.Resize(
            height=config.image_size,
            width=config.image_size,
            interpolation=cv2.INTER_CUBIC,
            p=1.0
        )
    ]

    # Normalization
    if config.normalize:
        if config.mean is not None and config.std is not None:
            # Use provided normalization statistics
            transforms_list.append(
                A.Normalize(
                    mean=config.mean,
                    std=config.std,
                    max_pixel_value=255.0,
                )
            )
        else:
            # Use ImageNet statistics (standard for ResNet backbone)
            transforms_list.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                )
            )

    # Convert to PyTorch tensor (C, H, W) format
    transforms_list.append(ToTensorV2())

    transform = A.Compose(transforms_list)

    logger.info(f"Created validation/test transform: resize={config.image_size}, normalize={config.normalize}")

    return transform


def get_transform(mode: str, config: Optional[DataConfig] = None) -> Callable:
    '''
    Get transformation pipeline based on dataset mode.

    Args:
        mode: Dataset mode ('train', 'val', 'test')
        config: DataConfig object with transformation parameters

    Returns:
        Appropriate transformation pipeline for the mode
    '''
    if mode == 'train':
        return get_training_transform(config)
    elif mode in ['val', 'test']:
        return get_validation_transform(config)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be one of ['train', 'val', 'test']")


def compute_normalization_stats(image_paths: list, sample_size: Optional[int] = None) -> tuple:
    '''
    Compute mean and std for dataset normalization.

    This function computes channel-wise mean and standard deviation from a sample of images in the dataset.

    Args:
        image_paths: List of paths to image files
        sample_size: Number of images to sample (None = use all)

    Returns:
        Tuple of (mean, std) as lists of length 3 for RGB channels

    Note:
        This function can be slow for large datasets. Consider using a smaller sample size.
    '''
    from PIL import Image
    import random
    
    if sample_size is not None and sample_size < len(image_paths):
        image_paths = random.sample(image_paths, sample_size)

    logger.info(f"Computing normalization stats from {len(image_paths)} images...")

    # Accumulate pixel values
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0     # Normalize to [0, 1]
        
        # Sum over height and width
        pixel_sum += img_array.sum(axis=(0, 1))
        pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1))
        pixel_count += img_array.shape[0] * img_array.shape[1]

    # Compute mean and std
    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    mean_list = mean.tolist()
    std_list = std.tolist()

    logger.info(f"Normalization stats computed: mean={mean_list}, std={std_list}")
    return mean_list, std_list