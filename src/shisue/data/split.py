import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from shisue.utils.config import DataConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


def load_image_mask_pairs(images_dir: Path, masks_dir: Path, extension: str = 'png') -> Tuple[List[str], List[str]]:
    '''
    Load matching image and mask filenames from directories.

    Args:
        images_dir: Directory containing image files
        masks_dir: Directory containing mask files
        extension: File extension to match

    Returns:
        Tuple of (image_filenames, mask_filenames) as sorted lists
    '''
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if not images_dir.exists():
        raise MRIScanException(f"Images directory {images_dir} does not exist")
    if not masks_dir.exists():
        raise MRIScanException(f"Masks directory {masks_dir} does not exist")

    # Get all files with the specified extension
    image_files = sorted([f.name for f in images_dir.glob(f'*.{extension}')])
    mask_files = sorted([f.name for f in masks_dir.glob(f'*.{extension}')])

    if len(image_files) == 0:
        raise MRIScanException(f"No image files found in {images_dir}")
    if len(mask_files) == 0:
        raise MRIScanException(f"No mask files found in {masks_dir}")

    # Find matching pairs
    image_set = set(image_files)
    mask_set = set(mask_files)
    matching_files = sorted(list(image_set & mask_set))

    if len(matching_files) == 0:
        raise MRIScanException(f"No matching image and mask files found in {images_dir} and {masks_dir}")

    missing_masks = image_set - mask_set
    missing_images = mask_set - image_set

    if missing_masks:
        logger.warning(f"Found {len(missing_masks)} images without matching masks")
    if missing_images:
        logger.warning(f"Found {len(missing_images)} masks without matching images")

    logger.info(f"Found {len(matching_files)} matching image-mask pairs")

    return matching_files, matching_files


def compute_stratification_labels(mask_paths: List[Path], num_classes: int) -> np.ndarray:
    '''
    Compute stratification labels based on foreground/background presence.

    Uses a simple binary approach:
    - Label 0: Images containing ONLY background (class 0)
    - Label 1: Images containing at least one foreground class (classes 1+)

    This ensures robust stratification without rare group issues.

    Args:
        mask_paths: List of paths to mask files
        num_classes: Total number of classes

    Returns:
        Array of stratification labels (0 or 1)
    '''
    logger.info(f"Computing stratification labels (binary: background-only vs has-foreground)...")

    stratification_labels = []

    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        unique_classes = np.unique(mask)

        # Check if mask contains any foreground class (class != 0)
        has_foreground = any(cls > 0 for cls in unique_classes if cls < num_classes)

        # Binary label: 0 = background only, 1 = has foreground
        label = 1 if has_foreground else 0
        stratification_labels.append(label)

    stratification_labels = np.array(stratification_labels)

    # Log distribution
    unique, counts = np.unique(stratification_labels, return_counts=True)
    logger.info(f"Stratification distribution:")
    for lbl, count in zip(unique, counts):
        label_name = "background-only" if lbl == 0 else "has-foreground"
        logger.info(f"  - Label {lbl} ({label_name}): {count} samples")

    return stratification_labels


def create_data_split(
    images_dir: Path,
    masks_dir: Path,
    config: Optional[DataConfig] = None,
    stratify: bool = True,
    extension: str = 'png'
) -> Dict[str, List[str]]:
    '''
    Create train/validation/test split from image and mask directories.

    Args:
        images_dir: Directory containing image files
        masks_dir: Directory containing mask files
        config: DataConfig object with split ratios and seed
        stratify: Whether to stratify the split based on class presence
        extension: File extension to match

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing filenames
    '''
    config = config or DataConfig()

    # Validate split ratios
    total_split = config.train_split + config.val_split + config.test_split
    if not np.isclose(total_split, 1.0):
        raise MRIScanException(f"Split ratios must sum to 1.0. Got {total_split}")

    # Load matching pairs
    filenames, _ = load_image_mask_pairs(images_dir, masks_dir, extension)

    # Set random seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Compute stratification labels if needed
    stratify_labels = None
    if stratify:
        mask_paths = [Path(masks_dir) / f for f in filenames]
        stratify_labels = compute_stratification_labels(mask_paths, config.num_classes)

    # First split: train_val and test set
    # Stratify labels are only used for stratification, not ground truth labels
    train_val_files, test_files, train_val_labels, _ = train_test_split(
        filenames,
        stratify_labels if stratify else None,
        test_size=config.test_split,
        random_state=config.seed,
        stratify=stratify_labels if stratify else None
    )

    # Second split: train and validation set
    # Adjust validation ratio relative to the remaining data
    # Stratify labels are only used for stratification, not ground truth labels
    val_ratio_adj = config.val_split / (config.train_split + config.val_split)
    train_files, val_files = train_test_split(
        train_val_files,
        test_size=val_ratio_adj,
        random_state=config.seed,
        stratify=train_val_labels if stratify else None
    )

    split_dict = {
        'train': sorted(train_files),
        'val': sorted(val_files),
        'test': sorted(test_files)
    }

    logger.info(f"Created data split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # Log split percentages
    total = len(filenames)
    logger.info(f"Split percentages: train={len(train_files)/total:.1%}, val={len(val_files)/total:.1%}, test={len(test_files)/total:.1%}")

    return split_dict


def save_split(split_dict: Dict[str, List[str]], output_path: Path) -> None:
    '''
    Save dataset split to JSON file for reproducibility.

    Args:
        split_dict: Dictionary with 'train', 'val', 'test' keys containing filenames
        output_path: Path to save the JSON file
    '''
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(split_dict, f, indent=2)

    logger.info(f"Saved dataset split to {output_path}")


def load_split(split_path: Path) -> Dict[str, List[str]]:
    '''
    Load dataset split from JSON file.

    Args:
        split_path: Path to JSON file containing the dataset split

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing filenames
    '''
    split_path = Path(split_path)

    if not split_path.exists():
        raise MRIScanException(f"Split file not found at {split_path}")

    try:
        with open(split_path, 'r') as f:
            split_dict = json.load(f)
    
        # Validate required keys
        required_keys = {'train', 'val', 'test'}
        if not required_keys.issubset(split_dict.keys()):
            raise MRIScanException(f"Invalid split file. Expected keys: {required_keys}, but got: {set(split_dict.keys())}")

        logger.info(f"Loaded dataset split from {split_path}: train={len(split_dict['train'])}, val={len(split_dict['val'])}, test={len(split_dict['test'])}")

        return split_dict

    except json.JSONDecodeError as e:
        raise MRIScanException(f"Invalid JSON in split file {split_path}", details=str(e))


def get_or_create_split(
    images_dir: Path,
    masks_dir: Path,
    split_path: Optional[Path] = None,
    config: Optional[DataConfig] = None,
    stratify: bool = True,
    force_recreate: bool = False
) -> Dict[str, List[str]]:
    '''
    Load existing split or create new one if it doesn't exist.

    This is the recommended function for obtaining dataset splits, as it ensures reproducibility while avoiding unnecessary recomputation.

    Args:
        images_dir: Directory containing image files
        masks_dir: Directory containing mask files
        split_path: Path to JSON file containing the dataset split (if None, create split without saving)
        config: DataConfig object with split ratios and seed
        stratify: Whether to stratify the split based on class presence
        force_recreate: If True, recreate split even if file exists

    Returns:
        Dictionary with 'train', 'val', 'test' keys containing filenames
    '''
    if split_path is None:
        logger.info(f"No split path provided, creating split without saving")
        return create_data_split(images_dir, masks_dir, config, stratify)

    split_path = Path(split_path)

    # Load existing split if available
    if split_path.exists() and not force_recreate:
        logger.info(f"Loading existing split from {split_path}")
        return load_split(split_path)

    # Create new split
    logger.info(f"Creating new dataset split...")
    split_dict = create_data_split(images_dir, masks_dir, config, stratify)

    # Save split for future use
    save_split(split_dict, split_path)

    return split_dict