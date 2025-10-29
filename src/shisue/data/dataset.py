from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from shisue.utils.config import DataConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class MRIScanDataset(Dataset):
    '''
    PyTorch Dataset for MRI segmentation with support for train/val/test splits.

    This dataset handles:
    - Loading grayscale MRI images (512x512) and multi-class masks
    - Applying mode-specific transformations (augmentation for training)
    - Converting grayscale images to RGB (required for ResNet backbone)
    - Resizing images from 512x512 to 224x224

    Attributes:
        mode: Dataset mode ('train', 'val', 'test')
        image_paths: List of paths to image files
        mask_paths: List of paths to mask files
        transform: Optional transformation function to apply
        config: DataConfig object with dataset parameters
    '''

    VALID_MODES = {'train', 'val', 'test'}

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        mode: str = 'train',
        transform: Optional[Callable] = None,
        config: Optional[DataConfig] = None
    ):
        '''
        Initialize the MRI scan dataset.

        Args:
            image_paths: List of paths to image files
            mask_paths: List of paths to mask files
            mode: Dataset mode ('train', 'val', 'test')
            transform: Optional transformation function to apply
            config: DataConfig object with dataset parameters
        '''
        if mode not in self.VALID_MODES:
            raise MRIScanException(f"Invalid dataset mode: {mode}. Must be one of {self.VALID_MODES}")

        if len(image_paths) != len(mask_paths):
            raise MRIScanException(f"Mismatched lengths: {len(image_paths)} images vs {len(mask_paths)} masks")

        if len(image_paths) == 0:
            raise MRIScanException(f"No images provided for {mode} dataset")

        self.mode = mode
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        self.transform = transform
        self.config = config or DataConfig()

        # Validate that all files exist
        self._validate_paths()

        logger.info(f"Initialized {mode} dataset with {len(self)} samples (num_classes={self.config.num_classes})")

    def _validate_paths(self) -> None:
        '''Validate that all image and mask files exist.'''
        missing_images = [p for p in self.image_paths if not p.exists()]
        missing_masks = [p for p in self.mask_paths if not p.exists()]

        if missing_images or missing_masks:
            error_msg = []
            if missing_images:
                error_msg.append(f"{len(missing_images)} missing images")
            if missing_masks:
                error_msg.append(f"{len(missing_masks)} missing masks")

            raise MRIScanException("; ".join(error_msg))

    def _load_image(self, path: Path) -> np.ndarray:
        '''
        Load grayscale image and convert to RGB format.

        Args:
            path: Path to image file

        Returns:
            RGB image as numpy array (H, W, 3) with dtype uint8
        '''
        try:
            # Load grayscale image and convert to RGB
            image = Image.open(path).convert('RGB')
            image_array = np.array(image, dtype=np.uint8)

            return image_array

        except Exception as e:
            raise MRIScanException(f"Failed to load image from {path}", details=str(e))

    def _load_mask(self, path: Path) -> np.ndarray:
        '''
        Load segmentation mask as grayscale.

        Args:
            path: Path to mask file

        Returns:
            Mask as numpy array (H, W) with dtype int64
        '''
        try:
            # Load mask as grayscale
            mask = Image.open(path).convert('L')
            mask_array = np.array(mask, dtype=np.int64)

            # Validate mask values are within expected range
            unique_values = np.unique(mask_array)
            invalid_values = unique_values[unique_values >= self.config.num_classes]

            if len(invalid_values) > 0:
                raise MRIScanException(
                    f"Mask contains invalid class labels {invalid_values.tolist()} (expected 0 - {self.config.num_classes - 1})")

            return mask_array
        
        except MRIScanException:
            raise
        except Exception as e:
            raise MRIScanException(f"Failed to load mask from {path}", details=str(e))

    def __len__(self) -> int:
        '''Return the number of samples in the dataset.'''
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        '''
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing:
                - 'image': RGB image tensor (C, H, W) with dtype float32
                - 'mask': Segmentation mask tensor (H, W) with dtype int64
                - 'filename': Original filename without extension
        '''
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        # Get paths
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = self._load_image(image_path)    # (H, W, 3) uint8
        mask = self._load_mask(mask_path)       # (H, W) int64

        # Apply transformation if provided
        if self.transform is not None:
            try:
                # Transform expects dict with 'image' and 'mask' keys
                # Returns transformed dict
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            except Exception as e:
                raise MRIScanException(f"Transformation failed for {image_path.name}", details=str(e))

        # Convert to torch tensors
        if not isinstance(image, torch.Tensor):
            # Convert numpy array to tensor and permute to (C, H, W)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            # Ensure float32 dtype
            image = image.float()

        if not isinstance(mask, torch.Tensor):
            # Convert mask to tensor
            mask = torch.from_numpy(mask).long()
        else:
            # Ensure int64 dtype (ToTensorV2 creates int32 by default)
            mask = mask.long()

        # Extract filename without extension
        filename = image_path.stem

        return {
            'image': image,
            'mask': mask,
            'filename': filename
        }

    @classmethod
    def from_split(
        cls,
        split_dict: Dict[str, List[str]],
        mode: str,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        config: Optional[DataConfig] = None
    ) -> 'MRIScanDataset':
        '''
        Create dataset from split dictionary (output from split.py).

        Args:
            split_dict: Dictionary with 'train', 'val', 'test' keys containing filenames
            mode: Dataset mode ('train', 'val', 'test')
            images_dir: Directory containing image files
            masks_dir: Directory containing mask files
            transform: Optional transformation function to apply
            config: DataConfig object with dataset parameters
        
        Returns:
            MRIScanDataset instance
        '''
        if mode not in split_dict:
            raise MRIScanException(f"Mode {mode} not found in split dictionary. Available modes: {list(split_dict.keys())}")

        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)

        if not images_dir.exists():
            raise MRIScanException(f"Images directory {images_dir} does not exist")
        if not masks_dir.exists():
            raise MRIScanException(f"Masks directory {masks_dir} does not exist")

        # Get filenames for this split
        filenames = split_dict[mode]

        # Construct full paths
        image_paths = [images_dir / f for f in filenames]
        mask_paths = [masks_dir / f for f in filenames]

        return cls(
            image_paths=image_paths,
            mask_paths=mask_paths,
            mode=mode,
            transform=transform,
            config=config
        )

    def get_class_distribution(self) -> Dict[int, int]:
        '''
        Calculate the distribution of classes in the dataset.

        This is useful for computing class weights for loss functions.

        Returns:
            Dictionary mapping class labels to pixel counts

        Note:
            This method loads all masks, which may be slow for large datasets.
        '''
        class_counts = {i: 0 for i in range(self.config.num_classes)}

        logger.info(f"Calculating class distribution for {len(self)} samples...")

        for mask_path in self.mask_paths:
            mask = self._load_mask(mask_path)
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[int(cls)] += int(count)

        logger.info(f"Class distribution computed: {class_counts}")
        return class_counts

    def get_sample_info(self, idx: int) -> Dict[str, Union[str, tuple]]:
        '''
        Get metadata about a sample without loading the full dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary with sample metadata
        '''
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load to get shapes without applying transformations
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        return {
            'filename': image_path.stem,
            'image_path': str(image_path),
            'mask_path': str(mask_path),
            'image_size': image.size,
            'mask_size': mask.size,
            'mode': self.mode
        }