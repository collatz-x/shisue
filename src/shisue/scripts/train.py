from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from shisue.data.dataset import MRIScanDataset
from shisue.data.split import get_or_create_split
from shisue.data.transform import get_transform
from shisue.models.transunet import build_transunet
from shisue.training.trainer import Trainer
from shisue.utils.config import Config, DataConfig, ModelConfig, TrainingConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger, setup_logging


logger = get_logger(__name__)


def load_config_from_hydra(cfg: DictConfig) -> Config:
    '''
    Convert Hydra DictConfig to typed Config dataclass.

    Args:
        cfg: Hydra configuration object

    Returns:
        Config object with typed sub-configs
    '''
    # Convert OmegaConf to dict
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create typed configs
    model_config = ModelConfig(**cfg_dict.get('model', {}))
    data_config = DataConfig(**cfg_dict.get('data', {}))
    training_config = TrainingConfig(**cfg_dict.get('training', {}))

    config = Config(
        model=model_config,
        data=data_config,
        training=training_config
    )

    return config


def create_dataloaders(split_dict: dict, config: DataConfig) -> tuple[DataLoader, DataLoader]:
    '''
    Create train and validation DataLoaders.

    Args:
        split_dict: Dictionary with train/val/test splits
        config: Data configuration

    Returns:
        Typed tuple of (train_loader, val_loader)
    '''
    # Get directories
    images_dir = Path(config.images_dir)
    masks_dir = Path(config.masks_dir)

    # Create transforms
    train_transform = get_transform('train', config)
    val_transform = get_transform('val', config)

    # Create datasets
    train_dataset = MRIScanDataset.from_split(
        split_dict=split_dict,
        mode='train',
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=train_transform,
        config=config
    )

    val_dataset = MRIScanDataset.from_split(
        split_dict=split_dict,
        mode='val',
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=val_transform,
        config=config
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True                          # Drop last incomplete batch for consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False                         # Keep last incomplete batch for validation
    )

    logger.info(f"Created DataLoaders: train={len(train_loader)} batches, val={len(val_loader)} batches")

    return train_loader, val_loader


def compute_class_weights_from_split(split_dict: dict, images_dir: Path, masks_dir: Path, config: DataConfig) -> Optional[torch.Tensor]:
    '''
    Compute class weights for imbalanced datasets from training split.

    Uses inverse frequency weighting: weight_c = 1 / freq_c

    Args:
        split_dict: Dictionary with train/val/test splits
        images_dir: Directory containing image files
        masks_dir: Directory containing mask files
        config: Data configuration

    Returns:
        Tensor of class weights of shape (C, ) or None if computation fails
    '''
    try:
        logger.info("Computing class weights from training masks...")

        # Get training mask paths
        train_filenames = split_dict['train']

        logger.info(f"Calculating class distribution for {len(train_filenames)} training masks...")

        # Count pixels per class across all training masks (use numpy for faster accumulation)
        class_counts = np.zeros(config.num_classes, dtype=np.float64)

        for filename in tqdm(train_filenames, desc="Scanning masks"):
            mask_path = masks_dir / filename

            if not mask_path.exists():
                logger.warning(f"Mask not found: {mask_path}, skipping...")
                continue
        
            # Load mask
            mask = np.array(Image.open(mask_path).convert('L'))

            # Count pixels per class
            unique, counts = np.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                if 0 <= cls < config.num_classes:
                    class_counts[cls] += count
                else:
                    logger.warning(f"Found invalid class {cls} in {filename}")

        # Convert to torch tensor for weight computation
        class_counts = torch.from_numpy(class_counts)

        # Compute inverse frequency weights
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        class_weights = 1.0 / (class_counts + epsilon)

        # Normalize weights to have mean = 1.0
        class_weights = class_weights / class_weights.mean()

        # Convert class weights to float32 for MPS compatibility
        class_weights = class_weights.float()

        logger.info(f"Class distribution (pixels): {class_counts.tolist()}")
        logger.info(f"Computed class weights: {class_weights.tolist()}")
        
        return class_weights
    
    except Exception as e:
        logger.error(f"Failed to compute class weights: {e}. Proceeding without class weights.")

        return None


@hydra.main(version_base=None, config_path="../../../config", config_name="config")
def main(cfg: DictConfig):
    '''
    Main training function.

    Args:
        cfg: Hydra configuration object loaded from YAML file
    '''
    # Setup logging
    setup_logging(log_dir=cfg.training.log_dir)

    logger.info("=" * 80)
    logger.info("TransUNet Training Script")
    logger.info("=" * 80)

    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Load typed config from Hydra
    config = load_config_from_hydra(cfg)

    # Validate configuration
    if config.model.n_classes != config.data.num_classes:
        raise MRIScanException(f"Mismatch between model.n_classes ({config.model.n_classes}) and data.num_classes ({config.data.num_classes})")

    if config.training.num_classes != config.data.num_classes:
        raise MRIScanException(f"Mismatch between training.num_classes ({config.training.num_classes}) and data.num_classes ({config.data.num_classes})")

    # Create or load data split
    logger.info("Loading dataset split...")
    images_dir = Path(config.data.images_dir)
    masks_dir = Path(config.data.masks_dir)
    split_path = Path(config.data.data_dir) / "split.json"

    split_dict = get_or_create_split(
        images_dir=images_dir,
        masks_dir=masks_dir,
        split_path=split_path,
        config=config.data
    )

    # Compute class weights for imbalanced datasets (optional)
    class_weights = None
    if config.training.use_class_weights and config.training.loss_type in ['ce', 'combined']:
        logger.info("Class weights enabled in config. Starting class weight computation...")
        class_weights = compute_class_weights_from_split(
            split_dict=split_dict,
            images_dir=images_dir,
            masks_dir=masks_dir,
            config=config.data
        )
    
    else:
        logger.info("Class weights disabled in config. Proceeding without class weights.")

    # Create DataLoaders
    logger.info("Creating DataLoaders...")
    train_loader, val_loader = create_dataloaders(split_dict, config.data)

    # Initialize model
    logger.info("Initializing TransUNet model...")
    model = build_transunet(config.model)

    # Log model info
    total_params, trainable_params = model.count_parameters()
    logger.info(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")

    # Initialize Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        config=config.training,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        class_names=config.data.class_names
    )

    # Start training
    try:
        logger.info("Starting training...")
        trainer.train()

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Best validation Dice score: {trainer.best_metric:.4f}")
        logger.info(f"Checkpoints saved to: {config.training.checkpoint_dir}")
        logger.info(f"TensorBoard logs saved to: {config.training.tensorboard_dir}")
        logger.info("=" * 80)        

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Saving checkpoint before exit...")

        # Save interrupted checkpoint
        interrupted_path = Path(config.training.checkpoint_dir) / 'interrupted_model.pth'
        trainer.save_checkpoint(interrupted_path)
        logger.info(f"Checkpoint saved to {interrupted_path}")        

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise MRIScanException(f"Training failed", details=str(e)) from e


if __name__ == '__main__':
    main()