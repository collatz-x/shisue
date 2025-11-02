from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from shisue.utils.config import TrainingConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class DiceLoss(nn.Module):
    '''
    Dice Loss for multi-class segmentation.

    Dice Loss measures the overlap between predicted and ground truth segmentation masks.
    It is particularly effective for handling class imbalance in medical image segmentation.

    Dice Coefficient:
        Dice = (2 • |X ∩ Y|) / (|X| + |Y|)
    
    Dice Loss:
        Loss = 1 - Dice

    Attributes:
        num_classes: Number of segmentation classes
        smooth: Smoothing factor to avoid division by zero
        ignore_index: Class index to ignore in loss computation (e.g., background)
    '''

    def __init__(self, config: TrainingConfig):
        '''
        Initialize Dice Loss.

        Args:
            config: Training configuration with loss parameters
        '''
        super().__init__()

        self.num_classes = config.num_classes
        self.smooth = config.smooth
        self.ignore_index = config.ignore_index

        if self.num_classes < 2:
            raise MRIScanException(f"Number of classes must be >= 2, got {self.num_classes}")

        if self.smooth <= 0:
            raise MRIScanException(f"Smoothing factor must be > 0, got {self.smooth}")

        logger.info(f"Initialized Dice Loss with {self.num_classes} classes, smooth={self.smooth}, ignore_index={self.ignore_index}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Compute Dice Loss.

        Args:
            logits: Raw model outputs of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W) with class indices

        Returns:
            Scalar tensor with Dice loss value
        '''
        # Input validation
        if logits.dim() != 4:
            raise MRIScanException(f"Expected 4D logits (B, C, H, W), got shape {logits.shape}")

        if targets.dim() != 3:
            raise MRIScanException(f"Expected 3D targets (B, H, W), got shape {targets.shape}")

        if logits.shape[0] != targets.shape[0]:
            raise MRIScanException(f"Batch size mismatch: logits has {logits.shape[0]} vs targets {targets.shape[0]}")

        if logits.shape[2:] != targets.shape[1:]:
            raise MRIScanException(f"Spatial dimensions mismatch: logits has {logits.shape[2:]} vs targets {targets.shape[1:]}")

        batch_size = logits.shape[0]

        # Apply softmax to get probabilities: (B, C, H, W)
        probs = F.softmax(logits, dim=1)
        
        # One-hot encode targets: (B, H, W) -> (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)          # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()               # (B, C, H, W)

        # Flatten spatial dimensions: (B, C, H * W)
        probs = probs.view(batch_size, self.num_classes, -1)
        targets_one_hot = targets_one_hot.view(batch_size, self.num_classes, -1)

        # Compute Dice coefficient per class
        intersection = (probs * targets_one_hot).sum(dim=2)                         # (B, C)
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)                       # (B, C)

        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)     # (B, C)

        # Handle ignore_index
        if self.ignore_index is not None:
            # Create mask for valid classes
            mask = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device)
            mask[self.ignore_index] = False

            # Select only valid class Dice coefficients and compute mean
            valid_dice = dice_coeff[:, mask]                                        # (B, num_valid_classes)
            dice_loss = 1.0 - valid_dice.mean()

        else:
            # Average over all classes and batch
            dice_loss = 1.0 - dice_coeff.mean()

        return dice_loss


class CrossEntropyLoss(nn.Module):
    '''
    Cross-Entropy Loss for multi-class segmentation.

    Wrapper around PyTorch's CrossEntropyLoss with optional class weights
    and label smoothing for better generalization.

    Attributes:
        weight: Optional class weights for imbalanced classes
        ignore_index: Class index to ignore in loss computation (e.g., background)
        label_smoothing: Label smoothing factor (0.0 for no smoothing)
    '''

    def __init__(self, config: TrainingConfig, class_weights: Optional[torch.Tensor] = None):
        '''
        Initialize Cross-Entropy Loss.
        
        Args:
            config: Training configuration with loss parameters
            class_weights: Optional tensor of class weights of shape (C, ) for handling class imbalance
        '''
        super().__init__()

        self.ignore_index = config.ignore_index if config.ignore_index is not None else -100
        self.label_smoothing = config.label_smoothing
        self.class_weights = class_weights

        if self.label_smoothing < 0.0 or self.label_smoothing > 1.0:
            raise MRIScanException(f"Label smoothing factor must be in range [0.0, 1.0], got {self.label_smoothing}")

        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing
        )

        logger.info(f"Initialized Cross-Entropy Loss: ignore_index={self.ignore_index}, label_smoothing={self.label_smoothing}, weighted={self.class_weights is not None}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Compute Cross-Entropy Loss.

        Args:
            logits: Raw model outputs of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W) with class indices

        Returns:
            Scaler tensor with Cross-Entropy loss value
        '''
        # Input validation
        if logits.dim() != 4:
            raise MRIScanException(f"Expected 4D logits (B, C, H, W), got shape {logits.shape}")
            
        if targets.dim() != 3:
            raise MRIScanException(f"Expected 3D targets (B, H, W), got shape {targets.shape}")

        if logits.shape[0] != targets.shape[0]:
            raise MRIScanException(f"Batch size mismatch: logits has {logits.shape[0]} vs targets {targets.shape[0]}")

        if logits.shape[2:] != targets.shape[1:]:
            raise MRIScanException(f"Spatial dimensions mismatch: logits has {logits.shape[2:]} vs targets {targets.shape[1:]}")

        return self.criterion(logits, targets)


class CombinedLoss(nn.Module):
    '''
    Combined Loss: Weighted sum of Dice and Cross-Entropy losses.

    This loss function combines the benefits of both:
    - Dice Loss: Optimizes for overlap, handles class imbalance well
    - Cross-Entropy Loss: Provides smooth gradients, good for pixel-wise accuracy

    Loss = dice_weight * DiceLoss + ce_weight * CrossEntropyLoss

    Attributes:
        dice_loss: DiceLoss module
        ce_loss: CrossEntropyLoss module
        dice_weight: Weight for Dice loss component
        ce_weight: Weight for Cross-Entropy loss component
    '''

    def __init__(
        self,
        config: TrainingConfig,
        class_weights: Optional[torch.Tensor] = None
    ):
        '''
        Initialize Combined Loss.

        Args:
            config: Training configuration with loss parameters
            class_weights: Optional tensor of class weights of shape (C, ) for handling class imbalance
        '''
        super().__init__()

        self.dice_weight = config.dice_weight
        self.ce_weight = config.ce_weight

        if self.dice_weight < 0 or self.ce_weight < 0:
            raise MRIScanException(f"Loss weights must be non-negative, got dice_weight={self.dice_weight}, ce_weight={self.ce_weight}")

        if self.dice_weight == 0 and self.ce_weight == 0:
            raise MRIScanException(f"At least one loss weight must be > 0, got dice_weight={self.dice_weight}, ce_weight={self.ce_weight}")

        # Initialize loss components
        self.dice_loss = DiceLoss(config=config)
        self.ce_loss = CrossEntropyLoss(config=config, class_weights=class_weights)

        logger.info(f"Initialized Combined Loss: dice_weight={self.dice_weight}, ce_weight={self.ce_weight}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        '''
        Compute Combined Loss.

        Args:
            logits: Raw model outputs of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W) with class indices

        Returns:
            Scaler tensor with Combined loss value
        '''
        dice_loss_val = self.dice_loss(logits, targets) if self.dice_weight > 0 else 0.0
        ce_loss_val = self.ce_loss(logits, targets) if self.ce_weight > 0 else 0.0

        combined_loss = self.dice_weight * dice_loss_val + self.ce_weight * ce_loss_val

        return combined_loss


def build_loss(config: TrainingConfig, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    '''
    Factory function to build loss function based on configuration.

    Args:
        config: Training configuration with loss parameters
        class_weights: Optional tensor of class weights of shape (C, ) for handling class imbalance

    Returns:
        Appropriate loss function based on configuration
    '''
    loss_type = config.loss_type.lower()
    
    if loss_type == 'dice':
        loss_fn = DiceLoss(config=config)
        logger.info(f"Built Dice loss for {config.num_classes} classes")

    elif loss_type == 'ce':
        loss_fn = CrossEntropyLoss(config=config, class_weights=class_weights)
        logger.info(f"Built Cross-Entropy loss for {config.num_classes} classes")

    elif loss_type == 'combined':
        loss_fn = CombinedLoss(config=config, class_weights=class_weights)
        logger.info(f"Built Combined loss (Dice: {config.dice_weight}, CE: {config.ce_weight}) for {config.num_classes} classes")

    else:
        raise MRIScanException(f"Invalid loss type: {config.loss_type}. Must be one of 'dice', 'ce', 'combined'")

    return loss_fn