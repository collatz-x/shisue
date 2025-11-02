from typing import Dict, Optional, Tuple

import numpy as np
import torch

from shisue.utils.config import TrainingConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class IoU:
    '''
    Intersection over Union (IoU) metric for multi-class segmentation.

    Also known as Jaccard Index, IoU measures the overlap between predicted and ground truth segmentations.

    IoU = TP / (TP + FP + FN)

    For multi-class segmentation, IoU is calculated per class and averaged.

    Attributes:
        num_classes: Number of segmentation classes
        ignore_index: Class index to ignore in metric computation (e.g., background)
        epsilon: Small constant to avoid division by zero
    '''

    def __init__(self, config: TrainingConfig):
        '''
        Initialize IoU metric.

        Args:
            config: Training configuration with metric parameters
        '''
        self.num_classes = config.num_classes
        self.ignore_index = config.ignore_index
        self.epsilon = config.epsilon

        if self.num_classes < 2:
            raise MRIScanException(f"Number of classes must be >= 2, got {self.num_classes}")

        logger.info(f"Initialized IoU metric: num_classes={self.num_classes}, ignore_index={self.ignore_index}, epsilon={self.epsilon}")

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, per_class: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        '''
        Compute IoU metric.

        Args:
            logits: Raw model outputs of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W) with class indices
            per_class: Whether to return per-class IoU values

        Returns:
            Tuple of:
                - Mean IoU (float) across all classes
                - Per-class IoU scores (numpy array of shape (C, )) if per_class=True, else None
        '''
        # Input validation
        if logits.dim() != 4:
            raise MRIScanException(f"Expected 4D logits (B, C, H, W), got shape {logits.shape}")

        if targets.dim() != 3:
            raise MRIScanException(f"Expected 3D targets (B, H, W), got shape {targets.shape}")

        if logits.shape[0] != targets.shape[0]:
            raise MRIScanException(f"Batch size mismatch: logits has {logits.shape[0]} vs targets {targets.shape[0]}")
            
        # Get predictions: (B, H, W)
        preds = torch.argmax(logits, dim=1)

        # Compute IoU per class
        iou_per_class = np.zeros(self.num_classes, dtype=np.float32)

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            # Binary masks for current class
            pred_mask = (preds == cls)
            target_mask = (targets == cls)

            # Compute intersection and union
            intersection = (pred_mask & target_mask).sum().item()
            union = (pred_mask | target_mask).sum().item()

            # Compute IoU for current class
            if union > 0:
                iou_per_class[cls] = intersection / (union + self.epsilon)
            else:
                # Class not present in batch
                iou_per_class[cls] = np.nan

        # Compute mean IoU (ignoring NaN values and ignore_index)
        valid_mask = ~np.isnan(iou_per_class)
        if self.ignore_index is not None:
            valid_mask[self.ignore_index] = False

        if valid_mask.sum() > 0:
            mean_iou = np.nanmean(iou_per_class[valid_mask])
        else:
            mean_iou = 0.0

        if per_class:
            return mean_iou, iou_per_class
        else:
            return mean_iou, None


class DiceScore:
    '''
    Dice Score metric for multi-class segmentation.

    Also known as F1 score or Sørensen–Dice coefficient,
    Dice Score measures the similarity between predicted and ground truth segmentations.

    Dice Score = (2 * TP) / (2 * TP + FP + FN)

    For multi-class segmentation, Dice Score is calculated per class and averaged.

    Attributes:
        num_classes: Number of segmentation classes
        ignore_index: Class index to ignore in metric computation (e.g., background)
        epsilon: Small constant to avoid division by zero
    '''

    def __init__(self, config: TrainingConfig):
        '''
        Initialize Dice Score metric.

        Args:
            config: Training configuration with metric parameters
        '''
        self.num_classes = config.num_classes
        self.ignore_index = config.ignore_index
        self.epsilon = config.epsilon

        if self.num_classes < 2:
            raise MRIScanException(f"Number of classes must be >= 2, got {self.num_classes}")

        logger.info(f"Initialized Dice Score metric: num_classes={self.num_classes}, ignore_index={self.ignore_index}, epsilon={self.epsilon}")

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor, per_class: bool = False) -> Tuple[float, Optional[np.ndarray]]:
        '''
        Compute Dice Score metric.

        Args:
            logits: Raw model outputs of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W) with class indices
            per_class: Whether to return per-class Dice Score values

        Returns:
            Tuple of:
                - Mean Dice Score (float) across all classes
                - Per-class Dice Score scores (numpy array of shape (C, )) if per_class=True, else None
        '''
        # Input validation
        if logits.dim() != 4:
            raise MRIScanException(f"Expected 4D logits (B, C, H, W), got shape {logits.shape}")

        if targets.dim() != 3:
            raise MRIScanException(f"Expected 3D targets (B, H, W), got shape {targets.shape}")

        if logits.shape[0] != targets.shape[0]:
            raise MRIScanException(f"Batch size mismatch: logits has {logits.shape[0]} vs targets {targets.shape[0]}")
            
        # Get predictions: (B, H, W)
        preds = torch.argmax(logits, dim=1)
        
        # Compute Dice Score per class
        dice_per_class = np.zeros(self.num_classes, dtype=np.float32)

        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue

            # Binary masks for current class
            pred_mask = (preds == cls)
            target_mask = (targets == cls)
            
            # Compute intersection and cardinalities
            intersection = (pred_mask & target_mask).sum().item()
            pred_sum = pred_mask.sum().item()
            target_sum = target_mask.sum().item()

            # Compute mean Dice Score for current class
            denominator = pred_sum + target_sum
            if denominator > 0:
                dice_per_class[cls] = (2.0 * intersection) / (denominator + self.epsilon)
            else:
                # Class not present in batch
                dice_per_class[cls] = np.nan

        # Compute mean Dice Score (ignoring NaN values and ignore_index)
        valid_mask = ~np.isnan(dice_per_class)
        if self.ignore_index is not None:
            valid_mask[self.ignore_index] = False

        if valid_mask.sum() > 0:
            mean_dice = np.nanmean(dice_per_class[valid_mask])
        else:
            mean_dice = 0.0

        if per_class:
            return mean_dice, dice_per_class
        else:
            return mean_dice, None


def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None) -> np.ndarray:
    '''
    Compute confusion matrix for multi-class segmentation.

    Args:
        preds: Predicted class indices of shape (B, H, W)
        targets: Ground truth class indices of shape (B, H, W)
        num_classes: Number of segmentation classes
        ignore_index: Class index to ignore in computation (e.g., background)

    Returns:
        Confusion matrix of shape (C, C) where entry [i, j] represents
        the number of pixels of class i predicted as class j.
    '''
    # Flatten tensors
    preds = preds.flatten().cpu().numpy()
    targets = targets.flatten().cpu().numpy()

    # Remove ignored indices
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    # Compute confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            confusion_matrix[true_class, pred_class] = np.sum(
                (targets == true_class) & (preds == pred_class)
            )

    return confusion_matrix


def compute_pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: Optional[int] = None) -> float:
    '''
    Compute pixel-wise accuracy.

    Args:
        logits: Raw model outputs of shape (B, C, H, W)
        targets: Ground truth masks of shape (B, H, W) with class indices
        ignore_index: Class index to ignore in computation (e.g., background)

    Returns:
        Pixel-wise accuracy as float
    '''
    # Get predictions: (B, H, W)
    preds = torch.argmax(logits, dim=1)

    # Create mask for valid pixels
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        correct = (preds[valid_mask] == targets[valid_mask]).sum().item()
        total = valid_mask.sum().item()
    else:
        correct = (preds == targets).sum().item()
        total = preds.numel()

    if total == 0:
        return 0.0

    accuracy = correct / total
    return accuracy


def compute_class_accuracy(logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None) -> np.ndarray:
    '''
    Compute per-class accuracy.

    For each class, computes the accuracy of pixels belonging to that class.

    Args:
        logits: Raw model outputs of shape (B, C, H, W)
        targets: Ground truth masks of shape (B, H, W) with class indices
        num_classes: Number of segmentation classes
        ignore_index: Class index to ignore in computation (e.g., background)

    Returns:
        Per-class accuracy as numpy array of shape (C, )
    '''
    # Get predictions: (B, H, W)
    preds = torch.argmax(logits, dim=1)

    class_accuracy = np.zeros(num_classes, dtype=np.float32)

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        # Mask for pixels of this class
        class_mask = targets == cls

        if class_mask.sum().item() > 0:
            correct = (preds[class_mask] == targets[class_mask]).sum().item()
            total = class_mask.sum().item()
            class_accuracy[cls] = correct / total
        else:
            # Class not present in batch
            class_accuracy[cls] = np.nan

    return class_accuracy


class MetricsTracker:
    '''
    Utility class to track and aggregate metrics across batches.

    This class maintains running statistics for multiple metrics and
    provides methods to update, compute averages, and reset.

    Attributes:
        config: Training configuration
        iou_metric: IoU metric instance
        dice_metric: DiceScore metric instance
        running_metrics: Dictionary storing accumulated metric values
        num_samples: Number of samples processed
    '''

    def __init__(self, config: TrainingConfig):
        '''
        Initialize metrics tracker.

        Args:
            config: Training configuration with metric parameters
        '''
        self.config = config
        self.num_classes = config.num_classes
        self.ignore_index = config.ignore_index

        # Initialize metric objects
        self.iou_metric = IoU(config=config)
        self.dice_metric = DiceScore(config=config)

        # Initialize running metrics for the next period
        self.reset()

        logger.info(f"Initialized MetricsTracker for {self.num_classes} classes")

    def reset(self):
        '''Reset all running metrics.'''
        self.running_metrics = {
            'iou': 0.0,
            'dice': 0.0,
            'pixel_accuracy': 0.0,
            'iou_per_class': np.zeros(self.num_classes, dtype=np.float32),
            'dice_per_class': np.zeros(self.num_classes, dtype=np.float32)
        }
        self.num_samples = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        '''
        Update metrics with a new batch.

        Args:
            logits: Raw model outputs of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W) with class indices
        '''
        batch_size = logits.shape[0]

        # Compute metrics for current batch
        mean_iou, iou_per_class = self.iou_metric(logits, targets, per_class=True)
        mean_dice, dice_per_class = self.dice_metric(logits, targets, per_class=True)
        pixel_accuracy = compute_pixel_accuracy(logits, targets, ignore_index=self.ignore_index)

        # Update running metrics
        self.running_metrics['iou'] += mean_iou * batch_size
        self.running_metrics['dice'] += mean_dice * batch_size
        self.running_metrics['pixel_accuracy'] += pixel_accuracy * batch_size

        # Update per-class metrics (ignore NaN values)
        valid_iou = ~np.isnan(iou_per_class)
        valid_dice = ~np.isnan(dice_per_class)

        self.running_metrics['iou_per_class'][valid_iou] += iou_per_class[valid_iou] * batch_size
        self.running_metrics['dice_per_class'][valid_dice] += dice_per_class[valid_dice] * batch_size

        # Increment sample count
        self.num_samples += batch_size

    def compute(self) -> Dict[str, float]:
        '''
        Compute average metrics across all batches.

        Returns:
            Dictionary with metric names and average values
        '''
        if self.num_samples == 0:
            logger.warning("No samples processed, returning zero metrics")
            return {
                'iou': 0.0,
                'dice': 0.0,
                'pixel_accuracy': 0.0
            }

        metrics = {
            'mean_iou': self.running_metrics['iou'] / self.num_samples,
            'mean_dice': self.running_metrics['dice'] / self.num_samples,
            'pixel_accuracy': self.running_metrics['pixel_accuracy'] / self.num_samples
        }

        return metrics

    def compute_per_class(self) -> Dict[str, np.ndarray]:
        '''
        Compute per-class metrics.

        Returns:
            Dictionary with per-class metric arrays
        '''
        if self.num_samples == 0:
            logger.warning("No samples processed, returning zero per-class metrics")
            return {
                'iou_per_class': np.zeros(self.num_classes),
                'dice_per_class': np.zeros(self.num_classes)
            }

        per_class_metrics = {
            'iou_per_class': self.running_metrics['iou_per_class'] / self.num_samples,
            'dice_per_class': self.running_metrics['dice_per_class'] / self.num_samples
        }

        return per_class_metrics