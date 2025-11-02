import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from shisue.training.losses import build_loss
from shisue.training.metrics import MetricsTracker
from shisue.utils.config import TrainingConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class Trainer:
    '''
    Trainer class for TransUNet model with comprehensive training features.
    
    Features:
    - Automatic device selection (MPS → CUDA → CPU)
    - Mixed precision training (AMP)
    - Checkpointing (best, last, periodic)
    - Early stopping
    - TensorBoard logging
    - Resume from checkpoint
    - Polynomial learning rate scheduling

    Attributes:
        model: TransUNet model
        config: Training configuration
        device: Training device (mps, cuda, cpu)
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        metrics_tracker: Metrics tracking utility
        scaler: Gradient scaler for AMP
        writer: TensorBoard writer
        best_metric: Best validation metric value
        current_epoch: Current training epoch
        early_stopping_counter: Counter for early stopping
    '''
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
    ):
        '''
        Initialize Trainer.

        Args:
            model: TransUNet model
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            class_weights: Optional tensor of class weights of shape (C, ) for loss function
        '''
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Set device with priority: MPS → CUDA → CPU
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")

        # Move model to device
        self.model = self.model.to(self.device)

        # Move class weights to device if provided
        if class_weights is not None:
            class_weights = class_weights.to(self.device)

        # Initialize loss function
        self.criterion = build_loss(config=config, class_weights=class_weights)

        # Initialize optimizer
        self.optimizer = self._build_optimizer()

        # Initialize learning rate scheduler
        self.scheduler = self._build_scheduler()

        # Initialize gradient scaler for AMP
        # Note: AMP support varies by device
        # - CUDA: Full support
        # - MPS: Limited/experimental support
        # - CPU: Not beneficial
        self.use_amp = config.use_amp and self.device.type in ['cuda', 'mps']
        if self.use_amp:
            if self.device.type == 'cuda':
                # Full AMP support on CUDA
                self.scaler = torch.amp.GradScaler('cuda')
                logger.info(f"Automatic Mixed Precision (AMP) enabled for CUDA")
            elif self.device.type == 'mps':
                # Experimental AMP support on MPS
                try:
                    self.scaler = torch.amp.GradScaler('mps')
                    logger.info(f"Automatic Mixed Precision (AMP) enabled for MPS (experimental)")
                
                except Exception as e:
                    logger.warning(f"Failed to enable AMP on MPS: {e}. Falling back to FP32")
                    
                    self.use_amp = False
                    self.scaler = None

        else:
            self.scaler = None
            if config.use_amp:
                logger.warning(f"AMP is not supported on {self.device.type}. Falling back to FP32")

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(config=config)

        # Setup directories
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_dir = Path(config.tensorboard_dir)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.early_stopping_counter = 0

        # Set random seed for reproducibility
        self._set_seed(config.seed)

        if config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info(f"Deterministic mode enabled for reproducibility")

        logger.info(f"Trainer initialized successfully")

    def _get_device(self) -> torch.device:
        '''
        Get device with priority: MPS → CUDA → CPU

        Returns:
            torch.device object
        '''
        # Priority 1: MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            return torch.device('mps')

        # Priority 2: CUDA (NVIDIA GPUs)
        elif torch.cuda.is_available():
            return torch.device('cuda')

        # Priority 3: CPU (fallback)
        else:
            logger.warning(f"No GPU found, using CPU. Performance may be limited.")
            return torch.device('cpu')

    def _set_seed(self, seed: int):
        '''
        Set random seed for reproducibility.

        Args:
            seed: Random seed value
        '''
        # Python random module
        random.seed(seed)

        # NumPy random module
        np.random.seed(seed)

        # Pytorch CPU
        torch.manual_seed(seed)

        # PyTorch CUDA
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # PyTorch MPS
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

        logger.info(f"Set random seed to {seed} for all libraries (random, numpy, torch) for reproducibility")

    def _build_optimizer(self) -> Optimizer:
        '''
        Build optimizer based on configuration.

        Returns:
            Optimizer instance
        '''
        opt_config = self.config.optimizer

        if opt_config.name.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config.lr,
                momentum=opt_config.momentum,
                weight_decay=opt_config.weight_decay,
            )
            logger.info(f"Built SGD optimizer: lr={opt_config.lr}, momentum={opt_config.momentum}, weight_decay={opt_config.weight_decay}")

        elif opt_config.name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config.lr,
                betas=opt_config.betas
            )
            logger.info(f"Built Adam optimizer: lr={opt_config.lr}, betas={opt_config.betas}")

        elif opt_config.name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas
            )
            logger.info(f"Built AdamW optimizer: lr={opt_config.lr}, weight_decay={opt_config.weight_decay}, betas={opt_config.betas}")

        else:
            raise MRIScanException(f"Unsupported optimizer: {opt_config.name}. Must be one of 'sgd', 'adam', 'adamw'")

        return optimizer

    def _build_scheduler(self) -> LRScheduler:
        '''
        Build learning rate scheduler based on configuration.

        Returns:
            Learning rate scheduler instance
        '''
        sched_config = self.config.scheduler

        if sched_config.name.lower() == 'polynomial':
            # Polynomial decay: lr = initial_lr * (1 - epoch / max_epochs) ^ power
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=self.config.epochs,
                power=sched_config.power
            )
            logger.info(f"Built Polynomial LR scheduler: power={sched_config.power}")

        elif sched_config.name.lower() == 'cosine':
            # Cosine decay: lr = initial_lr * (1 + cos(pi * epoch / max_epochs)) / 2
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
            )
            logger.info(f"Built Cosine Annealing LR scheduler: T_max={self.config.epochs}")

        elif sched_config.name.lower() == 'step':
            # Step decay: lr = initial_lr * (gamma ^ (epoch // step_size))
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.step_size,
                gamma=sched_config.gamma
            )
            logger.info(f"Built Step LR scheduler: step_size={sched_config.step_size}, gamma={sched_config.gamma}")

        else:
            raise MRIScanException(f"Unsupported scheduler: {sched_config.name}. Must be one of 'polynomial', 'cosine', 'step'")

        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        '''
        Train the model for one epoch.

        Returns:
            Dictionary with training metrics
        '''
        self.model.train()
        self.metrics_tracker.reset()

        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.config.epochs} [Training]"
        )

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with AMP
            if self.use_amp:
                # Use device-agnostic autocast
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Forward pass without AMP
                logits = self.model(images)
                loss = self.criterion(logits, masks)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )

                # Optimizer step
                self.optimizer.step()

            # Update metrics
            with torch.no_grad():
                self.metrics_tracker.update(logits, masks)

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    'loss': f"{loss.item():.4f}"
                }
            )

        # Compute average metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.metrics_tracker.compute()
        metrics['loss'] = avg_loss

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        '''
        Validate the model on validation set.

        Returns:
            Dictionary with validation metrics
        '''
        self.model.eval()
        self.metrics_tracker.reset()

        epoch_loss = 0.0
        num_batches = len(self.val_loader)

        # Progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch}/{self.config.epochs} [Validation]"
        )

        for batch in pbar:
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)

            # Forward pass
            logits = self.model(images)
            loss = self.criterion(logits, masks)

            # Update metrics
            self.metrics_tracker.update(logits, masks)

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar
            pbar.set_postfix(
                {
                    'loss': f"{loss.item():.4f}"
                }
            )

        # Compute average metrics
        avg_loss = epoch_loss / num_batches
        metrics = self.metrics_tracker.compute()
        metrics['loss'] = avg_loss

        return metrics

    def save_checkpoint(self, checkpoint_path: Path, metrics: Optional[Dict[str, float]] = None):
        '''
        Save model checkpoints.

        Args:
            checkpoint_path: Path to save the checkpoint
            metrics: Optional metrics to save with checkpoint
        '''
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
            'metrics': metrics or {}
        }

        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        '''
        Load model checkpoint.

        Args:
            checkpoint_path: Path to load the checkpoint
        '''
        if not checkpoint_path.exists():
            raise MRIScanException(f"Checkpoint file not found at {checkpoint_path}")

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.best_metric = checkpoint['best_metric']
            
            # Load scaler state if using AMP
            if self.use_amp and self.scaler is not None and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        except Exception as e:
            raise MRIScanException(f"Failed to load checkpoint states from {checkpoint_path}", details=str(e))

        logger.info(f"Checkpoint loaded from {checkpoint_path}: epoch={self.current_epoch}, best_metric={self.best_metric:.4f}")

    def train(self):
        '''Main training loop with validation, checkpointing, and early stopping.'''
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info(f"Total epochs: {self.config.epochs}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info("=" * 80)

        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint is not None:
            checkpoint_path = Path(self.config.resume_from_checkpoint)
            try:
                self.load_checkpoint(checkpoint_path)
                logger.info(f"Successfully resumed from checkpoint: {checkpoint_path}")

            except Exception as e:
                logger.error(f"Failed to load checkpoint from {checkpoint_path}. Cannot resume training. Error: {str(e)}")
                raise MRIScanException(f"Checkpoint loading failed", details=f"Path: {checkpoint_path}, Error: {str(e)}") from e

        # Training loop
        for epoch in range(self.current_epoch + 1, self.config.epochs + 1):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Log training metrics
            logger.info(
                f"Epoch {epoch}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train IoU: {train_metrics['mean_iou']:.4f}, "
                f"Train Dice: {train_metrics['mean_dice']:.4f}"
            )

            # TensorBoard logging - training
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Metrics/train_iou', train_metrics['mean_iou'], epoch)
            self.writer.add_scalar('Metrics/train_dice', train_metrics['mean_dice'], epoch)
            self.writer.add_scalar('Metrics/train_pixel_accuracy', train_metrics['pixel_accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Validation
            if epoch % self.config.val_every_n_epochs == 0:
                val_metrics = self.validate()

                # Log validation metrics
                logger.info(
                    f"Epoch {epoch}/{self.config.epochs} - "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val IoU: {val_metrics['mean_iou']:.4f}, "
                    f"Val Dice: {val_metrics['mean_dice']:.4f}"
                )

                # TensorBoard logging - validation
                self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                self.writer.add_scalar('Metrics/val_iou', val_metrics['mean_iou'], epoch)
                self.writer.add_scalar('Metrics/val_dice', val_metrics['mean_dice'], epoch)
                self.writer.add_scalar('Metrics/val_pixel_accuracy', val_metrics['pixel_accuracy'], epoch)

                # Check if this is the best model
                current_metric = val_metrics['mean_dice']   # Use Dice as primary metric

                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.early_stopping_counter = 0

                    # Save best checkpoint
                    if self.config.save_best:
                        best_path = self.checkpoint_dir / 'best_model.pth'
                        self.save_checkpoint(best_path, val_metrics)
                        logger.info(f"New best model saved with Dice: {self.best_metric:.4f}")

                else:
                    self.early_stopping_counter += 1
                    logger.info(f"No improvement for {self.early_stopping_counter} epochs (patience: {self.config.early_stopping_patience})")

            # Save at periodic checkpoints
            if epoch % self.config.save_every_n_epochs == 0:
                periodic_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
                self.save_checkpoint(periodic_path)

            # Save last checkpoint
            if self.config.save_last:
                last_path = self.checkpoint_dir / 'last_model.pth'
                self.save_checkpoint(last_path)

            # Update learning rate
            self.scheduler.step()

            # Early stopping check
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (patience: {self.config.early_stopping_patience})")
                break

        # Training completed
        logger.info("=" * 80)
        logger.info("Training completed successfully")
        logger.info(f"Best validation Dice: {self.best_metric:.4f}")
        logger.info("=" * 80)

        # Close TensorBoard writer
        self.writer.close()