from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path


@dataclass
class PatchConfig:
    '''Patch configuration for Vision Transformer.'''
    size: Tuple[int, int] = (16, 16)
    grid: Optional[Tuple[int, int]] = None


@dataclass
class TransformerConfig:
    '''Transformer encoder configuration.'''
    num_heads: int = 12
    num_layers: int = 12
    mlp_dim: int = 3072
    attention_dropout_rate: float = 0.0
    dropout_rate: float = 0.1


@dataclass
class ResNetConfig:
    '''ResNet backbone configuration.'''
    num_layers: Tuple[int, int, int] = (3, 4, 9)
    width_factor: int = 1


@dataclass
class ModelConfig:
    '''Model architecture configuration.'''

    # Architecture type
    model_name: str = 'R50-ViT-B16'

    # Patch settings
    patches: PatchConfig = field(default_factory=PatchConfig)
    patch_size: int = 16

    # Transformer settings
    hidden_size: int = 768
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    # ResNet settings (for hybrid model)
    resnet: Optional[ResNetConfig] = field(default_factory=ResNetConfig)

    # Decoder settings
    decoder_channels: Tuple[int, int, int, int] = (256, 128, 64, 16)
    skip_channels: List[int] = field(default_factory=lambda: [512, 256, 64, 16])
    n_skip: int = 3

    # Output settings
    n_classes: int = 2
    activation: str = 'softmax'

    # Pretrained weights
    pretrained_path: Optional[str] = None
    resnet_pretrained_path: Optional[str] = None

    # Other settings
    classifier: str = 'seg'
    representation_size: Optional[int] = None

    def __post_init__(self):
        '''Post-initialization processing.'''
        # Fallback to PatchConfig if no config is provided
        if not isinstance(self.patches, PatchConfig):
            self.patches = PatchConfig(**self.patches) if isinstance(self.patches, dict) else PatchConfig()

        # Fallback to TransformerConfig if no config is provided
        if not isinstance(self.transformer, TransformerConfig):
            self.transformer = TransformerConfig(**self.transformer) if isinstance(self.transformer, dict) else TransformerConfig()

        # Fallback to ResNetConfig if no config is provided
        if not isinstance(self.resnet, ResNetConfig):
            self.resnet = ResNetConfig(**self.resnet) if isinstance(self.resnet, dict) else ResNetConfig()

        # Set grid for hybrid model
        if self.model_name.startswith('R50'):
            if self.patches.grid is None:
                self.patches.grid = (16, 16)


@dataclass
class DataConfig:
    '''Data pipeline configuration.'''

    # Data paths
    data_dir: str = 'data'
    images_dir: str = 'data/images'
    masks_dir: str = 'data/masks'

    # Dataset splits
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15

    # Data properties
    image_size: int = 224
    num_classes: int = 9

    # Data loading
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = False

    # Augmentation
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotation_limit: int = 15

    # Normalization
    normalize: bool = True
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None

    # Random seed
    seed: int = 42


@dataclass
class OptimizerConfig:
    '''Optimizer configuration.'''
    name: str = 'adamw'
    lr: float = 1e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    '''Learning rate scheduler configuration.'''
    name: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # For step scheduler
    step_size: int = 30
    gamma: float = 0.1


@dataclass
class TrainingConfig:
    '''Training configuration.'''
    # Training duration
    epochs: int = 150
    early_stopping_patience: int = 20

    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Loss function
    loss_type: str = 'combined'     # 'dice', 'ce', 'combined'
    dice_weight: float = 0.5
    ce_weight: float = 0.5

    # Training settings
    use_amp: bool = True
    gradient_clip_val: float = 1.0

    # Checkpointing
    save_every_n_epochs: int = 10
    save_best: bool = True
    save_last: bool = True

    # Validation
    val_every_n_epochs: int = 1

    # Paths
    checkpoint_dir: str = 'experiments/checkpoints'
    log_dir: str = 'logs'
    tensorboard_dir: str = 'experiments/runs'

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    # Random seed
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        '''Post-initialization processing.'''
        # Fallback to OptimizerConfig if no config is provided
        if not isinstance(self.optimizer, OptimizerConfig):
            self.optimizer = OptimizerConfig(**self.optimizer) if isinstance(self.optimizer, dict) else OptimizerConfig()

        # Fallback to SchedulerConfig if no config is provided
        if not isinstance(self.scheduler, SchedulerConfig):
            self.scheduler = SchedulerConfig(**self.scheduler) if isinstance(self.scheduler, dict) else SchedulerConfig()


@dataclass
class Config:
    '''Main configuration combining all sub-configs.'''
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        '''Post-initialization processing.'''
        # Fallback to ModelConfig if no config is provided
        if not isinstance(self.model, ModelConfig):
            self.model = ModelConfig(**self.model) if isinstance(self.model, dict) else ModelConfig()

        # Fallback to DataConfig if no config is provided
        if not isinstance(self.data, DataConfig):
            self.data = DataConfig(**self.data) if isinstance(self.data, dict) else DataConfig()

        # Fallback to TrainingConfig if no config is provided
        if not isinstance(self.training, TrainingConfig):
            self.training = TrainingConfig(**self.training) if isinstance(self.training, dict) else TrainingConfig()


def get_r50_vit_b16_config() -> ModelConfig:
    '''Get R50-ViT-B16 model configuration.'''
    config = ModelConfig(
        model_name='R50-ViT-B16',
        hidden_size=768,
        patch_size=16,
        patches=PatchConfig(size=(16, 16), grid=(16, 16)),
        transformer=TransformerConfig(
            num_heads=12,
            num_layers=12,
            mlp_dim=3072,
            attention_dropout_rate=0.0,
            dropout_rate=0.1
        ),
        resnet=ResNetConfig(
            num_layers=(3, 4, 9),
            width_factor=1
        ),
        decoder_channels=(256, 128, 64, 16),
        skip_channels=[512, 256, 64, 16],
        n_skip=3,
        n_classes=2,
        activation='softmax',
        classifier='seg'
    )
    return config