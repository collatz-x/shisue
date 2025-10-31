from typing import Optional

import torch
import torch.nn as nn

from shisue.models.resnet import ResNetV2
from shisue.models.transformer import Transformer
from shisue.models.decoder import DecoderCUP
from shisue.utils.config import ModelConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class TransUNet(nn.Module):
    '''
    TransUNet: Hybrid CNN-Transformer U-Net architecture for medical image segmentation.

    This model combines the strengths of CNNs (local feature extraction via ResNet)
    and Transformers (global context modeling via ViT) for accurate medical image segmentation.

    Reference:
        Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A. L., & Zhou, Y. (2021, February 8).
        TransUNET: Transformers make strong encoders for medical image segmentation.
        arXiv.org. https://arxiv.org/abs/2102.04306

    Architecture:
    1. ResNet Backbone: Extracts hierarchical features with skip connections
        - Input: (B, 3, 224, 224)
        - Output: (B, 1024, 14, 14) + skip connections at multiple scales

    2. Transformer Encoder: Captures global dependencies with self-attention
        - Input: (B, 1024, 14, 14)
        - Output: (B, 768, 14, 14)

    3. Decoder with Cascaded Upsampling and Skip Connections: Recovers spatial resolution and combines multi-scale features
        - Input: (B, 768, 14, 14) + skip connections from ResNet
        - Output: (B, n_classes, 224, 224)

    Attributes:
        resnet: ResNet backbone
        transformer: Vision Transformer encoder
        decoder: Cascaded upsampling decoder
        config: Model configuration
    '''

    def __init__(self, config: ModelConfig):
        '''
        Initialize TransUNet model.

        Args:
            config: Model configuration with all architecture parameters
        '''
        super().__init__()

        self.config = config
        self.n_classes = config.n_classes

        # Initialize ResNet backbone
        logger.info(f"Building TransUNet model: {config.model_name}")
        self.resnet = ResNetV2(config)

        # Get ResNet output channels (last stage)
        resnet_out_channels = self.resnet.skip_channels[-1]     # 1024 for ResNet50

        # Initialize Transformer encoder
        self.transformer = Transformer(config=config, in_channels=resnet_out_channels)

        # Initialize Decoder
        self.decoder = DecoderCUP(config=config, in_channels=config.hidden_size)

        logger.info(f"TransUNet initialized successfully with {self.count_parameters():,} parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through TransUNet.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Segmentation logits of shape (B, n_classes, 224, 224)
        '''
        # Input validation
        if x.dim() != 4:
            raise MRIScanException(f"Expected 4D input (B, C, H, W), got {x.dim()}D tensor with shape {x.shape}")

        if x.shape[1] != 3:
            raise MRIScanException(f"Expected 3 input channels (RGB), got {x.shape[1]} channels")

        if x.shape[2] != 224 or x.shape[3] != 224:
            raise MRIScanException(f"Expected input size 224x224, got {x.shape[2]}x{x.shape[3]}")

        # Stage 1: ResNet feature extraction with skip connections
        # resnet_features: (B, 1024, 14, 14)
        # skip_connections: [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14)]
        resnet_features, skip_connections = self.resnet(x)

        # Stage 2: Transformer encoding
        # encoded_features: (B, 196, 768) - not used directly
        # spatial_features: (B, 768, 14, 14) - used for decoder
        encoded_features, spatial_features = self.transformer(resnet_features)

        # Stage 3: Decoder with skip connections
        # logits: (B, n_classes, 224, 224)
        logits = self.decoder(spatial_features, skip_connections)

        return logits

    def load_pretrained_resnet(self, pretrained_path: str):
        '''
        Load pretrained ResNet weights from ImageNet.

        Args:
            pretrained_path: Path to pretrained ResNet weights
        '''
        logger.info(f"Loading pretrained ResNet weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Filter out keys that don't match target ResNetV2 pattern for transfer learning
            model_dict = self.resnet.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            self.resnet.load_state_dict(model_dict)

            logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained ResNet")
        
        except Exception as e:
            raise MRIScanException(f"Failed to load pretrained ResNet weights from {pretrained_path}", details=str(e))

    def load_pretrained_transformer(self, pretrained_path: str):
        '''
        Load pretrained Vision Transformer weights from ImageNet-21k.

        Args:
            pretrained_path: Path to pretrained ViT weights
        '''
        logger.info(f"Loading pretrained Transformer weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Filter out keys that don't match target Transformer pattern for transfer learning
            model_dict = self.transformer.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}

            model_dict.update(pretrained_dict)
            self.transformer.load_state_dict(model_dict)

            logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained Transformer")
        
        except Exception as e:
            raise MRIScanException(f"Failed to load pretrained Transformer weights from {pretrained_path}", details=str(e))

    def count_parameters(self) -> int:
        '''
        Count total number of trainable parameters.

        Returns:
            Total number of trainable parameters
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        '''Freeze ResNet backbone parameters for transfer learning.'''
        logger.info(f"Freezing ResNet backbone")
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        '''Unfreeze ResNet backbone parameters for training.'''
        logger.info(f"Unfreezing ResNet backbone")
        for param in self.resnet.parameters():
            param.requires_grad = True


def build_transunet(config: ModelConfig) -> TransUNet:
    '''
    Factory function to build TransUNet model.

    Args:
        config: Model configuration

    Returns:
        TransUNet model instance
    '''
    model = TransUNet(config)

    # Load pretrained weights if specified in config
    if config.resnet_pretrained_path is not None:
        model.load_pretrained_resnet(config.resnet_pretrained_path)

    if config.transformer_pretrained_path is not None:
        model.load_pretrained_transformer(config.transformer_pretrained_path)

    return model