from typing import Tuple

import torch
import torch.nn as nn

from shisue.models.components import (
    PatchEmbeddings,
    PositionalEmbeddings,
    TransformerBlock
)
from shisue.utils.config import ModelConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class Transformer(nn.Module):
    '''
    Vision Transformer (ViT) encoder for processing ResNet feature maps in the TransUNet hybrid architecture.

    This encoder takes ResNet feature maps (14x14x1024),
    converts them to patch embeddings, add the learnt positional embeddings,
    and processes them through multiple transformer blocks with self-attention.

    Reference:
        Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020, October 22).
        An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
        arXiv.org. https://arxiv.org/abs/2010.11929

        Architecture:
        1. Patch Embedding: Convert 14x14x1024 feature map to 196x768 embeddings
        2. Positional Embedding: Add learnt positional embeddings
        3. Transformer Blocks: 12 layers of self-attention + MLP
        4. Layer Norm: Final normalization

        For TransUNet, 1x1 patch embedding is used on the 14x14 ResNet feature maps instead of 16x16 patch embeddings in standard ViT-B/16.
        ResNet performs the initial downsampling and local feature extraction, and the ViT treats each 1x1 spatial location as a patch.

        Attributes:
            patch_embeddings: Converts feature maps to patch embeddings
            positional_embeddings: Adds positional information
            transformer_blocks: Stack of transformer encoder blocks
            norm: Final normalization layer
    '''

    def __init__(self, config: ModelConfig, in_channels: int = 1024):
        '''
        Initialize Vision Transformer encoder.

        Args:
            config: Model configuration with transformer parameters
            in_channels: Number of channels from ResNet output (1024)
        '''
        super().__init__()

        if config.transformer is None:
            raise MRIScanException("Transformer configuration is required for hybrid model.")

        self.hidden_size = config.hidden_size
        self.num_layers = config.transformer.num_layers

        # Get patch configuration from config
        patch_size = config.patches.size
        grid_size = config.patches.grid
        num_patches = grid_size[0] * grid_size[1]

        # Patch embedding: project from in_channels to hidden_size
        # Input: (B, 1024, 14, 14) -> Output: (B, 196, 768)
        self.patch_embeddings = PatchEmbeddings(
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size
        )

        # Positional embeddings for spatial positions
        self.positional_embeddings = PositionalEmbeddings(
            num_patches=num_patches,
            hidden_size=self.hidden_size
        )

        # Dropout after embeddings
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

        # Stack of transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=config.transformer.num_heads,
                mlp_dim=config.transformer.mlp_dim,
                attention_dropout_rate=config.transformer.attention_dropout_rate,
                dropout_rate=config.transformer.dropout_rate
            )
            for _ in range(self.num_layers)
        ])

        # Final layer normalization
        self.norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

        # Store grid size for reshaping
        self.grid_size = grid_size

        logger.info(f"Initialized Transformer encoder with {self.num_layers} layers, hidden_size={self.hidden_size}, num_patches={num_patches}, grid_size={self.grid_size}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass through Transformer encoder.

        Args:
            x: ResNet feature maps of shape (B, 1024, 14, 14)

        Returns:
            Tuple of:
                - Encoded features of shape (B, 196, 768)
                - Spatial feature map of shape (B, 768, 14, 14) for decoder
        '''
        batch_size = x.shape[0]

        # Convert to patch embeddings: (B, 1024, 14, 14) -> (B, 196, 768)
        x = self.patch_embeddings(x)

        # Add positional embeddings
        x = self.positional_embeddings(x)
        x = self.dropout(x)

        # Process through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)    # (B, 196, 768)

        # Reshape back to spatial format for decoder
        # (B, 196, 768) -> (B, 768, 14, 14)
        x_spatial = x.transpose(1, 2).contiguous().view(batch_size, self.hidden_size, self.grid_size[0], self.grid_size[1])

        return x, x_spatial


class HybridEncoder(nn.Module):
    '''
    Hybrid encoder combining ResNet and Transformer.

    This is a convenience wrapper that combines ResNet backbone with Transformer encoder,
    handling the feature extraction and transformation.

    Attributes:
        resnet: ResNet backbone for initial feature extraction
        transformer: Vision Transformer for global context
    '''

    def __init__(self, resnet: nn.Module, transformer: nn.Module):
        '''
        Initialize hybrid encoder.

        Args:
            resnet: ResNet backbone module
            transformer: Transformer encoder module
        '''
        super().__init__()

        self.resnet = resnet
        self.transformer = transformer

        logger.info("Initialized hybrid ResNet + Transformer encoder")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
        '''
        Forward pass through hybrid encoder.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Tuple of:
                - Transformer encoded features: (B, 196, 768)
                - Spatial feature map: (B, 768, 14, 14)
                - Skip connections from ResNet: [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14)]
        '''
        # ResNet feature extraction with skip connections
        resnet_features, skip_connections = self.resnet(x)
        # resnet_features: (B, 1024, 14, 14)
        # skip_connections: [(B, 256, 56, 56), (B, 512, 28, 28), (B, 1024, 14, 14)]

        # Transformer feature extraction
        encoded_features, spatial_features = self.transformer(resnet_features)
        # encoded_features: (B, 196, 768)
        # spatial_features: (B, 768, 14, 14)

        return encoded_features, spatial_features, skip_connections