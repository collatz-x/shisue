from typing import List, Optional

import torch
import torch.nn as nn

from shisue.models.components import ConvBlock, UpConvBlock
from shisue.utils.config import ModelConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class DecoderBlock(nn.Module):
    '''
    Decoder block with skip connection fusion and upsampling.

    This block:
    1. Optionally fuses with skip connection from encoder
    2. Applies convolution for feature processing
    3. Upsamples by 2x using transposed convolution

    Attributes:
        skip_conv: Optional 1x1 conv to project skip connection channels
        conv: Convolutional block for feature processing
        upsample: Upsampling block
    '''

    def __init__(self, in_channels: int, out_channels: int, skip_channels: Optional[int] = None, use_skip: bool = True):
        '''
        Initialize decoder block.

        Args:
            in_channels: Number of input channels from previous layer
            out_channels: Number of output channels
            skip_channels: Number of channels in skip connection (if used)
            use_skip: Whether to use skip connection
        '''
        super().__init__()

        self.use_skip = use_skip

        # Project skip connection to match decoder channels
        if use_skip and skip_channels is not None:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, out_channels), num_channels=out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.skip_conv = None

        # Feature processing convolution
        conv_in_channels = in_channels + out_channels if use_skip else in_channels
        self.conv = ConvBlock(
            in_channels=conv_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_weight_std=False,
            activation='relu'
        )

        # Upsampling
        self.upsample = UpConvBlock(
            in_channels=out_channels,
            out_channels=out_channels
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Forward pass through decoder block.

        Args:
            x: Input features from previous decoder layer
            skip: Optional skip connection from encoder (if used)

        Returns:
            Upsampled features with skip connection fused
        '''
        # Fuse skip connection if available
        if self.use_skip and skip is not None:
            # Project skip connection
            skip = self.skip_conv(skip)

            # Concatenate along channel dimension
            x = torch.cat([x, skip], dim=1)

        # Process features
        x = self.conv(x)

        # Upsample
        x = self.upsample(x)

        return x


class DecoderCUP(nn.Module):
    '''
    Cascaded Upsampling Decoder (CUP) for TransUNet.

    This decoder uses cascaded upsampling blocks to progressively upsample transformer features from 14x14 to 224x224,
    incorporating skip connections from ResNet to recover spatial resolution and combine multi-scale features.

    Architecture:
    - Input: Transformer features (B, 768, 14, 14)
    - conv_more: Project to decoder channels (B, 256, 14, 14)
    - Stage 0: Fuse skip3 + upsample -> (B, 128, 28, 28)
    - Stage 1: Fuse skip2 + upsample -> (B, 64, 56, 56)
    - Stage 2: Fuse skip1 + upsample -> (B, 32, 112, 112)
    - Stage 3: No skip + upsample -> (B, 16, 224, 224)
    - Segmentation head: 1x1 conv to n_classes -> (B, n_classes, 224, 224)

    Attributes:
        conv_more: Initial projection from transformer channels
        decoder_blocks: List of decoder blocks for upsampling
        segmentation_head: Final 1x1 conv for class prediction
    '''

    def __init__(self, config: ModelConfig, in_channels: int = 768):
        '''
        Initialize decoder.

        Args:
            config: Model configuration
            in_channels: Number of channels from transformer output
        '''
        super().__init__()

        self.n_classes = config.n_classes
        decoder_channels = list(config.decoder_channels)
        skip_channels = list(config.skip_channels)
        n_skip = config.n_skip

        if len(decoder_channels) != 4:
            raise MRIScanException(f"Expected 4 decoder channels, got {len(decoder_channels)}")

        if n_skip > len(skip_channels):
            raise MRIScanException(f"n_skip ({n_skip}) cannot exceed number of skip channels ({len(skip_channels)})")

        # Initial projection from transformer features to first decoder channel
        self.conv_more = ConvBlock(
            in_channels=in_channels,
            out_channels=decoder_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            use_weight_std=False,
            activation='relu'
        )

        # Build decoder blocks
        # Stage 0: 14x14 (with skip3) -> 28x28
        # Stage 1: 28x28 (with skip2) -> 56x56
        # Stage 2: 56x56 (with skip1) -> 112x112
        # Stage 3: 112x112 (no skip) -> 224x224
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(decoder_channels)):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i + 1] if i < len(decoder_channels) - 1 else decoder_channels[i]

            # Use skip connection for first n_skip blocks
            use_skip = i < n_skip
            skip_ch = skip_channels[n_skip - 1 - i] if use_skip else None

            self.decoder_blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    skip_channels=skip_ch,
                    use_skip=use_skip
                )
            )

        # Segmentation head: 1x1 conv to n_classes
        self.segmentation_head = nn.Conv2d(
            decoder_channels[-1],
            self.n_classes,
            kernel_size=1,
            bias=True
        )

        logger.info(f"Initialized DecoderCUP with decoder_channels={decoder_channels}, skip_channels={skip_channels}, n_skip={n_skip}, n_classes={self.n_classes}")

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass through decoder.

        Args:
            x: Transformer features of shape (B, 768, 14, 14)
            skip_connections: List of skip connections from ResNet:
                [skip1: (B, 256, 56, 56),
                 skip2: (B, 512, 28, 28),
                 skip3: (B, 1024, 14, 14)]

        Returns:
            Segmentation logits of shape (B, n_classes, 224, 224)
        '''
        # Initial projection: (B, 768, 14, 14) -> (B, 256, 14, 14)
        x = self.conv_more(x)

        # Reverse skip connections for easier indexing
        skip_reversed = skip_connections[::-1]      # [skip3, skip2, skip1]

        # Progressive upsampling with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_reversed[i] if i < len(skip_reversed) else None
            x = decoder_block(x, skip)

        # Segmentation head: (B, 16, 224, 224) -> (B, n_classes, 224, 224)
        logits = self.segmentation_head(x)

        return logits