from typing import List, Optional

import torch
import torch.nn as nn

from shisue.models.components import ConvBlock
from shisue.utils.config import ModelConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class DecoderBlock(nn.Module):
    '''
    Decoder block with skip connection fusion and upsampling.

    This block:
    1. Upsamples by 2x using bilinear interpolation
    2. Optionally concatenates with skip connection from encoder
    3. Applies two sequential convolutions for feature processing

    Attributes:
        upsample: Bilinear upsampling
        conv1: First convolutional block for feature processing
        conv2: Second convolutional block for feature refinement
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

        # Bilinear upsampling (2x spatial resolution)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # First convolution: processes concatenated features
        conv1_in_channels = in_channels + skip_channels if (use_skip and skip_channels) else in_channels
        self.conv1 = ConvBlock(
            in_channels=conv1_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_weight_std=False,
            activation='relu'
        )

        # Second convolution: refines features
        self.conv2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_weight_std=False,
            activation='relu'
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Forward pass through decoder block.

        Args:
            x: Input features from previous decoder layer
            skip: Optional skip connection from encoder (if used)

        Returns:
            Processed and upsampled features
        '''
        # Upsample first
        x = self.upsample(x)

        # Concatenate with skip connection if available
        if self.use_skip and skip is not None:
            x = torch.cat([x, skip], dim=1)

        # Process features with two convolutions
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class DecoderCUP(nn.Module):
    '''
    Cascaded Upsampling Decoder (CUP) for TransUNet.

    This decoder uses cascaded upsampling blocks to progressively upsample transformer features from 14x14 to 224x224,
    incorporating skip connections from ResNet to recover spatial resolution and combine multi-scale features.

    Architecture (with n_skip=3):
    - Input: Transformer features (B, 768, 14, 14)
    - conv_more: Project to head channels (B, 512, 14, 14)
    - Stage 0: Upsample -> Fuse skip2 (512ch @ 28x28) -> Conv1 -> Conv2 -> (B, 256, 28, 28)
    - Stage 1: Upsample -> Fuse skip1 (256ch @ 56x56) -> Conv1 -> Conv2 -> (B, 128, 56, 56)
    - Stage 2: Upsample -> Fuse skip0 (64ch @ 112x112) -> Conv1 -> Conv2 -> (B, 64, 112, 112)
    - Stage 3: Upsample -> No skip -> Conv1 -> Conv2 -> (B, 16, 224, 224)
    - Segmentation head: 1x1 conv to n_classes -> (B, n_classes, 224, 224)
    
    Note: Bottleneck skip (14x14) is excluded as transformer output already contains this information.

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
        head_channels = config.head_channels
        decoder_channels = list(config.decoder_channels)
        skip_channels = list(config.skip_channels)
        n_skip = config.n_skip

        if len(decoder_channels) != 4:
            raise MRIScanException(f"Expected 4 decoder channels, got {len(decoder_channels)}")

        if n_skip > len(skip_channels):
            raise MRIScanException(f"n_skip ({n_skip}) cannot exceed number of skip channels ({len(skip_channels)})")

        # Initial projection from transformer features to head channels
        self.conv_more = ConvBlock(
            in_channels=in_channels,
            out_channels=head_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_weight_std=False,
            activation='relu'
        )

        # Build decoder blocks
        # Construct in_channels list: [head_channels] + decoder_channels[:-1]
        # in_channels:  [512, 256, 128, 64]
        # out_channels: [256, 128, 64, 16]
        in_channels_list = [head_channels] + decoder_channels[:-1]
        out_channels_list = decoder_channels
        
        # Stage 0: 14x14 -> upsample -> 28x28 + skip2 (512ch) -> 256ch
        # Stage 1: 28x28 -> upsample -> 56x56 + skip1 (256ch) -> 128ch
        # Stage 2: 56x56 -> upsample -> 112x112 + skip0 (64ch) -> 64ch
        # Stage 3: 112x112 -> upsample -> 224x224 (no skip) -> 16ch
        self.decoder_blocks = nn.ModuleList()

        for i in range(len(decoder_channels)):
            in_ch = in_channels_list[i]
            out_ch = out_channels_list[i]

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

        logger.info(f"Initialized DecoderCUP with head_channels={head_channels}, decoder_channels={decoder_channels}, skip_channels={skip_channels}, n_skip={n_skip}, n_classes={self.n_classes}")

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass through decoder.

        Args:
            x: Transformer features of shape (B, 768, 14, 14)
            skip_connections: List of skip connections from ResNet:
                [skip0: (B, 64, 112, 112),
                 skip1: (B, 256, 56, 56),
                 skip2: (B, 512, 28, 28),
                 skip3: (B, 1024, 14, 14)]
                Note: skip3 (bottleneck at 14x14) is excluded in decoder as
                transformer output already contains bottleneck information.

        Returns:
            Segmentation logits of shape (B, n_classes, 224, 224)
        '''
        # Initial projection: (B, 768, 14, 14) -> (B, 512, 14, 14)
        x = self.conv_more(x)

        # Exclude bottleneck skip (14x14) and reverse for upsample-first architecture
        # With upsample-first, we need skips at [28x28, 56x56, 112x112] not [14x14, ...]
        skip_reversed = skip_connections[:-1][::-1]  # [skip2, skip1, skip0]

        # Progressive upsampling with skip connections
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip = skip_reversed[i] if i < len(skip_reversed) else None
            x = decoder_block(x, skip)

        # Segmentation head: (B, 16, 224, 224) -> (B, n_classes, 224, 224)
        logits = self.segmentation_head(x)

        return logits