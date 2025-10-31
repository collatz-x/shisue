from typing import List, Tuple

import torch
import torch.nn as nn

from shisue.models.components import StdConv2d
from shisue.utils.config import ModelConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class BottleneckV2(nn.Module):
    '''
    ResNetv2 bottleneck block with pre-activation design.

    Architecture:
    - GroupNorm + ReLU + Conv1x1 (reduce channels)
    - GroupNorm + ReLU + Conv3x3 (spatial processing)
    - GroupNorm + ReLU + Conv1x1 (expand channels)
    - Residual connection (with projection if needed)

    Attributes:
        in_channels: Number of input channels
        out_channels: Number of output channels (after expansion)
        stride: Stride for spatial downsampling
        downsample: Optional projection for residual connection
    '''

    EXPANSION = 4   # Bottleneck expansion factor

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1, num_groups: int = 32):
        '''
        Initialize bottleneck block.

        Args:
            in_channels: Number of input channels
            mid_channels: Number of middle (bottleneck) channels
            stride: Stride for the 3x3 convolution (1 or 2)
            num_groups: Number of groups for GroupNorm
        '''
        super().__init__()

        out_channels = mid_channels * self.EXPANSION

        # Pre-activation for conv1
        self.norm1 = self._make_norm(in_channels, num_groups)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = StdConv2d(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # Pre-activation for conv2
        self.norm2 = self._make_norm(mid_channels, num_groups)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = StdConv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        # Pre-activation for conv3
        self.norm3 = self._make_norm(mid_channels, num_groups)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = StdConv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        # Projection shortcut if dimensions change
        if stride != 1 or in_channels != out_channels:
            self.downsample = StdConv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False
            )
        else:
            self.downsample = None

    def _make_norm(self, num_channels: int, num_groups: int) -> nn.GroupNorm:
        '''
        Create GroupNorm layer with appropriate number of groups.

        Args:
            num_channels: Number of channels to normalize
            num_groups: Desired number of groups

        Returns:
            GroupNorm layer
        '''
        effective_num_groups = min(num_groups, num_channels)
        while num_channels % effective_num_groups != 0:
            effective_num_groups -= 1

        return nn.GroupNorm(num_groups=effective_num_groups, num_channels=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through bottleneck block.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        '''
        # Save input for residual connection
        residual = x
        
        # First bottleneck: 1x1 reduction
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        # Second bottleneck: 3x3 spatial
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        # Third bottleneck: 1x1 expansion
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.conv3(x)

        # Apply projection to residual if needed
        if self.downsample is not None:
            residual = self.downsample(residual)

        # Add residual connection
        x = x + residual

        return x


class ResNetV2(nn.Module):
    '''
    ResNetV2 backbone for TransUNet hybrid architecture.

    This implementation follows a modified ResNet50v2 with:
    - Pre-activation design (norm + activation before convolution)
    - GroupNorm instead of BatchNorm for stability with small batch sizes
    - Weight standardization for better interaction with GroupNorm
    - Skip connection extraction for decoder upsampling

    Reference:
        He, K., Zhang, X., Ren, S., & Sun, J. (2016, March 16).
        Identity mappings in deep residual networks.
        arXiv.org. https://arxiv.org/abs/1603.05027

    Architecture:
    - Root block: 7x7 conv (224 -> 112) + 3x3 maxpool (112 -> 56)
    - Stage 1: n_blocks at 56x56 with 64 channels
    - Stage 2: n_blocks at 28x28 with 128 channels
    - Stage 3: n_blocks at 14x14 with 256 channels
    - Output: 14x14 feature map + skip connections

    Attributes:
        root: Initial convolution and pooling
        stage1, stage2, stage3: ResNet stages
        norm: Final normalization layer
    '''

    def __init__(self, config: ModelConfig):
        '''
        Initialize ResNetV2 backbone.

        Args:
            config: Model configuration containing ResNet parameters
        '''
        super().__init__()
        
        if config.resnet is None:
            raise MRIScanException("ResNet configuration is required for hybrid model.")

        num_layers = config.resnet.num_layers
        width_factor = config.resnet.width_factor

        if len(num_layers) != 3:
            raise MRIScanException(f"Expected 3 stage layers, got {len(num_layers)}")

        # Base channel dimensions (can be scaled by width factor)
        base_channels = [64, 128, 256]
        channels = [c * width_factor for c in base_channels]

        # Root block: initial convolution and pooling
        # Conv: (B, 3, 224, 224) -> (B, 64*width_factor, 112, 112)
        # MaxPool: (B, 64*width_factor, 112, 112) -> (B, 64*width_factor, 56, 56)
        self.root = nn.Sequential(
            StdConv2d(
                in_channels=3,
                out_channels=channels[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.GroupNorm(num_groups=min(32, channels[0]), num_channels=channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 1: 56x56 -> 56x56
        # Output channels: 64 * 4 = 256
        self.stage1 = self._make_stage(
            in_channels=channels[0],
            mid_channels=channels[0],
            num_blocks=num_layers[0],
            stride=1
        )

        # Stage 2: 56x56 -> 28x28
        # Output channels: 128 * 4 = 512
        self.stage2 = self._make_stage(
            in_channels=channels[0] * BottleneckV2.EXPANSION,
            mid_channels=channels[1],
            num_blocks=num_layers[1],
            stride=2
        )

        # Stage 3: 28x28 -> 14x14
        # Output channels: 256 * 4 = 1024
        self.stage3 = self._make_stage(
            in_channels=channels[1] * BottleneckV2.EXPANSION,
            mid_channels=channels[2],
            num_blocks=num_layers[2],
            stride=2
        )

        # Final normalization (pre-activation style)
        final_channels = channels[2] * BottleneckV2.EXPANSION
        self.norm = nn.GroupNorm(num_groups=min(32, final_channels), num_channels=final_channels)
        self.relu = nn.ReLU(inplace=True)

        # Store output channels for skip connections
        self.skip_channels = [
            channels[0] * BottleneckV2.EXPANSION,   # Stage 1: 256
            channels[1] * BottleneckV2.EXPANSION,   # Stage 2: 512
            channels[2] * BottleneckV2.EXPANSION    # Stage 3: 1024
        ]

        logger.info(f"Initialized ResNetV2 with {num_layers} layers, skip channels: {self.skip_channels}")

    def _make_stage(self, in_channels: int, mid_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        '''
        Create a ResNet stage with multiple bottleneck blocks.

        Args:
            in_channels: Input channels to the stage
            mid_channels: Bottleneck channels
            num_blocks: Number of bottleneck blocks in this stage
            stride: Stride for first block (2 for downsampling, 1 otherwise)

        Returns:
            Sequential container with bottleneck blocks
        '''
        layers = []

        # First block (may downsample)
        layers.append(
            BottleneckV2(
                in_channels=in_channels,
                mid_channels=mid_channels,
                stride=stride
            )
        )

        # Remaining blocks (no downsampling)
        out_channels = mid_channels * BottleneckV2.EXPANSION
        for _ in range(1, num_blocks):
            layers.append(
                BottleneckV2(
                    in_channels=out_channels,
                    mid_channels=mid_channels,
                    stride=1
                )
            )
            
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Forward pass through ResNetV2 backbone.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Tuple of:
                - Final feature map of shape (B, 1024, 14, 14)
                - List of skip connection features from each stage:
                    [stage1: (B, 256, 56, 56),
                     stage2: (B, 512, 28, 28),
                     stage3: (B, 1024, 14, 14)]
        '''
        # Root block: 224 -> 112 -> 56
        x = self.root(x)        # (B, 64, 56, 56)

        # Stage 1: 56 -> 56
        x1 = self.stage1(x)     # (B, 256, 56, 56)

        # Stage 2: 56 -> 28
        x2 = self.stage2(x1)    # (B, 512, 28, 28)

        # Stage 3: 28 -> 14
        x3 = self.stage3(x2)    # (B, 1024, 14, 14)

        # Final normalization and activation
        x_out = self.norm(x3)
        x_out = self.relu(x_out)

        # Collect skip connections
        skip_connections = [x1, x2, x3]

        return x_out, skip_connections