from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn

from shisue.models.components import StdConv2d
from shisue.utils.config import ModelConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class Bottleneck(nn.Module):
    '''
    ResNet bottleneck block with post-activation design.

    Architecture:
    - Conv1x1 + GroupNorm + ReLU (reduce channels)
    - Conv3x3 + GroupNorm + ReLU (spatial processing)
    - Conv1x1 + GroupNorm (expand channels)
    - Add residual connection
    - ReLU activation

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

        # conv1 -> norm1 -> relu1
        self.conv1 = StdConv2d(
            in_channels,
            mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm1 = self._make_norm(mid_channels, num_groups)
        self.relu1 = nn.ReLU(inplace=True)

        # conv2 -> norm2 -> relu2
        self.conv2 = StdConv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.norm2 = self._make_norm(mid_channels, num_groups)
        self.relu2 = nn.ReLU(inplace=True)

        # conv3 -> norm3 (no relu here)
        self.conv3 = StdConv2d(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.norm3 = self._make_norm(out_channels, num_groups)

        # Final ReLU after residual addition
        self.relu = nn.ReLU(inplace=True)

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
            self.gn_proj = self._make_norm(out_channels, num_groups)
        else:
            self.downsample = None
            self.gn_proj = None

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

        return nn.GroupNorm(num_groups=effective_num_groups, num_channels=num_channels, eps=1e-6)

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
        
        # First bottleneck: conv1 -> norm1 -> relu1
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Second bottleneck: conv2 -> norm2 -> relu2
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        # Third bottleneck: conv3 -> norm3
        x = self.conv3(x)
        x = self.norm3(x)

        # Apply projection to residual if needed
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = self.gn_proj(residual)

        # Add residual connection and apply final ReLU
        x = x + residual
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    '''
    ResNet backbone for TransUNet hybrid architecture.

    This implementation follows a modified ResNet50 with:
    - Post-activation design (conv + norm + activation)
    - GroupNorm instead of BatchNorm for stability with small batch sizes
    - Weight standardization for better interaction with GroupNorm
    - Skip connection extraction at multiple scales for decoder upsampling
    - Root skip connection at 112x112 resolution

    Reference:
        He, K., Zhang, X., Ren, S., & Sun, J. (2015, December 10).
        Deep residual learning for image recognition.
        arXiv.org. https://arxiv.org/abs/1512.03385

    Architecture:
    - Root block: 7x7 conv + norm + relu (224 -> 112)
    - MaxPool: 3x3 (112 -> 56, applied after collecting root skip)
    - Block 1: n_blocks at 56x56 with 64 channels -> 256 output channels
    - Block 2: n_blocks at 28x28 with 128 channels -> 512 output channels
    - Block 3: n_blocks at 14x14 with 256 channels -> 1024 output channels
    - Output: 14x14 feature map + skip connections from root and all blocks

    Attributes:
        root: Initial convolution and normalization
        body: Sequential container with three ResNet blocks
        skip_channels: List of channel dimensions for skip connections
    '''

    def __init__(self, config: ModelConfig):
        '''
        Initialize ResNet backbone.

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

        # Root block: initial convolution (MaxPool applied separately in forward)
        # Conv: (B, 3, 224, 224) -> (B, 64*width_factor, 112, 112)
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(
                in_channels=3,
                out_channels=channels[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )),
            ('gn', nn.GroupNorm(num_groups=min(32, channels[0]), num_channels=channels[0], eps=1e-6)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.body = nn.Sequential(OrderedDict([
            # Block 1: 56x56 -> 56x56
            # Output channels: 64 * 4 = 256
            ('block1', self._make_stage(
                in_channels=channels[0],
                mid_channels=channels[0],
                num_blocks=num_layers[0],
                stride=1
            )),
            # Block 2: 56x56 -> 28x28
            # Output channels: 128 * 4 = 512
            ('block2', self._make_stage(
                in_channels=channels[0] * Bottleneck.EXPANSION,
                mid_channels=channels[1],
                num_blocks=num_layers[1],
                stride=2
            )),
            # Block 3: 28x28 -> 14x14
            # Output channels: 256 * 4 = 1024
            ('block3', self._make_stage(
                in_channels=channels[1] * Bottleneck.EXPANSION,
                mid_channels=channels[2],
                num_blocks=num_layers[2],
                stride=2
            ))
        ]))

        # Store output channels for skip connections (including root)
        self.skip_channels = [
            channels[0],                            # Root: 64
            channels[0] * Bottleneck.EXPANSION,     # Stage 1: 256
            channels[1] * Bottleneck.EXPANSION,     # Stage 2: 512
            channels[2] * Bottleneck.EXPANSION      # Stage 3: 1024
        ]

        logger.info(f"Initialized ResNet with {num_layers} layers, skip channels: {self.skip_channels}")

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
            Bottleneck(
                in_channels=in_channels,
                mid_channels=mid_channels,
                stride=stride
            )
        )

        # Remaining blocks (no downsampling)
        out_channels = mid_channels * Bottleneck.EXPANSION
        for _ in range(1, num_blocks):
            layers.append(
                Bottleneck(
                    in_channels=out_channels,
                    mid_channels=mid_channels,
                    stride=1
                )
            )
            
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        '''
        Forward pass through ResNet backbone.

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Tuple of:
                - Final feature map of shape (B, 1024, 14, 14)
                - List of skip connection features from each stage:
                    [root: (B, 64, 112, 112),
                     block1: (B, 256, 56, 56),
                     block2: (B, 512, 28, 28),
                     block3: (B, 1024, 14, 14)]
        '''
        # Root block: 224 -> 112 (before MaxPool)
        x = self.root(x)            # (B, 64, 112, 112)
        x0 = x                      # Collect skip from root

        # Apply MaxPool separately: 112 -> 56
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)  # (B, 64, 56, 56)

        # Block 1: 56 -> 56
        x1 = self.body.block1(x)    # (B, 256, 56, 56)

        # Block 2: 56 -> 28
        x2 = self.body.block2(x1)   # (B, 512, 28, 28)

        # Block 3: 28 -> 14
        x3 = self.body.block3(x2)   # (B, 1024, 14, 14)

        # Collect skip connections (including root)
        skip_connections = [x0, x1, x2, x3]

        return x3, skip_connections