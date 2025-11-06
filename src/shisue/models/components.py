from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


class StdConv2d(nn.Conv2d):
    '''
    Standardized 2D convolution layer.

    Weight standardization normalizes the weights of convolutional layers,
    which works well with GroupNorm and improves training stability.

    Reference:
        Qiao, S., Wang, H., Liu, C., Shen, W., & Yuille, A. (2019, March 25).
        Micro-Batch Training with Batch-Channel Normalization and Weight Standardization.
        arXiv.org. https://arxiv.org/abs/1903.10520
    '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass with weight standardization.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        '''
        weight = self.weight

        # Calculate mean and variance per output channel
        weight_mean = weight.mean(dim=(1, 2, 3), keepdim=True)
        weight_std = weight.std(dim=(1, 2, 3), keepdim=True) + 1e-5

        # Standardize weights
        weight = (weight - weight_mean) / weight_std

        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class ConvBlock(nn.Module):
    '''
    Convolutional block with Conv2d + GroupNorm + Activation.

    This is the fundamental building block used throughout the architecture.
    GroupNorm is used instead of BatchNorm for better performance with small batch sizes.

    Attributes:
        conv: Convolutional layer (standard or weight-standardized)
        norm: Group normalization layer
        activation: Activation function (ReLU or GELU)
    '''

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        num_groups: int = 32,
        use_weight_std: bool = False,
        activation: str = 'relu'
    ) -> None:
        '''
        Initialize convolutional block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Padding added to both sides of the input
            groups: Number of blocked connections from input channels to output channels
            num_groups: Number of groups for GroupNorm
            use_weight_std: Whether to use weight standardization
            activation: Activation function (relu or gelu)
        '''
        super().__init__()

        if activation not in {'relu', 'gelu'}:
            raise MRIScanException(f"Invalid activation function: {activation}. Must be 'relu' or 'gelu'.")

        # Select convolution type
        conv_cls = StdConv2d if use_weight_std else nn.Conv2d
        self.conv = conv_cls(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            bias=False      # No bias needed before normalization
        )

        # GroupNorm: divide channels into groups
        # Ensure num_groups divides out_channels evenly
        effective_num_groups = min(num_groups, out_channels)
        while out_channels % effective_num_groups != 0:
            effective_num_groups -= 1

        self.norm = nn.GroupNorm(num_groups=effective_num_groups, num_channels=out_channels)

        # Activation function
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through conv-norm-activation block.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor of shape (B, C, H, W)
        '''
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class PatchEmbeddings(nn.Module):
    '''
    Convert image patches into embeddings for transformer input.

    For this hybrid model (ResNet + Vision Transformer),
    patches are extracted from ResNet feature maps rather than raw images.

    Attributes:
        patch_size: Size of each extracted patch
        hidden_size: Dimension of embedding vectors
        projection: Conv2d layer for patch extraction and projection
    '''

    def __init__(self, patch_size: Tuple[int, int], in_channels: int, hidden_size: int):
        '''
        Initialize patch embeddings.

        Args:
            patch_size: Size of each extracted patch (height, width)
            in_channels: Number of input channels from feature maps
            hidden_size: Dimension of output embeddings
        '''
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        # Conv2d projection: extracts patches and projects in one operation
        # kernel_size=stride=patch_size ensures non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Convert feature maps to patch embeddings.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Patch embeddings of shape (B, num_patches, hidden_size)
        '''
        batch_size, channels, height, width = x.shape

        # Validate input dimensions
        if height % self.patch_size[0] != 0 or width % self.patch_size[1] != 0:
            raise MRIScanException(f"Feature map size ({height}x{width}) must be divisible by patch size {self.patch_size}")

        # Apply convolution: (B, C, H, W) -> (B, hidden_size, H/patch_h, W/patch_w)
        x = self.projection(x)
        
        # Flatten spatial dimensions: (B, hidden_size, H', W') -> (B, hidden_size, num_patches)
        x = x.flatten(2)
        
        # Transpose to sequence format: (B, hidden_size, num_patches) -> (B, num_patches, hidden_size)
        x = x.transpose(-1, -2)
        
        return x


class PositionalEmbeddings(nn.Module):
    '''
    Learnable positional embeddings for transformer.

    Positional embeddings reintroduce spatial information lost in the patch embedding process.
    Unlike sinusoidal embeddings used in NLP Transformers, these positions embeddings are learnt during training.

    Attributes:
        num_patches: Number of spatial patches
        hidden_size: Dimension of embedding vectors
        positional_embeddings: Learnable embeddings for each patch position
    '''

    def __init__(self, num_patches: int, hidden_size: int):
        '''
        Initialize positional embeddings.

        Args:
            num_patches: Number of patches from the feature map
            hidden_size: Dimension of embedding vectors
        '''
        super().__init__()

        self.num_patches = num_patches
        self.hidden_size = hidden_size

        # Learnable positional embeddings
        self.positional_embeddings = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # Initialize with truncated normal distribution
        nn.init.trunc_normal_(self.positional_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Add positional embeddings to patch embeddings.

        Args:
            x: Patch embeddings of shape (B, num_patches, hidden_size)

        Returns:
            Embeddings with positional encoding of shape (B, num_patches, hidden_size)
        '''
        return x + self.positional_embeddings


class MultiHeadSelfAttention(nn.Module):
    '''
    Multi-head self-attention mechanism for transformer.

    This is the core attention mechanism that allows the model to attend to different positions in the input sequence,
    capturing both local and global dependencies between patches.

    Attributes:
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        hidden_size: Total dimension (num_heads * head_dim)
        query, key, value: Linear projections for attention
        out_projection: Output projection after concatenating heads
        attention_dropout: Dropout applied to attention weights
        projection_dropout: Dropout applied to output projection
    '''

    def __init__(self, hidden_size: int, num_heads: int, attention_dropout_rate: float = 0.0):
        '''
        Initialize multi-head self-attention.

        Args:
            hidden_size: Dimension of input embeddings
            num_heads: Number of attention heads
            attention_dropout_rate: Dropout rate for attention weights
        '''
        super().__init__()

        if hidden_size % num_heads != 0:
            raise MRIScanException(f"Hidden size {hidden_size} must be divisible by number of heads {num_heads}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.out_projection = nn.Linear(hidden_size, hidden_size)

        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout_rate)
        self.projection_dropout = nn.Dropout(attention_dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through multi-head self-attention.

        Args:
            x: Input embeddings of shape (B, num_patches, hidden_size)

        Returns:
            Attended embeddings of shape (B, num_patches, hidden_size)
        '''
        batch_size, num_patches, hidden_size = x.shape

        # Linear projections and reshape to (B, num_heads, num_patches, head_dim)
        q = self.query(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        # A = softmax(Q • K^T / sqrt(head_dim)) • V
        # Scaled dot-product attention to calculate the attention weights
        # (B, num_heads, num_patches, head_dim) @ (B, num_heads, head_dim, num_patches) -> (B, num_heads, num_patches, num_patches)
        # softmax(Q • K^T / sqrt(head_dim))
        attention_scores = (q @ k.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Apply attention to values
        # (B, num_heads, num_patches, num_patches) @ (B, num_heads, num_patches, head_dim) -> (B, num_heads, num_patches, head_dim)
        # Attention Weights • Values
        attended = attention_weights @ v

        # Concatenate heads: (B, num_heads, num_patches, head_dim) -> (B, num_patches, hidden_size)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, num_patches, -1)

        # Output projection with dropout
        output = self.out_projection(attended)
        output = self.projection_dropout(output)

        return output


class MLP(nn.Module):
    '''
    Multi-layer perceptron (feed-forward network) for transformer blocks.

    Standard MLP with one hidden layer, GELU activation, and dropout.
    Expands to mlp_dim for hidden dimension and projects back to hidden_size.
    
    Attributes:
        fc1: First linear layer (expansion)
        fc2: Second linear layer (projection)
        activation: GELU activation function
        dropout: Dropout for regularization
    '''

    def __init__(self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.1):
        '''
        Initialize MLP.

        Args:
            hidden_size: Input and output dimension
            mlp_dim: Hidden layer dimension
            dropout_rate: Dropout probability
        '''
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through MLP.
        
        Args:
            x: Input tensor of shape (B, num_patches, hidden_size)

        Returns:
            Output tensor of shape (B, num_patches, hidden_size)
        '''
        x = self.fc1(x)
        x = self.activation(x)      # (B, num_patches, mlp_dim)
        x = self.dropout(x)         # (B, num_patches, mlp_dim)
        x = self.fc2(x)             # (B, num_patches, hidden_size)
        x = self.dropout(x)         # (B, num_patches, hidden_size)

        return x


class TransformerBlock(nn.Module):
    '''
    Complete transformer encoder block with self-attention and MLP.

    Architecture follows ViT:
    - Layer Norm -> Multi-head Self-Attention -> Residual
    - Layer Norm -> MLP -> Residual

    Attributes:
        attention_norm: Layer normalization before attention
        attention: Multi-head self-attention layer
        mlp_norm: Layer normalization before MLP
        mlp: Feed-forward network
    '''

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        attention_dropout_rate: float = 0.0,
        dropout_rate: float = 0.1
    ):
        '''
        Initialize transformer block.

        Args:
            hidden_size: Dimension of input embeddings
            num_heads: Number of attention heads
            mlp_dim: Hidden layer dimension of MLP
            attention_dropout_rate: Dropout rate for attention weights
            dropout_rate: Dropout rate for other components
        '''
        super().__init__()

        # Pre-norm architecture (normalization before attention and MLP)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_dropout_rate=attention_dropout_rate
        )

        self.mlp_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = MLP(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, num_patches, hidden_size)

        Returns:
            Output embeddings of shape (B, num_patches, hidden_size)
        '''
        # Self-attention with residual connection
        attention_input = self.attention_norm(x)
        attention_output = self.attention(attention_input)
        x = x + attention_output

        # MLP with residual connection
        mlp_input = self.mlp_norm(x)
        mlp_output = self.mlp(mlp_input)
        x = x + mlp_output

        return x