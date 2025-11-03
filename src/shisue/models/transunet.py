from os.path import join as pjoin
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from shisue.models.decoder import DecoderCUP
from shisue.models.resnet import ResNetV2
from shisue.models.transformer import Transformer
from shisue.utils.config import ModelConfig
from shisue.utils.exceptions import MRIScanException
from shisue.utils.logger import get_logger


logger = get_logger(__name__)


def np2th(weights: np.ndarray, conv: bool = False) -> torch.Tensor:
    '''
    Helper function to convert NumPy weights to PyTorch tensors.
    
    Args:
        weights: NumPy array
        conv: If True, transpose from HWIO (TensorFlow) to OIHW (PyTorch)
    
    Returns:
        PyTorch tensor
    '''
    if conv:
        # TensorFlow conv: [height, width, in_channels, out_channels]
        # PyTorch conv: [out_channels, in_channels, height, width]
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


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

        logger.info(f"TransUNet initialized successfully")

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
        Load pretrained ResNet weights from ImageNet-21k (.npz format).
        
        The pretrained weights are from TensorFlow TransUNet implementation
        and need to be converted to PyTorch format.
        
        Args:
            pretrained_path: Path to pretrained ResNet weights (.npz file)
        '''
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            logger.warning(f"Pretrained ResNet file not found: {pretrained_path}. Skipping.")
            return
        
        logger.info(f"Loading pretrained ResNet weights from {pretrained_path}")
        
        try:
            # Load .npz file (NumPy format, not PyTorch)
            weights = np.load(pretrained_path)
            logger.info(f"Loaded .npz file with {len(weights.keys())} keys")
            
            with torch.no_grad():
                # Load ResNet root conv
                if 'conv_root/kernel' in weights:
                    root_conv_weight = np2th(weights['conv_root/kernel'], conv=True)
                    self.resnet.root.conv.weight.copy_(root_conv_weight)
                    logger.info("Loaded ResNet root conv weights")
                
                # Load ResNet root GroupNorm
                if 'gn_root/scale' in weights and 'gn_root/bias' in weights:
                    gn_weight = np2th(weights['gn_root/scale']).view(-1)
                    gn_bias = np2th(weights['gn_root/bias']).view(-1)
                    self.resnet.root.gn.weight.copy_(gn_weight)
                    self.resnet.root.gn.bias.copy_(gn_bias)
                    logger.info("Loaded ResNet root GroupNorm weights")
                
                # Load ResNet body blocks
                # The ResNet body consists of multiple stages (blocks)
                # Format: resnet/block{block_num}/unit{unit_num}/...
                loaded_units = 0
                for bname, block in self.resnet.body.named_children():
                    for uname, unit in block.named_children():
                        unit_prefix = f"resnet/block{bname}/unit{uname}"
                        
                        try:
                            # Load GroupNorm weights for each conv layer
                            # norm1 (before conv1)
                            if f"{unit_prefix}/a/gn/scale" in weights and f"{unit_prefix}/a/gn/bias" in weights:
                                gn_weight = np2th(weights[f"{unit_prefix}/a/gn/scale"]).view(-1)
                                gn_bias = np2th(weights[f"{unit_prefix}/a/gn/bias"]).view(-1)
                                unit.norm1.weight.copy_(gn_weight)
                                unit.norm1.bias.copy_(gn_bias)
                            
                            # norm2 (before conv2)
                            if f"{unit_prefix}/b/gn/scale" in weights and f"{unit_prefix}/b/gn/bias" in weights:
                                gn_weight = np2th(weights[f"{unit_prefix}/b/gn/scale"]).view(-1)
                                gn_bias = np2th(weights[f"{unit_prefix}/b/gn/bias"]).view(-1)
                                unit.norm2.weight.copy_(gn_weight)
                                unit.norm2.bias.copy_(gn_bias)
                            
                            # norm3 (before conv3)
                            if f"{unit_prefix}/c/gn/scale" in weights and f"{unit_prefix}/c/gn/bias" in weights:
                                gn_weight = np2th(weights[f"{unit_prefix}/c/gn/scale"]).view(-1)
                                gn_bias = np2th(weights[f"{unit_prefix}/c/gn/bias"]).view(-1)
                                unit.norm3.weight.copy_(gn_weight)
                                unit.norm3.bias.copy_(gn_bias)
                            
                            # Load Conv weights
                            # conv1 (1x1 reduction)
                            if f"{unit_prefix}/a/kernel" in weights:
                                conv1_weight = np2th(weights[f"{unit_prefix}/a/kernel"], conv=True)
                                unit.conv1.weight.copy_(conv1_weight)
                            
                            # conv2 (3x3 spatial)
                            if f"{unit_prefix}/b/kernel" in weights:
                                conv2_weight = np2th(weights[f"{unit_prefix}/b/kernel"], conv=True)
                                unit.conv2.weight.copy_(conv2_weight)
                            
                            # conv3 (1x1 expansion)
                            if f"{unit_prefix}/c/kernel" in weights:
                                conv3_weight = np2th(weights[f"{unit_prefix}/c/kernel"], conv=True)
                                unit.conv3.weight.copy_(conv3_weight)
                            
                            # Load downsample projection if it exists
                            if unit.downsample is not None and f"{unit_prefix}/a/proj/kernel" in weights:
                                proj_weight = np2th(weights[f"{unit_prefix}/a/proj/kernel"], conv=True)
                                unit.downsample.weight.copy_(proj_weight)
                            
                            loaded_units += 1
                            
                        except KeyError as e:
                            logger.warning(f"Missing keys for block {bname}, unit {uname}: {e}")
                        except Exception as e:
                            logger.warning(f"Could not load weights for block {bname}, unit {uname}: {e}")
                
                if loaded_units > 0:
                    logger.info(f"Loaded ResNet body with {loaded_units} units")
                else:
                    logger.warning("No ResNet body weights loaded - file may not contain body weights")
            
            logger.info(f"Successfully loaded pretrained ResNet weights")
        
        except Exception as e:
            logger.error(f"Failed to load pretrained ResNet weights: {e}")
            logger.warning("Continuing with random initialization for ResNet")

    def load_pretrained_transformer(self, pretrained_path: str):
        '''
        Load pretrained Vision Transformer weights from ImageNet-21k (.npz format).
        
        The pretrained weights are from TensorFlow TransUNet implementation
        and need to be converted to PyTorch format with proper key mapping.
        
        Args:
            pretrained_path: Path to pretrained ViT weights (.npz file)
        '''
        pretrained_path = Path(pretrained_path)
        if not pretrained_path.exists():
            logger.warning(f"Pretrained Transformer file not found: {pretrained_path}. Skipping.")
            return
        
        logger.info(f"Loading pretrained Transformer weights from {pretrained_path}")
        
        try:
            # Load .npz file (NumPy format, not PyTorch)
            weights = np.load(pretrained_path)
            logger.info(f"Loaded .npz file with keys: {list(weights.keys())[:10]}...")  # Show first 10 keys
            
            with torch.no_grad():
                # Load patch embedding weights
                # For R50+ViT hybrid model: Conv2d(1024, 768, kernel_size=(1,1), stride=(1,1))
                # TF format: [H, W, in_channels, out_channels] = [1, 1, 1024, 768]
                # Our format: Linear(1024, 768) = (in_features, out_features)
                if 'embedding/kernel' in weights and 'embedding/bias' in weights:
                    patch_weight = np2th(weights['embedding/kernel'], conv=True)
                    patch_bias = np2th(weights['embedding/bias'])
                    
                    # After np2th with conv=True: (out_channels, in_channels, H, W) = (768, 1024, 1, 1)
                    # For Linear layer, we need: (out_features, in_features) = (768, 1024)
                    # Simply squeeze the spatial dimensions (1, 1)
                    if patch_weight.dim() == 4:
                        patch_weight = patch_weight.squeeze(-1).squeeze(-1)  # (768, 1024)
                    
                    self.transformer.patch_embeddings.projection.weight.copy_(patch_weight)
                    self.transformer.patch_embeddings.projection.bias.copy_(patch_bias)
                    logger.info("Loaded patch embedding weights")
                
                # Load encoder norm (final LayerNorm)
                if 'Transformer/encoder_norm/scale' in weights and 'Transformer/encoder_norm/bias' in weights:
                    norm_weight = np2th(weights['Transformer/encoder_norm/scale'])
                    norm_bias = np2th(weights['Transformer/encoder_norm/bias'])
                    # YOUR attributes: transformer.norm
                    self.transformer.norm.weight.copy_(norm_weight)
                    self.transformer.norm.bias.copy_(norm_bias)
                    logger.info("Loaded encoder norm weights")
                
                # Load position embeddings (may need resizing)
                if 'Transformer/posembed_input/pos_embedding' in weights:
                    posemb = np2th(weights['Transformer/posembed_input/pos_embedding'])
                    # YOUR attributes: transformer.positional_embeddings.positional_embeddings
                    posemb_new = self.transformer.positional_embeddings.positional_embeddings
                    
                    if posemb.size() == posemb_new.size():
                        self.transformer.positional_embeddings.positional_embeddings.copy_(posemb)
                        logger.info(f"Loaded position embeddings: {posemb.size()}")
                    
                    elif posemb.size()[1] - 1 == posemb_new.size()[1]:
                        # Remove CLS token position embedding
                        posemb = posemb[:, 1:]
                        self.transformer.positional_embeddings.positional_embeddings.copy_(posemb)
                        logger.info(f"Loaded position embeddings (removed CLS token): {posemb.size()}")
                    
                    else:
                        # Resize position embeddings via interpolation
                        logger.info(f"Resizing position embeddings from {posemb.size()} to {posemb_new.size()}")
                        ntok_new = posemb_new.size(1)
                        
                        # Remove CLS token if present
                        if posemb.size(1) > ntok_new:
                            posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                        else:
                            posemb_grid = posemb[0]
                        
                        gs_old = int(np.sqrt(len(posemb_grid)))
                        gs_new = int(np.sqrt(ntok_new))
                        
                        logger.info(f"Position embedding grid size: {gs_old} -> {gs_new}")
                        
                        # Reshape to 2D grid
                        posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                        
                        # Interpolate using scipy
                        zoom_factor = (gs_new / gs_old, gs_new / gs_old, 1)
                        posemb_grid = ndimage.zoom(posemb_grid, zoom_factor, order=1)
                        
                        # Reshape back
                        posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                        posemb = torch.from_numpy(posemb_grid).float()
                        
                        self.transformer.positional_embeddings.positional_embeddings.copy_(posemb)
                        logger.info("Loaded and resized position embeddings")
                
                # Load transformer encoder blocks
                # Format: Transformer/encoderblock_{n}/...
                loaded_blocks = 0
                # YOUR attributes: transformer.transformer_blocks (list of TransformerBlock)
                for idx, block in enumerate(self.transformer.transformer_blocks):
                    block_prefix = f"Transformer/encoderblock_{idx}"
                    
                    try:
                        # Attention Query, Key, Value, Out
                        q_weight = np2th(weights[pjoin(block_prefix, "MultiHeadDotProductAttention_1/query/kernel")])
                        q_bias = np2th(weights[pjoin(block_prefix, "MultiHeadDotProductAttention_1/query/bias")])
                        k_weight = np2th(weights[pjoin(block_prefix, "MultiHeadDotProductAttention_1/key/kernel")])
                        k_bias = np2th(weights[pjoin(block_prefix, "MultiHeadDotProductAttention_1/key/bias")])
                        v_weight = np2th(weights[pjoin(block_prefix, "MultiHeadDotProductAttention_1/value/kernel")])
                        v_bias = np2th(weights[pjoin(block_prefix, "MultiHeadDotProductAttention_1/value/bias")])
                        out_weight = np2th(weights[pjoin(block_prefix, "MultiHeadDotProductAttention_1/out/kernel")])
                        out_bias = np2th(weights[pjoin(block_prefix, "MultiHeadDotProductAttention_1/out/bias")])
                        
                        # Reshape and transpose: TF uses (hidden_size, hidden_size), PyTorch Linear expects transposed
                        q_weight = q_weight.view(self.config.hidden_size, self.config.hidden_size).t()
                        k_weight = k_weight.view(self.config.hidden_size, self.config.hidden_size).t()
                        v_weight = v_weight.view(self.config.hidden_size, self.config.hidden_size).t()
                        out_weight = out_weight.view(self.config.hidden_size, self.config.hidden_size).t()
                        
                        # YOUR attributes: block.attention.query/key/value/out_projection
                        block.attention.query.weight.copy_(q_weight)
                        block.attention.query.bias.copy_(q_bias.view(-1))
                        block.attention.key.weight.copy_(k_weight)
                        block.attention.key.bias.copy_(k_bias.view(-1))
                        block.attention.value.weight.copy_(v_weight)
                        block.attention.value.bias.copy_(v_bias.view(-1))
                        block.attention.out_projection.weight.copy_(out_weight)
                        block.attention.out_projection.bias.copy_(out_bias.view(-1))
                        
                        # MLP weights
                        mlp_weight_0 = np2th(weights[pjoin(block_prefix, "MlpBlock_3/Dense_0/kernel")]).t()
                        mlp_bias_0 = np2th(weights[pjoin(block_prefix, "MlpBlock_3/Dense_0/bias")])
                        mlp_weight_1 = np2th(weights[pjoin(block_prefix, "MlpBlock_3/Dense_1/kernel")]).t()
                        mlp_bias_1 = np2th(weights[pjoin(block_prefix, "MlpBlock_3/Dense_1/bias")])
                        
                        # YOUR attributes: block.mlp.fc1/fc2
                        block.mlp.fc1.weight.copy_(mlp_weight_0)
                        block.mlp.fc1.bias.copy_(mlp_bias_0)
                        block.mlp.fc2.weight.copy_(mlp_weight_1)
                        block.mlp.fc2.bias.copy_(mlp_bias_1)
                        
                        # LayerNorm weights
                        attn_norm_weight = np2th(weights[pjoin(block_prefix, "LayerNorm_0/scale")])
                        attn_norm_bias = np2th(weights[pjoin(block_prefix, "LayerNorm_0/bias")])
                        mlp_norm_weight = np2th(weights[pjoin(block_prefix, "LayerNorm_2/scale")])
                        mlp_norm_bias = np2th(weights[pjoin(block_prefix, "LayerNorm_2/bias")])
                        
                        # YOUR attributes: block.attention_norm/mlp_norm
                        block.attention_norm.weight.copy_(attn_norm_weight)
                        block.attention_norm.bias.copy_(attn_norm_bias)
                        block.mlp_norm.weight.copy_(mlp_norm_weight)
                        block.mlp_norm.bias.copy_(mlp_norm_bias)
                        
                        loaded_blocks += 1
                    
                    except KeyError as e:
                        logger.warning(f"Could not load weights for block {idx}: missing key {e}")
                        break
                
                logger.info(f"Loaded {loaded_blocks}/{len(self.transformer.transformer_blocks)} transformer blocks")
            
            logger.info(f"Successfully loaded pretrained Transformer weights")
        
        except Exception as e:
            logger.error(f"Failed to load pretrained Transformer weights: {e}")
            logger.warning("Continuing with random initialization for Transformer")

    def count_parameters(self) -> Tuple[int, int]:
        '''
        Count total and trainable number of parameters.

        Returns:
            Tuple of (total number of parameters, number of trainable parameters)
        '''
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return total_params, trainable_params


def build_transunet(config: ModelConfig) -> TransUNet:
    '''
    Factory function to build TransUNet model and load pretrained weights if specified in config.

    Args:
        config: Model configuration

    Returns:
        TransUNet model instance
    '''
    model = TransUNet(config)

    # Load pretrained weights if specified in config
    # Both ResNet and Transformer load from the same file
    if config.pretrained_path is not None:
        model.load_pretrained_resnet(config.pretrained_path)
        model.load_pretrained_transformer(config.pretrained_path)

    return model