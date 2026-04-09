import mlx.core as mx
import mlx.nn as nn
import torch

"""
This file previously contained temporary functions to compensate for missing features in old versions of MLX 
(e.g., prior to v0.0.10). As of MLX 0.31+, most of these exist natively.

Native Equivalents:
- softplus -> nn.softplus(x)
- unsqueeze(x, axis) -> mx.expand_dims(x, axis)
- clamp(x, min, max) -> mx.clip(x, a_min=min, a_max=max)
- DepthWiseConv1d -> nn.Conv1d(..., groups=channels)
"""

def topk(x, k):
    """
    Returns the top k biggest values of x along the last dim.
    Ordered from lowest to biggest val so caller can access the bottom threshold at index 0.

    Args:
        x : (B, vocab_size). can be probs or logits
    Returns:
        values : (B, k). ordered from lowest to biggest val
    """
    return mx.sort(x)[:, -k:]


def torch_to_mlx_depthwise_weights(torch_weights):
    """
    Converts PyTorch grouped/depthwise conv weights to native MLX weights.
    In torch: (channels, 1, kernel_size)
    In MLX with native groups=channels: (channels, kernel_size, 1)
    """
    if torch_weights.type() == "torch.BFloat16Tensor":
        torch_weights = torch_weights.float()
        
    return torch_weights.transpose(1, 2).numpy()
