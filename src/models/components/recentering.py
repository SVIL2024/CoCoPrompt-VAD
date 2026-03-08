"""Unified Re-centering Module for AnomalyCLIP.

This module provides a centralized implementation of the re-centering transformation
used in AnomalyCLIP to shift the feature space around the normality prototype.
"""

import torch
from torch import Tensor


def apply_recentering(
    features: Tensor,
    ncentroid: Tensor,
    mode: str = "subtract",
) -> Tensor:
    """Apply re-centering transformation to features.

    Re-centering shifts the feature space so that the normality prototype is at the origin.
    This transformation is fundamental to AnomalyCLIP's approach where:
    - Magnitude in the re-centered space corresponds to anomaly level
    - Direction in the re-centered space corresponds to anomaly type

    Args:
        features: Input features tensor of shape [..., D] where D is feature dimension.
        ncentroid: Normal centroid tensor of shape [1, D] or [D]. This represents the
            normality prototype in the original CLIP feature space.
        mode: Re-centering mode, one of:
            - "none": No transformation, returns features unchanged
            - "subtract": Returns (features - ncentroid)
            - "direction": Returns normalized (features - ncentroid)

    Returns:
        Re-centered features tensor of shape [..., D].

    Raises:
        ValueError: If mode is not one of the supported values.

    Examples:
        >>> features = torch.randn(32, 512)  # Batch of 32 features
        >>> ncentroid = torch.randn(1, 512)  # Normality prototype
        >>> recentered = apply_recentering(features, ncentroid, mode="subtract")
        >>> recentered.shape
        torch.Size([32, 512])
    """
    if mode == "none":
        return features

    # Ensure ncentroid is broadcastable
    if ncentroid.dim() == 1:
        ncentroid = ncentroid.unsqueeze(0)

    centered = features - ncentroid

    if mode == "subtract":
        return centered
    elif mode == "direction":
        return centered / (centered.norm(dim=-1, keepdim=True) + 1e-8)
    else:
        raise ValueError(f"Unknown recentering mode: {mode}")
