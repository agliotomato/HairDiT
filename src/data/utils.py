"""Shared data utilities."""

import torch
import torch.nn.functional as F


def soft_composite(img: torch.Tensor, matte: torch.Tensor) -> torch.Tensor:
    """
    Compute hair region image: img * (matte / 255.0).

    Args:
        img:   (C, H, W) or (B, C, H, W), values in [0, 1]
        matte: (1, H, W) or (B, 1, H, W), values in [0, 255] OR [0, 1]

    Returns:
        hair_region: same shape as img, values in [0, 1]
    """
    if matte.max() > 1.0:
        matte = matte / 255.0
    return img * matte


def resize_matte_to_latent(matte: torch.Tensor, latent_size: int = 64) -> torch.Tensor:
    """
    Resize matte to latent spatial resolution.

    PDF Eq. 7의 m̄_64 = ChannelAvg(PixelUnshuffle(m, 8))와 정렬하기 위해 area(=8×8 블록 평균)
    다운샘플을 쓴다. 512→64는 정확히 factor 8이라 area == ChannelAvg(PixelUnshuffle(8))
    (실측 max|Δ| ≈ 3.6e-7). 기존 bilinear는 경계 alpha에서 최대 ~0.31까지 어긋나 논문과
    불일치했다(gate/blending은 이미 area 계열을 쓰므로 loss matte만 정렬).

    Args:
        matte:       (B, 1, H, W), values in [0, 1]
        latent_size: target spatial size (default 64 for 512px → 8× downsample)

    Returns:
        matte_latent: (B, 1, latent_size, latent_size)
    """
    return F.interpolate(matte, size=(latent_size, latent_size), mode="area")
