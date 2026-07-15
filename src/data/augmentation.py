"""
Augmentation pipeline for hair-DiT training.

All augmentations operate on the sample dict:
  {sketch, matte, target, img, filename}

Key design constraints:
  - StrokeColorSampler: per-stroke color sampling from actual target hair pixels
    (follows SketchHairSalon paper: each iteration samples a random pixel from
     the corresponding target region → color correspondence is maintained)
  - MatteBoundaryPerturbation: recomputes target after matte warp
"""

from __future__ import annotations

import random

import kornia.filters as KF
import kornia.morphology as KM
import torch
import torch.nn.functional as F

from .utils import soft_composite


# ---------------------------------------------------------------------------
# 1. Stroke Color Sampler  (SketchHairSalon 방식)
# ---------------------------------------------------------------------------

class StrokeColorSampler:
    """
    Per-stroke color sampling from actual target hair pixels.

    SketchHairSalon 논문 방식:
      각 stroke 영역에 대해, 매 iteration마다 target 이미지의 해당 위치 픽셀 중
      무작위로 1개를 샘플링하여 stroke 색으로 할당.

    효과:
      - 색 대응 유지: stroke 색 ∈ 실제 머리 색 범위 → 모델이 색 correspondence 학습
      - 데이터 증강: 같은 구조라도 매 iteration 색이 미세하게 달라져 다양성 확보

    Args:
        p:              적용 확률
        min_pixels:     matte 내부 유효 픽셀이 이 값 미만이면 stroke group 전체를 제거
        quantize_bits:  deprecated. flat-color exact RGB grouping에서는 사용하지 않음
    """

    def __init__(self, p: float = 1.0, min_pixels: int = 10, quantize_bits: int | None = None):
        self.p = p
        self.min_pixels = min_pixels
        # 조건부 dark-stroke floor: 대표색이 floor보다 어두우면 (7,7,10)으로 올림.
        # floor보다 밝은 색은 건드리지 않으므로(threshold = floor의 max 채널) 절대 어둡게 만들지 않는다.
        self.floor_value = torch.tensor([7.0, 7.0, 10.0]) / 255.0  # (3,)
        self.floor_threshold = self.floor_value.max().item()       # 10/255
        self._deprecated_quantize_bits = quantize_bits

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"]        # (3, H, W) float32 [0,1]
        sketch_u8 = sample["sketch_u8"]  # (3, H, W) uint8
        img_u8 = sample["img_u8"]        # (3, H, W) uint8
        matte_u8 = sample["matte_u8"]    # (1, H, W) uint8

        sketch_aug = self._resample_colors(sketch, sketch_u8, img_u8, matte_u8)
        return {**sample, "sketch": sketch_aug}

    def _resample_colors(
        self,
        sketch: torch.Tensor,
        sketch_u8: torch.Tensor,
        img_u8: torch.Tensor,
        matte_u8: torch.Tensor,
    ) -> torch.Tensor:
        flat = sketch_u8.view(3, -1).T  # (N, 3)
        unique_colors = torch.unique(flat, dim=0)  # (K, 3)

        out = sketch.clone()
        matte_mask = matte_u8[0] > 0

        for color in unique_colors:
            r, g, b = color.tolist()

            # 검은 배경(non-hair stroke)은 제외
            if r == 0 and g == 0 and b == 0:
                continue

            group_mask = (
                (sketch_u8[0] == r) &
                (sketch_u8[1] == g) &
                (sketch_u8[2] == b)
            )
            valid_mask = group_mask & matte_mask
            outside_mask = group_mask & (~matte_mask)

            if valid_mask.sum().item() < self.min_pixels:
                out[:, group_mask] = 0.0
                continue

            hair_pixels = img_u8[:, valid_mask].float() / 255.0  # (3, M)

            # SHS 스타일: 1/3 random pixel, 2/3 mean color
            if random.randint(0, 2) == 0:
                idx = random.randint(0, hair_pixels.shape[1] - 1)
                sampled_color = hair_pixels[:, idx]
            else:
                sampled_color = hair_pixels.mean(dim=1)

            if sampled_color.max().item() < self.floor_threshold:
                # 근-검정 색만 floor로 rescue. 통째 교체가 아니라 채널별 상향(torch.maximum)이라
                # 어느 채널도 어두워지지 않고 원색의 미세 구조를 보존한다.
                sampled_color = torch.maximum(sampled_color, self.floor_value.to(sampled_color))

            out[:, valid_mask] = sampled_color.unsqueeze(1)
            out[:, outside_mask] = 0.0

        return out


# ---------------------------------------------------------------------------
# 2. Sketch Thickness Jitter
# ---------------------------------------------------------------------------

class ThicknessJitter:
    """
    Randomly dilate sketch strokes by 0-2 pixels.

    Args:
        p:          probability of applying augmentation
        max_kernel: maximum dilation kernel size (default 3 → ±1px dilation)
    """

    def __init__(self, p: float = 0.5, max_kernel: int = 3):
        self.p = p
        self.max_kernel = max_kernel  # must be odd

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        sketch = sample["sketch"].unsqueeze(0)  # (1, 3, H, W)
        k = random.choice([k for k in range(3, self.max_kernel + 1, 2)])
        kernel = torch.ones(k, k, device=sketch.device)
        sketch_dilated = KM.dilation(sketch, kernel).squeeze(0)
        return {**sample, "sketch": sketch_dilated.clamp(0, 1)}


# ---------------------------------------------------------------------------
# 3. Matte Boundary Perturbation
# ---------------------------------------------------------------------------

class MatteBoundaryPerturbation:
    """
    Apply small elastic deformation to the matte, then recompute target.

    Args:
        p:         probability of applying augmentation
        amplitude: max pixel displacement (default 4)
        sigma:     smoothness of displacement field (default 10)
    """

    def __init__(self, p: float = 0.3, amplitude: float = 4.0, sigma: float = 10.0):
        self.p = p
        self.amplitude = amplitude
        self.sigma = sigma

    def __call__(self, sample: dict) -> dict:
        if random.random() > self.p:
            return sample

        matte = sample["matte"].unsqueeze(0)   # (1, 1, H, W)
        img   = sample["img"].unsqueeze(0)     # (1, 3, H, W)

        H, W = matte.shape[-2], matte.shape[-1]

        noise = torch.randn(1, 2, H, W, device=matte.device) * self.amplitude
        kernel_size = int(6 * self.sigma / H * H) | 1
        kernel_size = max(kernel_size, 3)
        noise = KF.gaussian_blur2d(noise, (kernel_size, kernel_size), (self.sigma, self.sigma))

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=matte.device),
            torch.linspace(-1, 1, W, device=matte.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        disp = noise.permute(0, 2, 3, 1)
        disp[..., 0] /= (W / 2)
        disp[..., 1] /= (H / 2)
        grid_warped = (grid + disp).clamp(-1, 1)

        matte_warped = F.grid_sample(matte.float(), grid_warped, mode="bilinear", align_corners=True)
        matte_warped = matte_warped.squeeze(0).clamp(0, 1)

        target_warped = soft_composite(sample["img"], matte_warped)
        return {**sample, "matte": matte_warped, "target": target_warped}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

class ComposeAug:
    """Sequential composition of augmentations."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_augmentation_pipeline(phase: str = "pretrain") -> ComposeAug:
    """
    Build the augmentation pipeline for a given training phase.

    StrokeColorSampler (p=1.0):
      매 iteration마다 target 실제 픽셀을 샘플링하여 stroke 색 할당.
      AppearanceJitter 제거: target 색을 흔들면 stroke↔target 색 대응이 깨짐.

    Args:
        phase: "pretrain" (unbraid) or "finetune" (braid)
    """
    if phase == "pretrain":
        return ComposeAug([
            StrokeColorSampler(p=1.0),
            ThicknessJitter(p=0.5),
            MatteBoundaryPerturbation(p=0.3),
        ])
    elif phase == "finetune":
        return ComposeAug([
            StrokeColorSampler(p=1.0),
            ThicknessJitter(p=0.3),
            MatteBoundaryPerturbation(p=0.2),
        ])
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be 'pretrain' or 'finetune'.")
