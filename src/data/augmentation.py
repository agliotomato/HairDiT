"""
Augmentation pipeline for hair-DiT training.

All augmentations operate on the sample dict:
  {sketch, matte, target, img, filename}

Key design constraints:
  - StrokeColorSampler: per-stroke color sampling from actual target hair pixels
    (follows SketchHairSalon paper: each iteration samples a random pixel from
     the corresponding target region → color correspondence is maintained)
  - MatteBoundaryPerturbation: recomputes target after matte warp
  - DensifyAug (run5): SHS 기하 마스크 로드 + 색 전파. 반드시 StrokeColorSampler '뒤'.
    epoch 단위 라운드로빈이므로 트레이너가 매 epoch ComposeAug.set_epoch(epoch)을 호출해야 한다.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import kornia.filters as KF
import kornia.morphology as KM
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
# 4. Stroke Densification  (SHS auto-completion, run5 밀도 혼합 증강)
# ---------------------------------------------------------------------------

class DensifyAug:
    """사전 생성된 SHS 기하 마스크를 로드해 추가 stroke에 색을 전파한다.

    reports/2026-08-05-run5-density-training-instructions.md §4~5.

    마스크는 `scripts/preprocess/densify_shs.py masks`로 오프라인 생성한다. 기하는 색과
    무관하므로 1회만 구우면 되고, 색 전파만 매 스텝 여기서 수행한다.

    두 가지 제약이 설계를 결정한다:
      - **StrokeColorSampler 바로 뒤에 배치해야 한다**(§4-4). StrokeColorSampler는 sketch의
        exact RGB 그룹 단위로 색을 다시 뽑으므로, 색 전파가 먼저 돌면 추가 stroke만 옛 색을
        갖는 불일치가 생긴다.
      - **epoch 단위 라운드로빈이며 샘플별 무작위가 아니다**(§4-2). 한 epoch 안의 모든 샘플이
        동일한 threshold를 쓴다. 트레이너가 매 epoch 시작 시 `set_epoch(epoch)`을 **0-based**
        루프 변수로 호출해야 한다 — 1-based를 넘기면 매핑이 한 칸 밀린다.

    색 전파는 `scripts/preprocess/densify_shs.propagate_color`와 동일한 semantics다
    (최근접 원본 stroke 픽셀의 색 복사). 기하(최근접 라벨)만 cv2로 얻고 색은 float 텐서에서
    gather하므로 uint8 양자화 손실이 없다. 원본 stroke는 `sketch.clone()`에서 시작해 추가
    픽셀만 덮으므로 구조적으로 보존된다(SHS §6.4: 수동 annotation이 hair wisp junction 형성에 필수).

    Args:
        mask_root:  마스크 루트. 하위에 T{threshold}/{stem}.png (0/255 이진)
        thresholds: epoch 라운드로빈 순서. None = ∞(원본 sketch, densification 없음)
    """

    def __init__(
        self,
        mask_root: str = "data/densify_masks",
        thresholds: tuple = (None, 21, 15, 12),
    ):
        self.mask_root = Path(mask_root)
        self.thresholds = list(thresholds)
        if not self.thresholds:
            raise ValueError("thresholds가 비어 있음")
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """0-based epoch 인덱스. trainer가 매 epoch 시작 시 호출한다."""
        self.epoch = epoch

    def current_threshold(self):
        return self.thresholds[self.epoch % len(self.thresholds)]

    def __call__(self, sample: dict) -> dict:
        t = self.current_threshold()
        if t is None:                                   # ∞ = 원본. 0으로 로깅
            return {**sample, "densify_t": 0}

        sketch = sample["sketch"]                       # (3,H,W) float32 [0,1], 재착색 완료
        path = self.mask_root / f"T{t}" / f"{sample['filename']}.png"
        if not path.exists():
            # 조용히 원본으로 폴백하면 "밀도 혼합으로 학습했다"는 전제가 로그상 참인 채로
            # 실제로는 깨진다. 반드시 크래시시킨다.
            raise FileNotFoundError(f"densify 마스크 없음: {path}")

        mask = torch.from_numpy(np.array(Image.open(path))) > 0     # (H,W) bool
        if mask.shape != sketch.shape[1:]:
            raise ValueError(
                f"mask/sketch 해상도 불일치: {path} {tuple(mask.shape)} vs {tuple(sketch.shape[1:])}"
            )

        src = sketch.max(dim=0).values > 0              # SHS 내부 기준 sk_gray>0과 동일
        add = mask & (~src)                             # 원본과 겹치면 원본 우선
        if not src.any() or not add.any():
            return {**sample, "densify_t": t}

        # 최근접 원본 stroke 픽셀의 인덱스 (기하만 cv2로 계산)
        src_np = src.numpy()
        add_np = add.numpy()
        _, labels = cv2.distanceTransformWithLabels(
            (~src_np).astype(np.uint8), cv2.DIST_L2, 3,
            labelType=cv2.DIST_LABEL_PIXEL)             # 0 = stroke(시드)
        ys, xs = np.nonzero(src_np)
        lut = np.zeros(labels.max() + 1, dtype=np.int64)
        lut[labels[ys, xs]] = np.arange(len(ys))        # label → 시드 인덱스
        ny, nx = np.nonzero(add_np)
        seed = lut[labels[ny, nx]]

        out = sketch.clone()                            # 원본 stroke는 건드리지 않는다
        out[:, torch.from_numpy(ny), torch.from_numpy(nx)] = \
            sketch[:, torch.from_numpy(ys[seed]), torch.from_numpy(xs[seed])]
        return {**sample, "sketch": out, "densify_t": t}


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

class ComposeAug:
    """Sequential composition of augmentations."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def set_epoch(self, epoch: int):
        """epoch 의존 augmentation(DensifyAug)에 0-based epoch을 전파한다."""
        for t in self.transforms:
            if hasattr(t, "set_epoch"):
                t.set_epoch(epoch)

    def __call__(self, sample: dict) -> dict:
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_augmentation_pipeline(
    phase: str = "pretrain",
    densify: dict | None = None,
) -> ComposeAug:
    """
    Build the augmentation pipeline for a given training phase.

    StrokeColorSampler (p=1.0):
      매 iteration마다 target 실제 픽셀을 샘플링하여 stroke 색 할당.
      AppearanceJitter 제거: target 색을 흔들면 stroke↔target 색 대응이 깨짐.

    Args:
        phase:   "pretrain" (unbraid) or "finetune" (braid)
        densify: run5 밀도 혼합 증강 설정 (config의 training.densify). None이거나
                 enabled가 false면 파이프라인이 run4와 완전히 동일하다.
                 densification은 unbraid 전용이므로(SHS는 braid에 절차적 3D 모델 사용)
                 finetune 단계에는 붙이지 않는다.
    """
    dens = []
    if densify and densify.get("enabled", False):
        dens = [DensifyAug(
            mask_root=densify.get("mask_root", "data/densify_masks"),
            thresholds=tuple(densify.get("thresholds", (None, 21, 15, 12))),
        )]

    if phase == "pretrain":
        return ComposeAug([
            StrokeColorSampler(p=1.0),
            *dens,                       # ← StrokeColorSampler 바로 뒤여야 한다 (§4-4)
            ThicknessJitter(p=0.5),
            MatteBoundaryPerturbation(p=0.3),
        ])
    elif phase == "finetune":
        if dens:
            raise ValueError(
                "densify는 unbraid(pretrain) 전용이다. finetune에서 켤 수 없다 (지침서 §3)."
            )
        return ComposeAug([
            StrokeColorSampler(p=1.0),
            ThicknessJitter(p=0.3),
            MatteBoundaryPerturbation(p=0.2),
        ])
    else:
        raise ValueError(f"Unknown phase: {phase}. Must be 'pretrain' or 'finetune'.")
