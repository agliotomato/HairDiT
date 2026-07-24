"""
HairRegionDataset

Loads (sketch, matte, target) triplets where:
  target = img * (matte / 255.0)  — hair region on black background

Supported splits:
  "unbraid_train", "unbraid_test", "braid_train", "braid_test"
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import soft_composite

DATASET_ROOT = Path(__file__).parent.parent.parent / "dataset"


def _build_stem_index(img_dir: Path) -> list[str]:
    """Return sorted list of file stems (without extension)."""
    stems = sorted(p.stem for p in img_dir.glob("*.png"))
    if not stems:
        raise FileNotFoundError(f"No PNG files found in {img_dir}")
    return stems


class HairRegionDataset(Dataset):
    """
    Each item:
        sketch:   (3, H, W)  float32 [0, 1]  — colored sketch
        matte:    (1, H, W)  float32 [0, 1]  — soft alpha matte
        target:   (3, H, W)  float32 [0, 1]  — img * matte (hair region)
        filename: str
    """

    VALID_SPLITS = ("unbraid_train", "unbraid_test", "braid_train", "braid_test")

    def __init__(
        self,
        split: str,
        image_size: int = 512,
        augmentation: Optional[Callable] = None,
        dataset_root: Optional[Path] = None,
    ):
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {self.VALID_SPLITS}, got '{split}'")
        if image_size != 512:
            raise ValueError(f"HairRegionDataset only supports image_size=512 for raw uint8 sketch recolor, got {image_size}")

        style, subset = split.rsplit("_", 1)
        root = Path(dataset_root) if dataset_root else DATASET_ROOT
        base = root / style

        self.img_dir    = base / "img"    / subset
        self.sketch_dir = base / "sketch" / subset
        self.matte_dir  = base / "matte"  / subset

        for d in [self.img_dir, self.sketch_dir, self.matte_dir]:
            if not d.exists():
                raise FileNotFoundError(f"Directory not found: {d}")

        self.stems = _build_stem_index(self.img_dir)
        self.augmentation = augmentation
        self.image_size = image_size

        # Basic transform: PIL → tensor [0,1], resize if needed
        self._to_tensor_rgb = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),   # [0,255] → [0,1], HWC → CHW
        ])
        self._to_tensor_gray = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.stems)

    @staticmethod
    def _pil_to_u8_tensor(image: Image.Image, *, gray: bool = False) -> torch.Tensor:
        """PIL image -> CHW uint8 tensor without normalization."""
        arr = np.asarray(image, dtype=np.uint8)
        if gray:
            arr = arr[None, ...]          # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)  # (3, H, W)
        return torch.from_numpy(arr.copy())

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        img    = Image.open(self.img_dir    / f"{stem}.png").convert("RGB")
        sketch = Image.open(self.sketch_dir / f"{stem}.png").convert("RGB")
        matte  = Image.open(self.matte_dir  / f"{stem}.png").convert("L")   # grayscale

        img_u8    = self._pil_to_u8_tensor(img)
        sketch_u8 = self._pil_to_u8_tensor(sketch)
        matte_u8  = self._pil_to_u8_tensor(matte, gray=True)

        img_t    = self._to_tensor_rgb(img)     # (3, H, W)
        sketch_t = self._to_tensor_rgb(sketch)  # (3, H, W)
        matte_t  = self._to_tensor_gray(matte)  # (1, H, W), values [0,1]

        # target = hair region (soft composite)
        target_t = soft_composite(img_t, matte_t)  # (3, H, W)

        sample = {
            "sketch":   sketch_t,
            "matte":    matte_t,
            "target":   target_t,
            "img":      img_t,      # kept for composite.py (Step 2)
            "sketch_u8": sketch_u8,
            "img_u8":    img_u8,
            "matte_u8":  matte_u8,
            "filename": stem,
        }

        if self.augmentation is not None:
            sample = self.augmentation(sample)

        return sample


def select_fixed_indices(n_items: int, n_select: int, seed: int) -> list[int]:
    """[0, n_items) 중 n_select개를 고정 seed로 결정론적 선택.

    HairRegionDataset.stems는 이미 정렬돼 있으므로(_build_stem_index), 이 인덱스를
    그대로 데이터셋 인덱싱에 쓰면 재실행해도 항상 동일한 stem 집합이 선택된다
    (held-out perceptual val 평가셋 고정용, [0724] planning §4-1).
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_items, generator=g).tolist()
    return sorted(perm[:n_select])


class StratifiedBatchSampler:
    """매 배치 unbraid n_a장 + braid n_b장 고정. s를 ~55로 안정화(scale-sync 리스크 3·4 대비, [0724] planning §5-2)."""

    def __init__(self, n_unbraid, n_braid, split_idx, na=8, nb=8, seed=0, drop_last=True):
        self.ua = list(range(split_idx)); self.ba = list(range(split_idx, split_idx + n_braid))
        self.na, self.nb, self.seed, self.epoch = na, nb, seed, 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch); self.epoch += 1
        ua = [self.ua[i] for i in torch.randperm(len(self.ua), generator=g)]
        ba = [self.ba[i] for i in torch.randperm(len(self.ba), generator=g)]
        nb_batches = min(len(ua) // self.na, len(ba) // self.nb)
        for i in range(nb_batches):
            yield ua[i * self.na:(i + 1) * self.na] + ba[i * self.nb:(i + 1) * self.nb]

    def __len__(self):
        return min(len(self.ua) // self.na, len(self.ba) // self.nb)
