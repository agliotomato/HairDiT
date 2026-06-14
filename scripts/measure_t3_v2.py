"""[0613]T3_v2.md 보고용 정량 측정 (CM_1005 + CM_1101, 다중 모델).

측정 정의:
  - LPIPS(A, B) hair-masked
    = LPIPS(pred_A, pred_B), matte 영역만 alpha-composite 후 비교.
      외형 누설 의존도 직접 지표.
  - region IoU (B, matte 영역)
    = |pred_hair ∩ matte| / |pred_hair ∪ matte|,
      pred_hair = (pred_B 와 face_B 의 픽셀 Lab ΔE > θ) 영역.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_matte_fit import hair_mask_pixel_diff, metrics
from scripts.eval_metrics import hair_masked_lpips


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


CASES = [
    {
        "stem":   "CM_1005",
        "matte":  ROOT / "staging/T3_v3/matte/CM_1005.png",
        "face_b": ROOT / "staging/T3_v3/face_B/CM_1005.png",
        "models": [
            ("mcs1 (Ours)",        ROOT / "custom_results/T3_v3/mcs1_A/CM_1005.png",
                                   ROOT / "custom_results/T3_v3/mcs1_B/CM_1005.png"),
            ("mcs3 (Sketch-only)", ROOT / "custom_results/T3_v3/mcs3_A/CM_1005.png",
                                   ROOT / "custom_results/T3_v3/mcs3_B/CM_1005.png"),
            ("mcs5 (Raw-only)",    ROOT / "custom_results/T3_v3/mcs5_A/CM_1005.png",
                                   ROOT / "custom_results/T3_v3/mcs5_B/CM_1005.png"),
        ],
    },
    {
        "stem":   "CM_1101",
        "matte":  ROOT / "data/unbraid/matte/test/CM_1101.png",
        "face_b": ROOT / "data/unbraid/img/CM_1101.png",
        "models": [
            ("mcs1 (Ours)",        ROOT / "custom_results/T3_v2/mcs1_A/CM_1101.png",
                                   ROOT / "custom_results/T3_v2/mcs1_B/CM_1101.png"),
            ("mcs2 (Ours+Gate)",   ROOT / "custom_results/T3_v3/mcs2_A/CM_1101.png",
                                   ROOT / "custom_results/T3_v3/mcs2_B/CM_1101.png"),
            ("mcs3 (Sketch-only)", ROOT / "custom_results/T3_v2/mcs3_A/CM_1101.png",
                                   ROOT / "custom_results/T3_v2/mcs3_B/CM_1101.png"),
        ],
    },
]


def measure_one(pa_path: Path, pb_path: Path, matte_path: Path,
                face_b_path: Path, delta_e: float = 10.0) -> tuple[float, float]:
    matte  = load_l(matte_path)
    face_b = load_rgb(face_b_path)
    pa     = load_rgb(pa_path)
    pb     = load_rgb(pb_path)
    h, w   = matte.shape[:2]
    pa     = resize_to(pa, h, w)
    pb     = resize_to(pb, h, w)
    face_b = resize_to(face_b, h, w)

    lpips_ab = hair_masked_lpips(pa, pb, matte)
    hair = hair_mask_pixel_diff(pb, face_b, delta_e)
    ri   = metrics(hair, matte > 127)["region_iou"]
    return float(lpips_ab), float(ri)


def main():
    for c in CASES:
        print("=" * 60)
        print(f" {c['stem']}")
        print("=" * 60)
        print(f"  {'model':<22}{'LPIPS(A,B)↓':>14}{'region IoU(B)↑':>17}")
        print("  " + "-" * 53)
        for label, pa_p, pb_p in c["models"]:
            if pb_p is None or not pb_p.exists():
                print(f"  {label:<22}{'N/A':>14}{'N/A':>17}    (missing B)")
                continue
            # mcs5 etc: A condition 없음 → region IoU 만
            if pa_p is None:
                matte  = load_l(c["matte"])
                face_b = load_rgb(c["face_b"])
                pb     = load_rgb(pb_p)
                h, w   = matte.shape[:2]
                pb     = resize_to(pb,     h, w)
                face_b = resize_to(face_b, h, w)
                from scripts.eval_matte_fit import hair_mask_pixel_diff, metrics
                hair = hair_mask_pixel_diff(pb, face_b, 10.0)
                ri   = metrics(hair, matte > 127)["region_iou"]
                print(f"  {label:<22}{'N/A':>14}{ri:>17.4f}")
                continue
            if not pa_p.exists():
                print(f"  {label:<22}{'N/A':>14}{'N/A':>17}    (missing A)")
                continue
            lp, ri = measure_one(pa_p, pb_p, c["matte"], c["face_b"])
            print(f"  {label:<22}{lp:>14.4f}{ri:>17.4f}")
        print()


if __name__ == "__main__":
    main()
