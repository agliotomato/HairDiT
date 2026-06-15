"""T1_v4 정량 측정 — 8 stems × mcs1/6/5/3 × region IoU (B, matte 영역).

측정 정의:
  - region IoU (B) = |pred_hair ∩ matte| / |pred_hair ∪ matte|,
                    pred_hair = pred 와 face_B 의 Lab ΔE > 10 (BLD pixel-diff).

T1 은 face_B 만 → LPIPS(A, B) 측정 불가.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_matte_fit import hair_mask_pixel_diff, metrics


STEMS  = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
          "CM_1077", "CM_1082", "CM_1101", "CM_1106"]

MODELS = [("mcs1", "Ours"),
          ("mcs2", "Ours+Gate"),
          ("mcs6", "Matte-CNN-only"),
          ("mcs5", "Raw-only"),
          ("mcs3", "Sketch-only"),
          ("mcs4", "Sketch-only+Gate")]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def measure(stem: str, mkey: str) -> float | None:
    p = ROOT / f"custom_results/T1_v4/{mkey}/{stem}.png"
    if not p.exists():
        return None
    matte  = load_l(ROOT / f"staging/T1_v4/matte/{stem}.png")
    face_b = load_rgb(ROOT / f"staging/T1_v4/face_B/{stem}.png")
    pred   = load_rgb(p)
    h, w   = matte.shape[:2]
    pred   = resize_to(pred,   h, w)
    face_b = resize_to(face_b, h, w)
    hair = hair_mask_pixel_diff(pred, face_b, 10.0)
    return float(metrics(hair, matte > 127)["region_iou"])


def main():
    rows = []
    for stem in STEMS:
        print(f"\n--- {stem} ---")
        print(f"  {'model':<22}{'region IoU(B)↑':>17}")
        print("  " + "-" * 39)
        for mkey, mlabel in MODELS:
            ri = measure(stem, mkey)
            if ri is None:
                print(f"  {mlabel:<22}{'N/A':>17}    (missing)")
                rows.append({"stem": stem, "model": mlabel, "region_iou": None})
                continue
            print(f"  {mlabel:<22}{ri:>17.4f}")
            rows.append({"stem": stem, "model": mlabel, "region_iou": ri})

    # model 평균
    print("\n=== 모델별 평균 (8 stems) ===")
    print(f"  {'model':<22}{'region IoU(B)↑':>17}")
    print("  " + "-" * 39)
    for _, mlabel in MODELS:
        vals = [r["region_iou"] for r in rows if r["model"] == mlabel
                and r["region_iou"] is not None]
        m = sum(vals) / len(vals) if vals else None
        print(f"  {mlabel:<22}{m:>17.4f}")

    out_dir = ROOT / "custom_results/T1_v4/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "model", "region_iou"])
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_dir/'per_image.csv'}")


if __name__ == "__main__":
    main()
