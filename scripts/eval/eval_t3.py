"""T3_v4 정량 측정 — 8 stems × mcs1/3/5 × LPIPS(A,B) + region IoU (B, matte 영역).

측정 정의:
  - LPIPS(A, B) = LPIPS(pred_A, pred_B), matte 영역 alpha-composite 후 비교.
  - region IoU (B) = |pred_hair ∩ matte| / |pred_hair ∪ matte|,
                    pred_hair = pred_B 와 face_B 의 Lab ΔE > 10 (BLD pixel-diff).
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
from scripts.eval_metrics import hair_masked_lpips


STEMS  = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
          "CM_1077", "CM_1082", "CM_1101", "CM_1106"]

MODELS = [("mcs1", "Ours"),
          ("mcs2", "Ours+Gate"),
          ("mcs3", "Sketch-only"),
          ("mcs4", "Sketch-only+Gate"),
          ("mcs5", "Raw-only"),
          ("mcs6", "Matte-CNN-only")]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def measure_pair(stem: str, mkey: str) -> dict | None:
    pa_p = ROOT / f"custom_results/T3_v4/{mkey}_A/{stem}.png"
    pb_p = ROOT / f"custom_results/T3_v4/{mkey}_B/{stem}.png"
    if not (pa_p.exists() and pb_p.exists()):
        return None
    matte  = load_l(ROOT / f"staging/T3_v4/matte/{stem}.png")
    face_b = load_rgb(ROOT / f"staging/T3_v4/face_B/{stem}.png")
    pa     = load_rgb(pa_p)
    pb     = load_rgb(pb_p)
    h, w   = matte.shape[:2]
    pa     = resize_to(pa,     h, w)
    pb     = resize_to(pb,     h, w)
    face_b = resize_to(face_b, h, w)

    lpips_ab = hair_masked_lpips(pa, pb, matte)
    hair = hair_mask_pixel_diff(pb, face_b, 10.0)
    ri   = metrics(hair, matte > 127)["region_iou"]
    return {"lpips_ab": float(lpips_ab), "region_iou": float(ri)}


def main():
    rows = []
    for stem in STEMS:
        print(f"\n--- {stem} ---")
        print(f"  {'model':<14}{'LPIPS(A,B)↓':>14}{'region IoU(B)↑':>17}")
        print("  " + "-" * 45)
        for mkey, mlabel in MODELS:
            r = measure_pair(stem, mkey)
            if r is None:
                print(f"  {mlabel:<14}{'N/A':>14}{'N/A':>17}    (missing)")
                rows.append({"stem": stem, "model": mlabel,
                             "lpips_ab": None, "region_iou": None})
                continue
            print(f"  {mlabel:<14}{r['lpips_ab']:>14.4f}{r['region_iou']:>17.4f}")
            rows.append({"stem": stem, "model": mlabel, **r})

    out_dir = ROOT / "custom_results/T3_v4/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "model", "lpips_ab", "region_iou"])
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_dir/'per_image.csv'}")


if __name__ == "__main__":
    main()
