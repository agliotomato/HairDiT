"""T3 A→B drop 측정 (design.md L78 충실).

design.md 명시:
  "측정: A→B 에서 metric(Edge IoU, region IoU, 정성) drop 폭"

구현:
  - Edge IoU drop  = Edge_IoU(pred_A, sketch, matte) − Edge_IoU(pred_B, sketch, matte)
  - region IoU drop = region_IoU(pred_A) − region_IoU(pred_B)
    pred_hair = ΔE(pred, face_B) > 10 (face_B 공통 reference, BLD pixel-diff 휴리스틱)

해석:
  - drop ↑ = A→B 전환 시 metric 크게 떨어짐 → 외형 누설 의존 큼
  - design.md 예측: Sketch-only drop ≫ Ours drop
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
from scripts.eval_metrics import canny_edges, edge_iou

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
MODELS = [("mcs1", "Ours"), ("mcs2", "Ours+Gate"),
          ("mcs3", "Sketch-only"), ("mcs4", "Sketch-only+Gate"),
          ("mcs5", "Raw-only"), ("mcs6", "Matte-CNN-only")]


def load_rgb(p):  return np.array(Image.open(p).convert("RGB"))
def load_l(p):    return np.array(Image.open(p).convert("L"))


def resize_to(arr, h, w):
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def measure_one(stem, mcs):
    sketch = load_rgb(ROOT / f"staging/T3_v4/sketch_recolor/{stem}.png")
    matte  = load_l(ROOT  / f"staging/T3_v4/matte/{stem}.png")
    face_b = load_rgb(ROOT / f"staging/T3_v4/face_B/{stem}.png")
    pred_a = load_rgb(ROOT / f"custom_results/T3_v4/{mcs}_A/{stem}.png")
    pred_b = load_rgb(ROOT / f"custom_results/T3_v4/{mcs}_B/{stem}.png")

    h, w = matte.shape[:2]
    sketch = resize_to(sketch, h, w)
    face_b = resize_to(face_b, h, w)
    pred_a = resize_to(pred_a, h, w)
    pred_b = resize_to(pred_b, h, w)

    sk_e = canny_edges(sketch)

    # Edge IoU (vs sketch, matte 영역)
    ei_a = edge_iou(canny_edges(pred_a), sk_e, matte)
    ei_b = edge_iou(canny_edges(pred_b), sk_e, matte)
    ei_drop = ei_a - ei_b

    # region IoU (face_B 공통 reference, BLD pixel-diff)
    matte_bin = matte > 127
    hair_a = hair_mask_pixel_diff(pred_a, face_b, 10.0)
    hair_b = hair_mask_pixel_diff(pred_b, face_b, 10.0)
    ri_a = metrics(hair_a, matte_bin)["region_iou"]
    ri_b = metrics(hair_b, matte_bin)["region_iou"]
    ri_drop = ri_a - ri_b

    return {
        "stem": stem, "model": mcs,
        "edge_iou_A": ei_a, "edge_iou_B": ei_b, "edge_iou_drop": ei_drop,
        "region_iou_A": ri_a, "region_iou_B": ri_b, "region_iou_drop": ri_drop,
    }


def main():
    rows = []
    print(f"{'stem':<10}{'model':<6}{'EI_A':>8}{'EI_B':>8}{'EI_drop':>10}"
          f"{'RI_A':>8}{'RI_B':>8}{'RI_drop':>10}")
    print("-" * 70)
    for stem in STEMS:
        for mcs, _ in MODELS:
            r = measure_one(stem, mcs)
            rows.append(r)
            print(f"{stem:<10}{mcs:<6}"
                  f"{r['edge_iou_A']:>8.4f}{r['edge_iou_B']:>8.4f}{r['edge_iou_drop']:>+10.4f}"
                  f"{r['region_iou_A']:>8.4f}{r['region_iou_B']:>8.4f}{r['region_iou_drop']:>+10.4f}")

    # 모델별 평균 drop
    print("\n=== 모델별 평균 drop (8 stems) ===")
    print(f"{'model':<22}{'EI drop':>12}{'RI drop':>12}")
    print("-" * 46)
    for mcs, label in MODELS:
        sub = [r for r in rows if r["model"] == mcs]
        ei = sum(r["edge_iou_drop"] for r in sub) / len(sub)
        ri = sum(r["region_iou_drop"] for r in sub) / len(sub)
        print(f"{mcs} ({label:<16}){ei:>+12.4f}{ri:>+12.4f}")

    out_dir = ROOT / "custom_results/T3_v4/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "drop_per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_dir / 'drop_per_image.csv'}")


if __name__ == "__main__":
    main()
