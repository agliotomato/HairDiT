"""T2 측정 — Edge IoU + hair 영역 chroma 평균.

design.md L103:
  - Edge IoU (vs sketch, matte 영역)
  - hair 영역 chroma 평균 (낮으면 muddy)
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

from scripts.eval_metrics import canny_edges, edge_iou

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
SPARSITY = ["dense", "mid", "sparse"]
MODELS   = ["mcs1", "mcs2", "mcs3", "mcs4", "mcs5", "mcs6"]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def resize_to(arr, h, w):
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def hair_chroma_mean(pred: np.ndarray, matte: np.ndarray) -> float:
    lab = cv2.cvtColor(pred, cv2.COLOR_RGB2LAB).astype(np.float64)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    C = np.sqrt(a ** 2 + b ** 2)
    m = matte.astype(np.float64) / 255.0
    s = m.sum()
    if s < 1e-6:
        return 0.0
    return float((C * m).sum() / s)


def main():
    rows = []
    print(f"{'stem':<10}{'sparsity':<8}{'model':<6}{'Edge IoU↑':>12}{'Chroma↑':>10}")
    print("-" * 46)
    for stem in STEMS:
        matte = load_l(ROOT / f"staging/T2_v1/matte/{stem}.png")
        h, w = matte.shape[:2]
        for sk in SPARSITY:
            sketch = resize_to(load_rgb(ROOT / f"staging/T2_v1/sketch_{sk}/{stem}.png"), h, w)
            sk_e = canny_edges(sketch)
            for mcs in MODELS:
                p = ROOT / f"custom_results/T2_v1/{mcs}_{sk}/{stem}.png"
                pred = resize_to(load_rgb(p), h, w)
                pred_e = canny_edges(pred)
                ei = edge_iou(pred_e, sk_e, matte)
                C = hair_chroma_mean(pred, matte)
                rows.append({"stem": stem, "sparsity": sk, "model": mcs,
                             "edge_iou": float(ei), "chroma": C})
                print(f"{stem:<10}{sk:<8}{mcs:<6}{ei:>12.4f}{C:>10.2f}")

    # 평균 — (sparsity, model)
    print("\n=== (sparsity, model) 평균 (8 stems) ===")
    print(f"{'sparsity':<8}{'model':<6}{'Edge IoU↑':>12}{'Chroma↑':>10}")
    print("-" * 38)
    for sk in SPARSITY:
        for mcs in MODELS:
            sub = [r for r in rows if r["sparsity"] == sk and r["model"] == mcs]
            ei = sum(r["edge_iou"] for r in sub) / len(sub)
            C  = sum(r["chroma"]   for r in sub) / len(sub)
            print(f"{sk:<8}{mcs:<6}{ei:>12.4f}{C:>10.2f}")

    out_dir = ROOT / "custom_results/T2_v1/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "sparsity", "model", "edge_iou", "chroma"])
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_dir / 'per_image.csv'}")


if __name__ == "__main__":
    main()
