"""T5 — Stroke 배치 대응 측정 (내부 확인용, design.md L106-108).

타겟 B + raw colored sparse sketch (recolor X).

측정: stroke 주변 dominant hue 일치율
  각 stroke 색 → 그 stroke 위치의 pred hair 색 → Lab hue 차이(°)
  작을수록 stroke 색 지시를 잘 따른 것.
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

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
MODELS = [("mcs1", "Ours"), ("mcs2", "Ours+Gate"),
          ("mcs3", "Sketch-only"), ("mcs4", "Sketch-only+Gate"),
          ("mcs5", "Raw-only"), ("mcs6", "Matte-CNN-only")]


def load_rgb(p): return np.array(Image.open(p).convert("RGB"))
def load_l(p):   return np.array(Image.open(p).convert("L"))


def rgb_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def delta_hue_signed(h_v, h_o):
    return (h_v - h_o + 540) % 360 - 180


def stroke_masks(sketch_rgb, q_bits=5, min_px=20):
    shift = 8 - q_bits
    q = (sketch_rgb >> shift) << shift
    flat = q.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    out = {}
    for c in uniq:
        if c[0] == 0 and c[1] == 0 and c[2] == 0:
            continue
        m = (q[:, :, 0] == c[0]) & (q[:, :, 1] == c[1]) & (q[:, :, 2] == c[2])
        if m.sum() < min_px:
            continue
        out[tuple(int(v) for v in c)] = m
    return out


def measure(stem, mcs):
    sketch = load_rgb(ROOT / f"staging/T5_v1/sketch_sparse/{stem}.png")
    pred   = load_rgb(ROOT / f"custom_results/T5_v1/{mcs}/{stem}.png")
    matte  = load_l(ROOT / f"staging/T5_v1/matte/{stem}.png")
    h, w = matte.shape[:2]
    if pred.shape[:2] != (h, w):
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    if sketch.shape[:2] != (h, w):
        sketch = cv2.resize(sketch, (w, h), interpolation=cv2.INTER_LINEAR)

    masks = stroke_masks(sketch)
    if not masks:
        return None
    pred_lab = rgb_to_lab(pred)
    matte_bin = matte > 127

    dhues = []
    sizes = []
    for color_q, m in masks.items():
        c_arr = np.array(color_q, dtype=np.uint8).reshape(1, 1, 3)
        c_lab = rgb_to_lab(c_arr)[0, 0]
        c_hue = float(np.degrees(np.arctan2(c_lab[2], c_lab[1])))

        m_v = m & matte_bin
        if m_v.sum() < 10:
            continue
        p_lab = pred_lab[m_v].mean(axis=0)
        p_hue = float(np.degrees(np.arctan2(p_lab[2], p_lab[1])))

        dhues.append(abs(delta_hue_signed(p_hue, c_hue)))
        sizes.append(int(m.sum()))

    if not dhues:
        return None
    total = sum(sizes)
    wavg = sum(d * s for d, s in zip(dhues, sizes)) / total
    return {"stem": stem, "model": mcs,
            "n_strokes": len(dhues), "mean_abs_dhue": wavg}


def main():
    rows = []
    print(f"{'stem':<10}{'model':<6}{'n':>5}{'|Δhue°|↓':>12}")
    print("-" * 35)
    for stem in STEMS:
        for mcs, _ in MODELS:
            r = measure(stem, mcs)
            if r is None:
                continue
            rows.append(r)
            print(f"{stem:<10}{mcs:<6}{r['n_strokes']:>5}{r['mean_abs_dhue']:>12.2f}")

    print(f"\n=== 모델별 평균 |Δhue°| (8 stems, ↓=stroke 색 지시 잘 따름) ===")
    print(f"{'model':<22}{'|Δhue°| (↓)':>14}")
    print("-" * 36)
    for mcs, label in MODELS:
        sub = [r for r in rows if r["model"] == mcs]
        avg = sum(r["mean_abs_dhue"] for r in sub) / len(sub) if sub else float("nan")
        print(f"{mcs} ({label:<16}){avg:>14.2f}")

    out_dir = ROOT / "custom_results/T5_v1/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dominant_hue.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "model", "n_strokes", "mean_abs_dhue"])
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_dir / 'dominant_hue.csv'}")


if __name__ == "__main__":
    main()
