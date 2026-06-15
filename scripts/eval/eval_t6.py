"""T6 tracking ratio 측정 (--version 1 or 2).

design.md L91:
  tracking ratio = 출력 헤어(matte 내부)의 hue·luminance 이동량 ÷ 적용한 T

분모 T = 실측 (face_variant vs face_origin, 헤어 영역)
분자 = pred_variant vs pred_origin, 헤어 영역

네 가지 ratio 보고:
  - tracking_L   = (pred Δ L)   / (face Δ L)          — luminance 추종
  - tracking_hue = (pred Δ hue°) / (face Δ hue°)      — Lab hue 각도 추종 (design.md "hue" 정확)
  - tracking_b   = (pred Δ b)   / (face Δ b)          — yellow-blue 축 (보조)
  - tracking_C   = (pred Δ C)   / (face Δ C)          — chroma 추종
"""

from __future__ import annotations

import argparse
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
VARIANTS = ["warm", "cool", "dim"]
MODELS   = ["mcs1", "mcs2", "mcs3", "mcs4", "mcs5", "mcs6"]


def load_rgb(p): return np.array(Image.open(p).convert("RGB"))
def load_l(p):   return np.array(Image.open(p).convert("L"))


def rgb_to_lab(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def mean_lab_in_mask(img, matte):
    lab = rgb_to_lab(img)
    a = (matte.astype(np.float64) / 255.0)[..., None]
    s = a.sum()
    if s < 1e-6:
        return np.zeros(3)
    return (lab * a).sum(axis=(0, 1)) / s


def lab_chroma(lab):  return float(np.sqrt(lab[1] ** 2 + lab[2] ** 2))
def lab_hue(lab):     return float(np.degrees(np.arctan2(lab[2], lab[1])))


def delta_hue(h_v, h_o):
    """signed hue difference normalized to (-180, 180]."""
    return (h_v - h_o + 540) % 360 - 180


def safe_div(num, den, eps=1e-3):
    if abs(den) < eps:
        return float("nan")
    return num / den


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["1", "2"], required=True)
    args = p.parse_args()
    v = f"T6_v{args.version}"

    rows = []
    for stem in STEMS:
        matte = load_l(ROOT / f"staging/{v}/matte/{stem}.png")
        face_o = load_rgb(ROOT / f"staging/{v}/face_origin/{stem}.png")
        lab_face_o = mean_lab_in_mask(face_o, matte)

        for mcs in MODELS:
            pred_o = load_rgb(ROOT / f"custom_results/{v}/{mcs}_origin/{stem}.png")
            lab_p_o = mean_lab_in_mask(pred_o, matte)

            for vname in VARIANTS:
                face_v = load_rgb(ROOT / f"staging/{v}/face_{vname}/{stem}.png")
                pred_v = load_rgb(ROOT / f"custom_results/{v}/{mcs}_{vname}/{stem}.png")
                lab_face_v = mean_lab_in_mask(face_v, matte)
                lab_p_v    = mean_lab_in_mask(pred_v, matte)

                # face deltas (denominator)
                dL_f   = lab_face_v[0] - lab_face_o[0]
                db_f   = lab_face_v[2] - lab_face_o[2]
                dC_f   = lab_chroma(lab_face_v) - lab_chroma(lab_face_o)
                dhue_f = delta_hue(lab_hue(lab_face_v), lab_hue(lab_face_o))

                # pred deltas (numerator)
                dL_p   = lab_p_v[0] - lab_p_o[0]
                db_p   = lab_p_v[2] - lab_p_o[2]
                dC_p   = lab_chroma(lab_p_v) - lab_chroma(lab_p_o)
                dhue_p = delta_hue(lab_hue(lab_p_v), lab_hue(lab_p_o))

                rows.append({
                    "stem": stem, "model": mcs, "variant": vname,
                    "face_dL":   dL_f,   "face_db":   db_f,
                    "face_dC":   dC_f,   "face_dhue": dhue_f,
                    "pred_dL":   dL_p,   "pred_db":   db_p,
                    "pred_dC":   dC_p,   "pred_dhue": dhue_p,
                    "tracking_L":   safe_div(dL_p,   dL_f),
                    "tracking_b":   safe_div(db_p,   db_f),
                    "tracking_C":   safe_div(dC_p,   dC_f),
                    "tracking_hue": safe_div(dhue_p, dhue_f, eps=2.0),
                })

    # 모델별 평균
    print(f"\n=== {v} 모델별 평균 tracking ratio (8 stems × 3 variants = 24 samples) ===")
    print(f"{'model':<6}{'L':>10}{'hue (°)':>10}{'b':>10}{'C':>10}")
    print("-" * 46)
    for mcs in MODELS:
        sub = [r for r in rows if r["model"] == mcs]
        def avg(k):
            vs = [r[k] for r in sub if not np.isnan(r[k])]
            return sum(vs) / len(vs) if vs else float("nan")
        print(f"{mcs:<6}{avg('tracking_L'):>10.3f}"
              f"{avg('tracking_hue'):>10.3f}"
              f"{avg('tracking_b'):>10.3f}"
              f"{avg('tracking_C'):>10.3f}")

    out_dir = ROOT / f"custom_results/{v}/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tracking_ratio.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_dir / 'tracking_ratio.csv'}")


if __name__ == "__main__":
    main()
