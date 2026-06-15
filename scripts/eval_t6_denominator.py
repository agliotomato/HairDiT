"""T6 분모 T 실측 — 원본 헤어 영역의 ΔL·Δhue·Δa·Δb (각 변형별).

design.md L66 핵심:
  "tracking ratio 분모 T = 실측" — 이론 gain 아님.
  변환을 원본 이미지 헤어 영역에 적용했을 때 실제 생긴 Δhue/ΔL.

계산:
  Lab(orig hair)  vs  Lab(warm hair)  → Δa, Δb, ΔL
  마찬가지 cool, dim
  hair 영역 = matte (alpha-weighted 평균)
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
VARIANTS = ["origin", "warm", "cool", "dim"]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def mean_lab_in_mask(img: np.ndarray, matte: np.ndarray) -> np.ndarray:
    lab = rgb_to_lab(img)
    a = (matte.astype(np.float64) / 255.0)[..., None]
    w = a.sum() * 1.0  # broadcasted over 3 channels
    if w < 1e-6:
        return np.zeros(3)
    return (lab * a).sum(axis=(0, 1)) / a.sum()


def main():
    rows = []
    print(f"{'stem':<10}{'variant':<8}{'ΔL':>10}{'Δa':>10}{'Δb':>10}{'Δhue°':>10}{'ΔChroma':>10}")
    print("-" * 68)
    for stem in STEMS:
        matte = load_l(ROOT / f"staging/T6_v1/matte/{stem}.png")
        orig  = load_rgb(ROOT / f"staging/T6_v1/face_origin/{stem}.png")
        lab_o = mean_lab_in_mask(orig, matte)

        rows.append({"stem": stem, "variant": "origin",
                     "L": lab_o[0], "a": lab_o[1], "b": lab_o[2],
                     "dL": 0, "da": 0, "db": 0, "dhue_deg": 0, "dC": 0})
        print(f"{stem:<10}{'origin':<8}{lab_o[0]:>10.2f}{lab_o[1]:>10.2f}{lab_o[2]:>10.2f}"
              f"{'-':>10}{'-':>10}")

        for v in ["warm", "cool", "dim"]:
            img = load_rgb(ROOT / f"staging/T6_v1/face_{v}/{stem}.png")
            lab_v = mean_lab_in_mask(img, matte)
            dL = lab_v[0] - lab_o[0]
            da = lab_v[1] - lab_o[1]
            db = lab_v[2] - lab_o[2]
            # hue (Lab) = atan2(b, a)
            h_o = np.degrees(np.arctan2(lab_o[2], lab_o[1]))
            h_v = np.degrees(np.arctan2(lab_v[2], lab_v[1]))
            dhue = (h_v - h_o + 540) % 360 - 180
            C_o = np.sqrt(lab_o[1] ** 2 + lab_o[2] ** 2)
            C_v = np.sqrt(lab_v[1] ** 2 + lab_v[2] ** 2)
            dC = C_v - C_o
            rows.append({"stem": stem, "variant": v,
                         "L": lab_v[0], "a": lab_v[1], "b": lab_v[2],
                         "dL": dL, "da": da, "db": db,
                         "dhue_deg": dhue, "dC": dC})
            print(f"{stem:<10}{v:<8}{dL:>10.2f}{da:>10.2f}{db:>10.2f}"
                  f"{dhue:>10.2f}{dC:>10.2f}")

    # 변형별 평균
    print("\n=== 변형별 평균 (8 stems, 헤어 영역) ===")
    print(f"{'variant':<8}{'ΔL':>10}{'Δa':>10}{'Δb':>10}{'Δhue°':>10}{'ΔChroma':>10}")
    print("-" * 58)
    for v in ["warm", "cool", "dim"]:
        ms = [r for r in rows if r["variant"] == v]
        dL  = sum(r["dL"]  for r in ms) / len(ms)
        da  = sum(r["da"]  for r in ms) / len(ms)
        db  = sum(r["db"]  for r in ms) / len(ms)
        dhu = sum(r["dhue_deg"] for r in ms) / len(ms)
        dC  = sum(r["dC"]  for r in ms) / len(ms)
        print(f"{v:<8}{dL:>10.2f}{da:>10.2f}{db:>10.2f}{dhu:>10.2f}{dC:>10.2f}")

    out_dir = ROOT / "custom_results/T6_v1/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "T_denominator.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "variant", "L", "a", "b",
                                          "dL", "da", "db", "dhue_deg", "dC"])
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_dir / 'T_denominator.csv'}")


if __name__ == "__main__":
    main()
