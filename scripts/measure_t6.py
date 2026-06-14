"""T6 tracking ratio 측정.

design.md L91:
  tracking ratio = 출력 헤어(matte 내부)의 hue·luminance 이동량 ÷ 적용한 T

분모 T = 실측 (face_variant vs face_origin, 헤어 영역) — measure_t6_T.py 와 동일 logic
분자 = pred_variant vs pred_origin, 헤어 영역

세 가지 ratio 보고:
  - tracking_L  = (pred Δ L) / (face Δ L)        — luminance tracking
  - tracking_b  = (pred Δ b) / (face Δ b)        — hue (yellow-blue) tracking
  - tracking_C  = (pred Δ Chroma) / (face Δ Chroma) — chroma tracking

예측 (design.md L92):
  - Ours (mcs1) ≈ 1 (scene harmonize)
  - Ours+Gate (mcs2) 낮음 (배경 독립)
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
VARIANTS = ["warm", "cool", "dim"]
MODELS   = ["mcs1", "mcs2", "mcs3", "mcs4", "mcs5", "mcs6"]


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
    s = a.sum()
    if s < 1e-6:
        return np.zeros(3)
    return (lab * a).sum(axis=(0, 1)) / s


def lab_chroma(L_a_b: np.ndarray) -> float:
    return float(np.sqrt(L_a_b[1] ** 2 + L_a_b[2] ** 2))


def safe_div(num: float, den: float) -> float:
    if abs(den) < 1e-3:
        return float("nan")
    return num / den


def main():
    rows = []
    print(f"{'stem':<10}{'model':<6}{'variant':<8}"
          f"{'pred ΔL':>10}{'pred Δb':>10}{'pred ΔC':>10}"
          f"{'face ΔL':>10}{'face Δb':>10}{'face ΔC':>10}"
          f"{'L ratio':>10}{'b ratio':>10}{'C ratio':>10}")
    print("-" * 116)

    for stem in STEMS:
        matte = load_l(ROOT / f"staging/T6_v2/matte/{stem}.png")
        face_o = load_rgb(ROOT / f"staging/T6_v2/face_origin/{stem}.png")
        lab_face_o = mean_lab_in_mask(face_o, matte)
        Co = lab_chroma(lab_face_o)

        for mcs in MODELS:
            pred_o = load_rgb(ROOT / f"custom_results/T6_v2/{mcs}_origin/{stem}.png")
            lab_p_o = mean_lab_in_mask(pred_o, matte)
            Co_p = lab_chroma(lab_p_o)

            for v in VARIANTS:
                face_v = load_rgb(ROOT / f"staging/T6_v2/face_{v}/{stem}.png")
                pred_v = load_rgb(ROOT / f"custom_results/T6_v2/{mcs}_{v}/{stem}.png")

                lab_face_v = mean_lab_in_mask(face_v, matte)
                lab_p_v    = mean_lab_in_mask(pred_v, matte)

                # face deltas (denominator T)
                dL_f = lab_face_v[0] - lab_face_o[0]
                db_f = lab_face_v[2] - lab_face_o[2]
                dC_f = lab_chroma(lab_face_v) - Co

                # pred deltas (numerator)
                dL_p = lab_p_v[0] - lab_p_o[0]
                db_p = lab_p_v[2] - lab_p_o[2]
                dC_p = lab_chroma(lab_p_v) - Co_p

                tL = safe_div(dL_p, dL_f)
                tb = safe_div(db_p, db_f)
                tC = safe_div(dC_p, dC_f)

                rows.append({
                    "stem": stem, "model": mcs, "variant": v,
                    "pred_dL": dL_p, "pred_db": db_p, "pred_dC": dC_p,
                    "face_dL": dL_f, "face_db": db_f, "face_dC": dC_f,
                    "tracking_L": tL, "tracking_b": tb, "tracking_C": tC,
                })
                print(f"{stem:<10}{mcs:<6}{v:<8}"
                      f"{dL_p:>10.2f}{db_p:>10.2f}{dC_p:>10.2f}"
                      f"{dL_f:>10.2f}{db_f:>10.2f}{dC_f:>10.2f}"
                      f"{tL:>10.3f}{tb:>10.3f}{tC:>10.3f}")

    # 평균 — 모델별 (모든 stems, 모든 변형)
    print("\n=== 모델별 tracking ratio 평균 (8 stems × 3 variants = 24 samples) ===")
    print(f"{'model':<6}{'L ratio':>12}{'b ratio (hue)':>16}{'C ratio':>12}")
    print("-" * 50)
    for mcs in MODELS:
        sub = [r for r in rows if r["model"] == mcs]
        tLs = [r["tracking_L"] for r in sub if not np.isnan(r["tracking_L"])]
        tbs = [r["tracking_b"] for r in sub if not np.isnan(r["tracking_b"])]
        tCs = [r["tracking_C"] for r in sub if not np.isnan(r["tracking_C"])]
        m_L = sum(tLs) / len(tLs) if tLs else float("nan")
        m_b = sum(tbs) / len(tbs) if tbs else float("nan")
        m_C = sum(tCs) / len(tCs) if tCs else float("nan")
        print(f"{mcs:<6}{m_L:>12.3f}{m_b:>16.3f}{m_C:>12.3f}")

    # 변형별 모델 평균
    print("\n=== 변형별 모델 평균 ===")
    print(f"{'variant':<8}{'model':<6}{'L ratio':>12}{'b ratio':>12}{'C ratio':>12}")
    print("-" * 50)
    for v in VARIANTS:
        for mcs in MODELS:
            sub = [r for r in rows if r["model"] == mcs and r["variant"] == v]
            tLs = [r["tracking_L"] for r in sub if not np.isnan(r["tracking_L"])]
            tbs = [r["tracking_b"] for r in sub if not np.isnan(r["tracking_b"])]
            tCs = [r["tracking_C"] for r in sub if not np.isnan(r["tracking_C"])]
            m_L = sum(tLs) / len(tLs) if tLs else float("nan")
            m_b = sum(tbs) / len(tbs) if tbs else float("nan")
            m_C = sum(tCs) / len(tCs) if tCs else float("nan")
            print(f"{v:<8}{mcs:<6}{m_L:>12.3f}{m_b:>12.3f}{m_C:>12.3f}")

    out_dir = ROOT / "custom_results/T6_v2/eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tracking_ratio.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stem", "model", "variant",
            "pred_dL", "pred_db", "pred_dC",
            "face_dL", "face_db", "face_dC",
            "tracking_L", "tracking_b", "tracking_C",
        ])
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {out_dir / 'tracking_ratio.csv'}")


if __name__ == "__main__":
    main()
