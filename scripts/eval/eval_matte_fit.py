"""매트 정합성 분석 — 출력 hair가 GT matte 경계에 얼마나 정확히 맞는가.

휴리스틱 hair mask (matte 제한 없음, 전체 이미지에서):
  CIE Lab 공간에서 pred 픽셀과 face_B(두피 reference)의 ΔE > threshold → hair.

매트 정합성 지표:
  1. fill_rate    = |pred_hair ∩ matte_bin| / |matte_bin|     (matte 안 얼마나 채움, recall)
  2. spillage     = |pred_hair \\ matte_bin| / |pred_hair|     (pred_hair 중 matte 밖 비율, false-positive rate)
  3. region_iou   = |pred_hair ∩ matte_bin| / |pred_hair ∪ matte_bin|   (제한 없는 IoU)
  4. boundary_iou = matte의 경계 band 안에서의 (pred_hair ∩ matte) / (pred_hair ∪ matte)

Figure (단일 stem):
  N rows = N 모델
  cols = [pred output, matte contour overlay on pred, pred_hair mask (binary), GT matte]

Usage:
  python scripts/eval_matte_fit.py \\
      --pred-base custom_results/T1_v3 \\
      --models    mcs1 mcs6 mcs5 mcs3 \\
      --face-b    staging/T1_v3/face_B/CM_1005.png \\
      --matte     staging/T1_v3/matte/CM_1005.png \\
      --stem      CM_1005 \\
      --out-dir   custom_results/T1_v3/matte_fit \\
      --delta-e   25
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def scalp_color(face_b: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """face_B의 matte 영역 내 픽셀 평균 Lab — bald 두피 reference."""
    lab = rgb_to_lab(face_b)
    m   = matte > 127
    if m.sum() == 0:
        return np.array([50.0, 0.0, 0.0])
    return lab[m].mean(axis=0)


def hair_mask_unrestricted(pred: np.ndarray, scalp_lab: np.ndarray,
                           threshold: float) -> np.ndarray:
    """pred 전체에서 ΔE(pred, scalp) > θ 픽셀 = hair. (matte 제한 없음)

    한계: 얼굴 피부·배경·옷 등 두피색과 다른 모든 픽셀이 hair로 분류돼
    false-positive 다수 → mode='pixel_diff' 사용 추천.
    """
    lab  = rgb_to_lab(pred)
    diff = lab - scalp_lab[None, None, :]
    de   = np.sqrt((diff ** 2).sum(axis=-1))
    return de > threshold


def hair_mask_pixel_diff(pred: np.ndarray, face_b: np.ndarray,
                         threshold: float) -> np.ndarray:
    """pred와 face_B의 Lab ΔE > θ 픽셀 = "모델이 변경/추가한 영역".

    BLD가 matte 밖을 face_B로 유지한다는 가정 → pred ≠ face_B 인 픽셀 = 모델 painted.
    얼굴·배경 false-positive 없음 (그 영역은 pred ≈ face_B).
    """
    lab_p = rgb_to_lab(pred)
    lab_f = rgb_to_lab(face_b)
    diff  = lab_p - lab_f
    de    = np.sqrt((diff ** 2).sum(axis=-1))
    return de > threshold


def boundary_band(matte_bin: np.ndarray, dilate_px: int = 12,
                  erode_px: int = 12) -> np.ndarray:
    """matte 경계 band = dilation(matte) \\ erosion(matte)."""
    mb = matte_bin.astype(np.uint8) * 255
    k_d = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
    k_e = np.ones((erode_px  * 2 + 1, erode_px  * 2 + 1), np.uint8)
    d = cv2.dilate(mb, k_d)
    e = cv2.erode(mb, k_e)
    return (d > 0) & ~(e > 0)


def metrics(pred_hair: np.ndarray, matte_bin: np.ndarray) -> dict:
    inter = (pred_hair & matte_bin).sum()
    union = (pred_hair | matte_bin).sum()
    return {
        "fill_rate":  float(inter / matte_bin.sum())   if matte_bin.sum()  > 0 else 0.0,
        "spillage":   float((pred_hair & ~matte_bin).sum() / pred_hair.sum())
                      if pred_hair.sum() > 0 else 0.0,
        "region_iou": float(inter / union)             if union > 0 else 0.0,
    }


def boundary_iou(pred_hair: np.ndarray, matte_bin: np.ndarray,
                 band_px: int = 12) -> float:
    band  = boundary_band(matte_bin, band_px, band_px)
    p_b   = pred_hair & band
    m_b   = matte_bin & band
    inter = (p_b & m_b).sum()
    union = (p_b | m_b).sum()
    return float(inter / union) if union > 0 else 0.0


def overlay_contour(img: np.ndarray, matte_bin: np.ndarray,
                    color=(255, 0, 0), thickness: int = 2) -> np.ndarray:
    """img 위에 matte boundary contour를 색 선으로 overlay."""
    out = img.copy()
    contours, _ = cv2.findContours(matte_bin.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def make_figure(rows: list[dict], pred_base: Path, stem: str, matte_bin: np.ndarray,
                pred_hair_by_model: dict[str, np.ndarray], out_path: Path):
    import matplotlib.pyplot as plt
    models = [r["model"] for r in rows]
    n = len(models)
    cols = ["pred output", "+ matte contour", "pred_hair (ΔE>θ)", "GT matte"]
    fig, axes = plt.subplots(n, len(cols), figsize=(3.5 * len(cols), 3.5 * n))
    if n == 1:
        axes = axes[None, :]
    for i, m in enumerate(models):
        pred = load_rgb(pred_base / m / f"{stem}.png")
        if pred.shape[:2] != matte_bin.shape[:2]:
            pred = resize_to(pred, *matte_bin.shape[:2])
        overlay = overlay_contour(pred, matte_bin)
        hair    = pred_hair_by_model[m].astype(np.uint8) * 255
        gt      = matte_bin.astype(np.uint8) * 255
        axes[i, 0].imshow(pred);              axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])
        axes[i, 1].imshow(overlay);           axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])
        axes[i, 2].imshow(hair, cmap="gray"); axes[i, 2].set_xticks([]); axes[i, 2].set_yticks([])
        axes[i, 3].imshow(gt,   cmap="gray"); axes[i, 3].set_xticks([]); axes[i, 3].set_yticks([])
        axes[i, 0].set_ylabel(m, fontsize=11)
        if i == 0:
            for j, c in enumerate(cols):
                axes[i, j].set_title(c, fontsize=11)
    fig.suptitle(f"Matte fit analysis — {stem}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"figure: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-base", type=Path, required=True)
    p.add_argument("--models",    nargs="+", required=True,
                   help="pred-base 안의 model 서브폴더 이름들 (예: mcs1 mcs6 mcs5 mcs3)")
    p.add_argument("--face-b",    type=Path, required=True)
    p.add_argument("--matte",     type=Path, required=True)
    p.add_argument("--stem",      type=str, required=True)
    p.add_argument("--out-dir",   type=Path, required=True)
    p.add_argument("--delta-e",   type=float, default=25.0)
    p.add_argument("--band-px",   type=int,   default=12)
    p.add_argument("--mode",      choices=["scalp_de", "pixel_diff"],
                   default="pixel_diff",
                   help="hair mask 생성 방식. "
                        "scalp_de = pred vs face_B 두피 평균색 ΔE (false-positive 많음). "
                        "pixel_diff = pred vs face_B 픽셀단위 ΔE (BLD 가정, 권장).")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    face_b = load_rgb(args.face_b)
    matte  = load_l(args.matte)
    h, w   = matte.shape[:2]
    face_b = resize_to(face_b, h, w)
    matte_bin = matte > 127

    scalp = scalp_color(face_b, matte)

    rows = []
    pred_hair_by_model = {}
    for m in args.models:
        pred_p = args.pred_base / m / f"{args.stem}.png"
        if not pred_p.exists():
            print(f"  [SKIP] {m}: {pred_p} 없음")
            continue
        pred = resize_to(load_rgb(pred_p), h, w)
        if args.mode == "scalp_de":
            hair = hair_mask_unrestricted(pred, scalp, args.delta_e)
        else:
            hair = hair_mask_pixel_diff(pred, face_b, args.delta_e)
        pred_hair_by_model[m] = hair

        mt = metrics(hair, matte_bin)
        bi = boundary_iou(hair, matte_bin, args.band_px)
        rows.append({
            "model":         m,
            "fill_rate":     mt["fill_rate"],
            "spillage":      mt["spillage"],
            "region_iou":    mt["region_iou"],
            "boundary_iou":  bi,
        })

    # console table
    print(f"\nstem = {args.stem}   ΔE>{args.delta_e}   boundary band={args.band_px}px   mode={args.mode}")
    print(f"{'model':<6}{'fill↑':>12}{'spillage↓':>14}{'region IoU↑':>14}{'boundary IoU↑':>16}")
    print("-" * 64)
    for r in rows:
        print(f"{r['model']:<6}"
              f"{r['fill_rate']:>12.4f}"
              f"{r['spillage']:>14.4f}"
              f"{r['region_iou']:>14.4f}"
              f"{r['boundary_iou']:>16.4f}")

    # csv
    with open(args.out_dir / "matte_fit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "fill_rate", "spillage",
                                          "region_iou", "boundary_iou"])
        w.writeheader()
        w.writerows(rows)
    print(f"\ncsv: {args.out_dir / 'matte_fit.csv'}")

    make_figure(rows, args.pred_base, args.stem, matte_bin,
                pred_hair_by_model, args.out_dir / "figure.png")


if __name__ == "__main__":
    main()
