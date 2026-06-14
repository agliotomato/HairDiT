"""T1 (Region fill) 평가: 4 models × matte 영역 채움 능력.

측정 — region IoU (휴리스틱):
  pred에서 matte 영역 내 픽셀 중 face_B(bald 두피)의 두피색과
  CIE Lab ΔE > θ 인 픽셀 = "hair-painted" pixel로 분류.
  → IoU(hair_painted_mask, gt_matte_binary)

  두피색 reference = face_B의 matte 영역 내 픽셀 평균 Lab.
                     (그 영역이 두피로 inpaint된 픽셀이라는 가정)

출력:
  - per-image, per-model region IoU 표
  - matte fill ratio (matte 영역 중 hair-painted 비율, recall-like)
  - figure: 4 models × N stems grid (sparse sketch + matte + face_B + 4 outputs)

Usage:
  python scripts/eval_t1.py \\
      --pred-base custom_results/T1 \\
      --sketch    staging/T1/sketch \\
      --matte     staging/T1/matte \\
      --face-b    staging/T1/face_B \\
      --out-dir   custom_results/T1/eval \\
      --delta-e   25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODELS = [("mcs1", "Ours"),
          ("mcs6", "Matte-CNN-only"),
          ("mcs5", "Raw-only"),
          ("mcs3", "Sketch-only")]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """uint8 RGB → float64 Lab (L∈[0,100], a,b∈[-128,127])."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def scalp_color(face_b: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """face_B의 matte 영역 내 픽셀 평균 Lab — '두피 reference color'."""
    lab = rgb_to_lab(face_b)
    m = matte > 127
    if m.sum() == 0:
        return np.array([50.0, 0.0, 0.0])
    return lab[m].mean(axis=0)


def hair_mask_by_delta_e(pred: np.ndarray, scalp_lab: np.ndarray,
                         matte: np.ndarray, threshold: float) -> np.ndarray:
    """pred에서 matte 영역 ∩ ΔE(pred, scalp) > θ → hair_painted pixels."""
    lab = rgb_to_lab(pred)
    diff = lab - scalp_lab[None, None, :]
    de = np.sqrt((diff ** 2).sum(axis=-1))
    m = matte > 127
    return (de > threshold) & m


def region_iou(pred_hair: np.ndarray, gt_matte_bin: np.ndarray) -> float:
    inter = (pred_hair & gt_matte_bin).sum()
    union = (pred_hair | gt_matte_bin).sum()
    return float(inter / union) if union > 0 else 0.0


def fill_ratio(pred_hair: np.ndarray, gt_matte_bin: np.ndarray) -> float:
    """matte 영역 중 hair_painted 픽셀 비율 (recall-like)."""
    denom = gt_matte_bin.sum()
    return float((pred_hair & gt_matte_bin).sum() / denom) if denom > 0 else 0.0


def compute_metrics(pred_base: Path, sketch_dir: Path, matte_dir: Path,
                    face_b_dir: Path, stems: list[str],
                    delta_e: float) -> list[dict]:
    rows = []
    for stem in stems:
        matte  = load_l(matte_dir / f"{stem}.png")
        face_b = load_rgb(face_b_dir / f"{stem}.png")

        h, w   = matte.shape[:2]
        face_b = resize_to(face_b, h, w)
        gt_bin = matte > 127

        scalp = scalp_color(face_b, matte)

        for mkey, mlabel in MODELS:
            p = pred_base / mkey / f"{stem}.png"
            if not p.exists():
                rows.append({"stem": stem, "model": mlabel,
                             "region_iou": None, "fill_ratio": None})
                continue
            pred = resize_to(load_rgb(p), h, w)
            hair = hair_mask_by_delta_e(pred, scalp, matte, delta_e)
            rows.append({
                "stem":        stem,
                "model":       mlabel,
                "region_iou":  region_iou(hair, gt_bin),
                "fill_ratio":  fill_ratio(hair, gt_bin),
            })
    return rows


def summarize_by_model(rows: list[dict]) -> list[dict]:
    out = []
    for _, mlabel in MODELS:
        vals_iou = [r["region_iou"] for r in rows
                    if r["model"] == mlabel and r["region_iou"] is not None]
        vals_fr  = [r["fill_ratio"] for r in rows
                    if r["model"] == mlabel and r["fill_ratio"] is not None]
        out.append({
            "model":          mlabel,
            "region_iou":     sum(vals_iou) / len(vals_iou) if vals_iou else None,
            "fill_ratio":     sum(vals_fr) / len(vals_fr) if vals_fr else None,
            "n":              len(vals_iou),
        })
    return out


def make_figure(pred_base: Path, sketch_dir: Path, matte_dir: Path,
                face_b_dir: Path, stems: list[str], out_path: Path,
                delta_e: float):
    """N rows × 7 cols: [sparse sketch, matte, face_B, mcs1, mcs6, mcs5, mcs3]."""
    import matplotlib.pyplot as plt
    n = len(stems)
    cols = ["sparse sketch", "matte", "target B (bald)",
            "Ours (mcs1)", "Matte-CNN-only (mcs6)",
            "Raw-only (mcs5)", "Sketch-only (mcs3)"]
    fig, axes = plt.subplots(n, len(cols), figsize=(3.0 * len(cols), 3.0 * n))
    if n == 1:
        axes = axes[None, :]
    for i, stem in enumerate(stems):
        matte  = load_l(matte_dir / f"{stem}.png")
        face_b = load_rgb(face_b_dir / f"{stem}.png")
        imgs = [
            load_rgb(sketch_dir / f"{stem}.png"),
            np.stack([matte] * 3, -1),
            face_b,
            load_rgb(pred_base / "mcs1" / f"{stem}.png"),
            load_rgb(pred_base / "mcs6" / f"{stem}.png"),
            load_rgb(pred_base / "mcs5" / f"{stem}.png"),
            load_rgb(pred_base / "mcs3" / f"{stem}.png"),
        ]
        for j, (img, c) in enumerate(zip(imgs, cols)):
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(c, fontsize=10)
        axes[i, 0].set_ylabel(stem, fontsize=9)
    fig.suptitle(f"T1 — Region fill (target B, sparse sketch, ΔE>{delta_e})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"figure: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-base", type=Path, required=True)
    p.add_argument("--sketch",    type=Path, required=True)
    p.add_argument("--matte",     type=Path, required=True)
    p.add_argument("--face-b",    type=Path, required=True)
    p.add_argument("--out-dir",   type=Path, required=True)
    p.add_argument("--delta-e",   type=float, default=25.0,
                   help="hair vs scalp 임계 ΔE (기본 25)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stems = sorted([p.stem for p in args.sketch.glob("*.png")])
    print(f"stems: {stems}")
    print(f"ΔE threshold: {args.delta_e}")

    rows = compute_metrics(args.pred_base, args.sketch, args.matte,
                           args.face_b, stems, args.delta_e)
    summary = summarize_by_model(rows)

    print("\n=== per-image region IoU ===")
    print(f"{'stem':<14}{'model':<22}{'region IoU↑':>14}{'fill ratio↑':>14}")
    print("-" * 64)
    for r in rows:
        ri = f"{r['region_iou']:.4f}" if r['region_iou'] is not None else "N/A"
        fr = f"{r['fill_ratio']:.4f}" if r['fill_ratio'] is not None else "N/A"
        print(f"{r['stem']:<14}{r['model']:<22}{ri:>14}{fr:>14}")

    print("\n=== model-averaged (n images) ===")
    print(f"{'model':<22}{'region IoU↑':>14}{'fill ratio↑':>14}{'n':>4}")
    print("-" * 56)
    for d in summary:
        ri = f"{d['region_iou']:.4f}" if d['region_iou'] is not None else "N/A"
        fr = f"{d['fill_ratio']:.4f}" if d['fill_ratio'] is not None else "N/A"
        print(f"{d['model']:<22}{ri:>14}{fr:>14}{d['n']:>4}")

    import csv
    with open(args.out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "model", "region_iou", "fill_ratio"])
        w.writeheader()
        w.writerows(rows)
    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "region_iou", "fill_ratio", "n"])
        w.writeheader()
        w.writerows(summary)
    print(f"\ncsv: {args.out_dir}/per_image.csv, summary.csv")

    make_figure(args.pred_base, args.sketch, args.matte, args.face_b,
                stems, args.out_dir / "figure.png", args.delta_e)


if __name__ == "__main__":
    main()
