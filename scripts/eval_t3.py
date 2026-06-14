"""T3 (Leakage) 단일 변수 평가: Sketch-only(mcs3) vs Ours(mcs1) × Target {A,B}.

측정:
  - Edge IoU (vs GT sketch, hair 영역) — 구조 보존
  - hair-masked LPIPS (vs GT 원본, hair 영역) — 외형 충실 (A 조건에서만 의미)
  - A → B drop = |M(A) − M(B)|   (방향성 포함)

출력:
  - 콘솔 표: per-image 4 conditions × 2 metrics
  - drop summary: Sketch-only drop vs Ours drop
  - figure: 4 conditions × N images grid (input column + 4 outputs)

Usage:
  python scripts/eval_t3.py \\
      --pred-base custom_results/T3 \\
      --sketch    staging/T3/sketch \\
      --matte     staging/T3/matte \\
      --face-a    staging/T3/face_A \\
      --out-dir   custom_results/T3/eval
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.eval_metrics import (
    canny_edges,
    edge_iou,
    hair_masked_lpips,
)

CONDS = [("mcs1_A", "Ours×A"), ("mcs1_B", "Ours×B"),
         ("mcs3_A", "Sketch-only×A"), ("mcs3_B", "Sketch-only×B")]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    import cv2
    if arr.shape[:2] == (h, w):
        return arr
    return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)


def compute_per_image(pred_base: Path, sketch_dir: Path, matte_dir: Path,
                      face_a_dir: Path, stems: list[str]) -> list[dict]:
    """각 (stem, condition) 조합의 metric을 dict 리스트로 반환."""
    rows = []
    for stem in stems:
        sk    = load_rgb(sketch_dir / f"{stem}.png")
        matte = load_l(matte_dir / f"{stem}.png")
        gt_a  = load_rgb(face_a_dir / f"{stem}.png")

        h, w  = sk.shape[:2]
        matte = resize_to(matte, h, w)
        gt_a  = resize_to(gt_a, h, w)
        sk_e  = canny_edges(sk)

        for cond_key, cond_label in CONDS:
            pred_p = pred_base / cond_key / f"{stem}.png"
            if not pred_p.exists():
                rows.append({"stem": stem, "cond": cond_label, "edge_iou": None, "lpips_gt": None})
                continue
            pred = resize_to(load_rgb(pred_p), h, w)
            pred_e = canny_edges(pred)
            ei = edge_iou(pred_e, sk_e, matte)
            lp = hair_masked_lpips(pred, gt_a, matte)  # A를 GT로 — B 조건 LPIPS는 reference frame 동일
            rows.append({"stem": stem, "cond": cond_label, "edge_iou": ei, "lpips_gt": lp})
    return rows


def summarize_drop(rows: list[dict]) -> list[dict]:
    """모델별 A→B drop 폭 (per-image 평균)."""
    out = []
    by_stem = {}
    for r in rows:
        by_stem.setdefault(r["stem"], {})[r["cond"]] = r
    for model_label, a_key, b_key in [
        ("Ours (mcs1)",        "Ours×A",        "Ours×B"),
        ("Sketch-only (mcs3)", "Sketch-only×A", "Sketch-only×B"),
    ]:
        e_drops, l_drops = [], []
        for stem, d in by_stem.items():
            ea, eb = d[a_key]["edge_iou"], d[b_key]["edge_iou"]
            la, lb = d[a_key]["lpips_gt"], d[b_key]["lpips_gt"]
            if ea is not None and eb is not None:
                e_drops.append(ea - eb)   # 부호: + 면 A에서 더 높음(= B에서 drop)
            if la is not None and lb is not None:
                l_drops.append(lb - la)   # 부호: + 면 B에서 더 멀어짐(= drop)
        out.append({
            "model":           model_label,
            "edge_iou_drop":   sum(e_drops) / len(e_drops) if e_drops else None,
            "lpips_drop":      sum(l_drops) / len(l_drops) if l_drops else None,
            "n":               len(e_drops),
        })
    return out


def make_figure(pred_base: Path, sketch_dir: Path, matte_dir: Path,
                face_a_dir: Path, face_b_dir: Path, stems: list[str], out_path: Path):
    """Figure: N rows × 7 cols.
        cols: [sketch, matte, target A, target B, Ours×A, Ours×B, Sketch-only×A, Sketch-only×B]
        (8 cols total)
    """
    import matplotlib.pyplot as plt
    n = len(stems)
    cols = ["sketch", "matte", "target A", "target B",
            "Ours×A", "Ours×B", "Sketch×A", "Sketch×B"]
    fig, axes = plt.subplots(n, len(cols), figsize=(3.2 * len(cols), 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    for i, stem in enumerate(stems):
        imgs = [
            load_rgb(sketch_dir / f"{stem}.png"),
            np.stack([load_l(matte_dir / f"{stem}.png")] * 3, -1),
            load_rgb(face_a_dir / f"{stem}.png"),
            load_rgb(face_b_dir / f"{stem}.png"),
            load_rgb(pred_base / "mcs1_A" / f"{stem}.png"),
            load_rgb(pred_base / "mcs1_B" / f"{stem}.png"),
            load_rgb(pred_base / "mcs3_A" / f"{stem}.png"),
            load_rgb(pred_base / "mcs3_B" / f"{stem}.png"),
        ]
        for j, (img, c) in enumerate(zip(imgs, cols)):
            axes[i, j].imshow(img)
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(c, fontsize=10)
        axes[i, 0].set_ylabel(stem, fontsize=9)
    fig.suptitle("T3 — Leakage (A vs B) × {Ours, Sketch-only}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"figure: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-base", required=True, type=Path)
    p.add_argument("--sketch",    required=True, type=Path)
    p.add_argument("--matte",     required=True, type=Path)
    p.add_argument("--face-a",    required=True, type=Path)
    p.add_argument("--face-b",    required=True, type=Path)
    p.add_argument("--out-dir",   required=True, type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted([p.stem for p in args.sketch.glob("*.png")])
    print(f"stems: {stems}")

    rows = compute_per_image(args.pred_base, args.sketch, args.matte, args.face_a, stems)
    drops = summarize_drop(rows)

    # console table
    print("\n=== per-image metrics ===")
    print(f"{'stem':<14}{'cond':<18}{'Edge IoU↑':>12}{'LPIPS↓':>10}")
    print("-" * 56)
    for r in rows:
        ei = f"{r['edge_iou']:.4f}" if r['edge_iou'] is not None else "N/A"
        lp = f"{r['lpips_gt']:.4f}" if r['lpips_gt'] is not None else "N/A"
        print(f"{r['stem']:<14}{r['cond']:<18}{ei:>12}{lp:>10}")

    print("\n=== A → B drop (positive = drop from A to B) ===")
    print(f"{'model':<22}{'Edge IoU drop':>16}{'LPIPS drop':>14}{'n':>4}")
    print("-" * 56)
    for d in drops:
        ei = f"{d['edge_iou_drop']:+.4f}" if d['edge_iou_drop'] is not None else "N/A"
        lp = f"{d['lpips_drop']:+.4f}"    if d['lpips_drop']    is not None else "N/A"
        print(f"{d['model']:<22}{ei:>16}{lp:>14}{d['n']:>4}")

    # save csv
    import csv
    with open(args.out_dir / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "cond", "edge_iou", "lpips_gt"])
        w.writeheader()
        w.writerows(rows)
    with open(args.out_dir / "drop_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "edge_iou_drop", "lpips_drop", "n"])
        w.writeheader()
        w.writerows(drops)
    print(f"\ncsv: {args.out_dir}/per_image.csv, drop_summary.csv")

    make_figure(args.pred_base, args.sketch, args.matte, args.face_a, args.face_b,
                stems, args.out_dir / "figure.png")


if __name__ == "__main__":
    main()
