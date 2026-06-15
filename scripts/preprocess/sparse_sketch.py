"""Sparse sketch generator (color-group keep).

각 stroke를 색 단위로 그룹핑하고, random하게 keep_ratio 만큼만 유지.
drop된 색의 픽셀은 검정으로 변환.

Algorithm:
  1. RGB → 5-bit quantize (32 bins per channel) — 같은 stroke의 미세 색 변화 grouping
  2. unique 색 추출 (검정 제외)
  3. shuffle(seed=fixed) → 앞 keep_ratio 만큼 keep, 나머지 drop
  4. drop된 색의 모든 픽셀 → 0 (검정)

Usage:
  python scripts/sparse_sketch.py \\
      --src   data/unbraid/sketch/test    \\
      --stems CM_1005 CM_1007 braid_2562  \\
      --src-braid data/braid/sketch/test  \\
      --keep 0.5 --seed 42                 \\
      --out   staging/T1/sketch
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def sparsify_sketch(
    sketch: np.ndarray,
    keep_ratio: float,
    seed: int,
    quantize_bits: int = 5,
) -> tuple[np.ndarray, int, int]:
    """Sparse sketch by color-group keep.

    Returns (sparse_sketch_uint8, n_total_colors, n_kept_colors).
    """
    shift = 8 - quantize_bits
    q = (sketch >> shift) << shift  # (H, W, 3) quantized

    flat = q.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    colors = [tuple(int(v) for v in c) for c in uniq
              if not (c[0] == 0 and c[1] == 0 and c[2] == 0)]

    n_total = len(colors)
    if n_total == 0:
        return sketch.copy(), 0, 0

    n_keep = max(1, int(round(n_total * keep_ratio)))
    rng = random.Random(seed)
    shuffled = list(colors)
    rng.shuffle(shuffled)
    drop = set(shuffled[n_keep:])

    out = sketch.copy()
    if drop:
        # vectorized drop: build mask for any (r,g,b) in drop
        # (R,G,B) integer pack
        packed = (q[:, :, 0].astype(np.int32) << 16) | (q[:, :, 1].astype(np.int32) << 8) | q[:, :, 2].astype(np.int32)
        drop_packed = set((r << 16) | (g << 8) | b for r, g, b in drop)
        drop_mask = np.isin(packed, list(drop_packed))
        out[drop_mask] = 0
    return out, n_total, n_keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src",       type=Path, required=True,
                   help="dense sketch 디렉토리 (예: data/unbraid/sketch/test)")
    p.add_argument("--src-braid", type=Path, default=None,
                   help="braid sketch 디렉토리 (stems 중 'braid_' prefix용)")
    p.add_argument("--stems",     nargs="+", required=True)
    p.add_argument("--keep",      type=float, default=0.5)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--out",       type=Path,  required=True)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for stem in args.stems:
        src_dir = args.src_braid if stem.startswith("braid_") and args.src_braid else args.src
        src_p = src_dir / f"{stem}.png"
        if not src_p.exists():
            print(f"  [SKIP] {src_p} 없음")
            continue
        sk = np.array(Image.open(src_p).convert("RGB"))
        sparse, n_total, n_kept = sparsify_sketch(sk, args.keep, args.seed)
        out_p = args.out / f"{stem}.png"
        Image.fromarray(sparse).save(out_p)
        print(f"  {stem}: {n_total} colors → kept {n_kept} ({n_kept/max(1,n_total):.0%}) → {out_p}")


if __name__ == "__main__":
    main()
