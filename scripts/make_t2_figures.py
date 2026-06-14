"""T2 figures: per-stem 7×3 grid.
  row 0: sketch (dense / mid / sparse)
  row 1: target B (동일 image, 3 col 채움)
  rows 2..7: 6 model outputs × 3 sparsity
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "custom_results/T2_v1/eval/figures"
OUT.mkdir(parents=True, exist_ok=True)

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
SPARSITY = ["dense", "mid", "sparse"]
MODELS   = [("mcs1", "Ours"),
            ("mcs2", "Ours+Gate"),
            ("mcs3", "Sketch-only"),
            ("mcs4", "Sketch-only+Gate"),
            ("mcs5", "Raw-only"),
            ("mcs6", "Matte-CNN-only")]


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def make_stem_figure(stem):
    nrows = 1 + 1 + len(MODELS)
    ncols = len(SPARSITY)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))

    # row 0: sketch
    for j, sk in enumerate(SPARSITY):
        img = load_rgb(ROOT / f"staging/T2_v1/sketch_{sk}/{stem}.png")
        ax = axes[0, j]
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"sketch {sk}", fontsize=10)
    axes[0, 0].set_ylabel("sketch", fontsize=10)

    # row 1: target B
    face_b = load_rgb(ROOT / f"staging/T2_v1/face_B/{stem}.png")
    for j in range(ncols):
        ax = axes[1, j]
        ax.imshow(face_b); ax.set_xticks([]); ax.set_yticks([])
    axes[1, 0].set_ylabel("target B", fontsize=10)

    # rows 2..: model outputs
    for i, (mcs, label) in enumerate(MODELS, start=2):
        for j, sk in enumerate(SPARSITY):
            p = ROOT / f"custom_results/T2_v1/{mcs}_{sk}/{stem}.png"
            img = load_rgb(p)
            ax = axes[i, j]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(f"{mcs}\n{label}", fontsize=9)

    fig.suptitle(f"{stem} — T2 sparsity ladder (dense/mid/sparse) × mcs1~6",
                 fontsize=12)
    fig.tight_layout()
    out_p = OUT / f"{stem}.png"
    fig.savefig(out_p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_p}")


def main():
    for s in STEMS:
        make_stem_figure(s)


if __name__ == "__main__":
    main()
