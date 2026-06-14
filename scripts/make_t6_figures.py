"""T6 figures.

per-stem figure: 7 rows × 4 cols
  row 0: target faces       — face_origin / face_warm / face_cool / face_dim
  row 1: Ours (mcs1)        — origin / warm / cool / dim 출력
  row 2: Ours+Gate (mcs2)
  row 3: Sketch-only (mcs3)
  row 4: Sketch-only+Gate (mcs4)
  row 5: Raw-only (mcs5)
  row 6: Matte-CNN-only (mcs6)

8 stems → 8 figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "custom_results/T6_v2/eval/figures"
OUT.mkdir(parents=True, exist_ok=True)

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
VARIANTS = ["origin", "warm", "cool", "dim"]
MODELS   = [("mcs1", "Ours"),
            ("mcs2", "Ours+Gate"),
            ("mcs3", "Sketch-only"),
            ("mcs4", "Sketch-only+Gate"),
            ("mcs5", "Raw-only"),
            ("mcs6", "Matte-CNN-only")]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def make_stem_figure(stem: str):
    nrows = 1 + len(MODELS)
    ncols = len(VARIANTS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))

    # row 0: target faces
    for j, v in enumerate(VARIANTS):
        img = load_rgb(ROOT / f"staging/T6_v2/face_{v}/{stem}.png")
        ax = axes[0, j]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"target {v}", fontsize=10)
    axes[0, 0].set_ylabel("target", fontsize=10)

    # rows 1..N: model outputs
    for i, (mcs, label) in enumerate(MODELS, start=1):
        for j, v in enumerate(VARIANTS):
            p = ROOT / f"custom_results/T6_v2/{mcs}_{v}/{stem}.png"
            img = load_rgb(p)
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
        axes[i, 0].set_ylabel(f"{mcs}\n{label}", fontsize=9)

    fig.suptitle(f"{stem} — T6 harmony tracking (PI 강도, mcs1~6 × origin/warm/cool/dim)",
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
