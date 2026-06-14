"""T1_v4 overview figure — 8 rows × 7 cols.

cols: sparse sketch(GT-recolor) / matte / target B / Ours(mcs1) / Matte-CNN-only(mcs6) / Raw-only(mcs5) / Sketch-only(mcs3)
rows: 8 stems
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "custom_results/T1_v4/eval/figure.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]

COLS = ["sparse sketch", "matte", "target B",
        "Ours (mcs1)", "Ours+Gate (mcs2)",
        "Matte-CNN-only (mcs6)", "Raw-only (mcs5)",
        "Sketch-only (mcs3)", "Sketch-only+Gate (mcs4)"]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def main():
    n_rows, n_cols = len(STEMS), len(COLS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 2.4 * n_rows))
    for i, stem in enumerate(STEMS):
        imgs = [
            load_rgb(ROOT / f"staging/T1_v4/sketch_recolor/{stem}.png"),
            np.stack([load_l(ROOT / f"staging/T1_v4/matte/{stem}.png")] * 3, -1),
            load_rgb(ROOT / f"staging/T1_v4/face_B/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T1_v4/mcs1/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T1_v4/mcs2/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T1_v4/mcs6/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T1_v4/mcs5/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T1_v4/mcs3/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T1_v4/mcs4/{stem}.png"),
        ]
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(COLS[j], fontsize=10)
        axes[i, 0].set_ylabel(stem, fontsize=10)
    fig.suptitle("T1_v4 — sparse GT-recolor sketch, 8 stems, 4 구성 비교",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {OUT}")


if __name__ == "__main__":
    main()
