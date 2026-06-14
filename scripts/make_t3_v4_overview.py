"""T3_v4 overview figure (8 rows × 7 cols, mcs5 제외).

cols: sketch / matte / GT(face_A) / mcs1×A / mcs1×B / mcs3×A / mcs3×B
rows: 8 stems
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "custom_results/T3_v4/eval/figures/overview.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]

COLS = ["sketch", "matte", "target A", "target B",
        "mcs1 × A", "mcs1 × B", "mcs2 × A", "mcs2 × B",
        "mcs3 × A", "mcs3 × B", "mcs4 × A", "mcs4 × B",
        "mcs5 × A", "mcs5 × B", "mcs6 × A", "mcs6 × B"]


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
            load_rgb(ROOT / f"staging/T3_v4/sketch_recolor/{stem}.png"),
            np.stack([load_l(ROOT / f"staging/T3_v4/matte/{stem}.png")] * 3, -1),
            load_rgb(ROOT / f"staging/T3_v4/face_A/{stem}.png"),
            load_rgb(ROOT / f"staging/T3_v4/face_B/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs1_A/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs1_B/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs2_A/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs2_B/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs3_A/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs3_B/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs4_A/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs4_B/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs5_A/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs5_B/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs6_A/{stem}.png"),
            load_rgb(ROOT / f"custom_results/T3_v4/mcs6_B/{stem}.png"),
        ]
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(COLS[j], fontsize=10)
        axes[i, 0].set_ylabel(stem, fontsize=10)
    fig.suptitle("T3_v4 — GT-recolor sketch, 8 stems, mcs1 vs mcs3 (A/B)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {OUT}")


if __name__ == "__main__":
    main()
