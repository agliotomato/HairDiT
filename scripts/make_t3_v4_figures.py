"""T3_v4 figures — 각 stem 별 1×8 cells:
  sketch / matte / GT(face_A) / mcs1×A / mcs1×B / mcs3×A / mcs3×B / mcs5×A / mcs5×B

8 stems × 9 cells. 또는 stem 별 단일 figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "custom_results/T3_v4/eval/figures"
OUT.mkdir(parents=True, exist_ok=True)

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def make_row(stem: str):
    sketch = load_rgb(ROOT / f"staging/T3_v4/sketch_recolor/{stem}.png")
    matte  = load_l(ROOT / f"staging/T3_v4/matte/{stem}.png")
    gt_a   = load_rgb(ROOT / f"staging/T3_v4/face_A/{stem}.png")
    cells = [
        ("sketch (GT-recolor)", sketch),
        ("matte",       np.stack([matte] * 3, -1)),
        ("GT (face_A)", gt_a),
        ("mcs1 × A",    load_rgb(ROOT / f"custom_results/T3_v4/mcs1_A/{stem}.png")),
        ("mcs1 × B",    load_rgb(ROOT / f"custom_results/T3_v4/mcs1_B/{stem}.png")),
        ("mcs3 × A",    load_rgb(ROOT / f"custom_results/T3_v4/mcs3_A/{stem}.png")),
        ("mcs3 × B",    load_rgb(ROOT / f"custom_results/T3_v4/mcs3_B/{stem}.png")),
        ("mcs5 × A",    load_rgb(ROOT / f"custom_results/T3_v4/mcs5_A/{stem}.png")),
        ("mcs5 × B",    load_rgb(ROOT / f"custom_results/T3_v4/mcs5_B/{stem}.png")),
    ]
    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6 + 0.4))
    for ax, (lbl, img) in zip(axes, cells):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(lbl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{stem} — GT-recolor (T3_v4)", fontsize=11)
    fig.tight_layout()
    out_p = OUT / f"{stem}.png"
    fig.savefig(out_p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_p}")


def main():
    for s in STEMS:
        make_row(s)


if __name__ == "__main__":
    main()
