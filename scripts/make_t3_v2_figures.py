"""[0613]T3_v2.md 보고용 figure 생성.

CM_1005:  sketch, matte, GT, mcs1×A/B, mcs3×A/B  (1 row × 7 cells)
CM_1101:  sketch, matte, GT, mcs1×A/B, mcs2×A/B, mcs3×A/B  (1 row × 9 cells)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "custom_results/T3_v2/eval/figures"
OUT.mkdir(parents=True, exist_ok=True)


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def make_row_figure(cells: list[tuple[str, np.ndarray]], out_path: Path,
                    title: str, cell_size: float = 2.4):
    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(cell_size * n, cell_size + 0.4))
    if n == 1:
        axes = [axes]
    for ax, (lbl, img) in zip(axes, cells):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(lbl, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    # CM_1005 — GT-recolor sketch (T3_v3) + mcs5×B
    sketch_1005 = load_rgb(ROOT / "staging/T3_v3/sketch_recolor/CM_1005.png")
    matte_1005  = load_l(ROOT / "staging/T3_v3/matte/CM_1005.png")
    gt_1005     = load_rgb(ROOT / "staging/T3_v3/face_A/CM_1005.png")
    face_b_1005 = load_rgb(ROOT / "staging/T3_v3/face_B/CM_1005.png")
    cells_1005 = [
        ("sketch (GT-recolor)", sketch_1005),
        ("matte",       np.stack([matte_1005] * 3, -1)),
        ("target A",    gt_1005),
        ("target B",    face_b_1005),
        ("mcs1 × A",    load_rgb(ROOT / "custom_results/T3_v3/mcs1_A/CM_1005.png")),
        ("mcs1 × B",    load_rgb(ROOT / "custom_results/T3_v3/mcs1_B/CM_1005.png")),
        ("mcs3 × A",    load_rgb(ROOT / "custom_results/T3_v3/mcs3_A/CM_1005.png")),
        ("mcs3 × B",    load_rgb(ROOT / "custom_results/T3_v3/mcs3_B/CM_1005.png")),
        ("mcs5 × A",    load_rgb(ROOT / "custom_results/T3_v3/mcs5_A/CM_1005.png")),
        ("mcs5 × B",    load_rgb(ROOT / "custom_results/T3_v3/mcs5_B/CM_1005.png")),
    ]
    make_row_figure(cells_1005, OUT / "CM_1005.png",
                    "CM_1005 — GT-recolor sketch (T3_v3) + mcs5×B")

    # CM_1101
    sketch_1101 = load_rgb(ROOT / "data/unbraid/sketch/test/CM_1101.png")
    matte_1101  = load_l(ROOT / "data/unbraid/matte/test/CM_1101.png")
    gt_1101     = load_rgb(ROOT / "data/unbraid/img_ori/test/CM_1101.png")
    face_b_1101 = load_rgb(ROOT / "data/unbraid/img/CM_1101.png")
    cells_1101 = [
        ("sketch",      sketch_1101),
        ("matte",       np.stack([matte_1101] * 3, -1)),
        ("target A",    gt_1101),
        ("target B",    face_b_1101),
        ("mcs1 × A",    load_rgb(ROOT / "custom_results/T3_v2/mcs1_A/CM_1101.png")),
        ("mcs1 × B",    load_rgb(ROOT / "custom_results/T3_v2/mcs1_B/CM_1101.png")),
        ("mcs2 × A",    load_rgb(ROOT / "custom_results/T3_v3/mcs2_A/CM_1101.png")),
        ("mcs2 × B",    load_rgb(ROOT / "custom_results/T3_v3/mcs2_B/CM_1101.png")),
        ("mcs3 × A",    load_rgb(ROOT / "custom_results/T3_v2/mcs3_A/CM_1101.png")),
        ("mcs3 × B",    load_rgb(ROOT / "custom_results/T3_v2/mcs3_B/CM_1101.png")),
    ]
    make_row_figure(cells_1101, OUT / "CM_1101.png",
                    "CM_1101 — colorful sketch (T3_v2 + mcs2 from T3_v3)")


if __name__ == "__main__":
    main()
