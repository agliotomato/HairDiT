"""T5 — stroke-overlay figure (design.md L107 정성).

per-stem figure: 1 row × 8 cells
  cells: sketch / target B / mcs1+overlay / mcs2+overlay / mcs3+overlay / mcs4+overlay / mcs5+overlay / mcs6+overlay

overlay: sketch 반투명(alpha=0.4) over model output.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "custom_results/T5_v1/eval/figures"
OUT.mkdir(parents=True, exist_ok=True)

STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
MODELS = [("mcs1", "Ours"), ("mcs2", "Ours+Gate"),
          ("mcs3", "Sketch-only"), ("mcs4", "Sketch-only+Gate"),
          ("mcs5", "Raw-only"), ("mcs6", "Matte-CNN-only")]


def load_rgb(p): return np.array(Image.open(p).convert("RGB"))


def overlay(pred, sketch, alpha=0.4):
    """sketch 비-검정 영역만 alpha 합성."""
    out = pred.astype(np.float32).copy()
    mask = (sketch.astype(np.float32).sum(axis=-1) > 10)[..., None].astype(np.float32)
    out = out * (1 - alpha * mask) + sketch.astype(np.float32) * (alpha * mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_stem_figure(stem):
    sketch = load_rgb(ROOT / f"staging/T5_v1/sketch_sparse/{stem}.png")
    face_b = load_rgb(ROOT / f"staging/T5_v1/face_B/{stem}.png")

    cells = [("sketch", sketch), ("target B", face_b)]
    for mcs, label in MODELS:
        pred = load_rgb(ROOT / f"custom_results/T5_v1/{mcs}/{stem}.png")
        if pred.shape != sketch.shape:
            import cv2
            pred = cv2.resize(pred, (sketch.shape[1], sketch.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        ov = overlay(pred, sketch, alpha=0.5)
        cells.append((f"{mcs}\n{label}", ov))

    n = len(cells)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.6 + 0.5))
    for ax, (lbl, img) in zip(axes, cells):
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(lbl, fontsize=9)
    fig.suptitle(f"{stem} — T5 stroke-overlay (sparse raw + alpha 0.5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {OUT / f'{stem}.png'}")


def main():
    for s in STEMS:
        make_stem_figure(s)


if __name__ == "__main__":
    main()
