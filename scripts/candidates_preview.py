"""상위 unbraid 후보 시각 비교용 figure.

각 stem: [원본 img_ori / bald img / matte 알파 오버레이]
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "custom_results/T3_v2/eval/figures/candidates3.png"


STEMS = [
    # CM_ 다음 batch (이미 본 7개 제외, score 순)
    "CM_1038", "CM_1052", "CM_1045", "CM_1135", "CM_1210",
    "CM_1084", "CM_1152", "CM_1153", "CM_1186", "CM_1011",
]


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def overlay_matte(img: np.ndarray, matte: np.ndarray, alpha: float = 0.45,
                  color=(255, 0, 0)) -> np.ndarray:
    """img 위에 matte 영역을 반투명 색으로 덮음."""
    out = img.astype(np.float32).copy()
    m   = (matte > 127).astype(np.float32)[..., None]
    color_arr = np.array(color, dtype=np.float32)[None, None, :]
    out = out * (1 - alpha * m) + color_arr * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    n = len(STEMS)
    cols = ["original (face_A)", "bald (face_B)", "original + matte overlay"]
    fig, axes = plt.subplots(n, 3, figsize=(3.0 * 3, 3.0 * n))
    for i, stem in enumerate(STEMS):
        orig  = load_rgb(ROOT / f"data/unbraid/img_ori/test/{stem}.png")
        bald  = load_rgb(ROOT / f"data/unbraid/img/{stem}.png")
        matte = load_l(ROOT / f"data/unbraid/matte/test/{stem}.png")
        # resize matte to img shape if different
        if matte.shape != orig.shape[:2]:
            import cv2
            matte = cv2.resize(matte, (orig.shape[1], orig.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        if bald.shape != orig.shape:
            import cv2
            bald = cv2.resize(bald, (orig.shape[1], orig.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        ov = overlay_matte(orig, matte)
        for j, img in enumerate([orig, bald, ov]):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(cols[j], fontsize=11)
        axes[i, 0].set_ylabel(stem, fontsize=10)
    fig.suptitle("Unbraid top-6 candidates — matte coverage 검수", fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print(f"figure: {OUT}")


if __name__ == "__main__":
    main()
