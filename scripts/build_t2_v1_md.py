"""[0614]T2_v1.md per-stem 섹션 자동 생성.

T1_v2.md 스타일 — per-stem 이미지 grid + 측정 표.
"""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
SPARSITY = ["dense", "mid", "sparse"]
MODELS = [("mcs1", "Ours"),
          ("mcs2", "Ours+Gate"),
          ("mcs3", "Sketch-only"),
          ("mcs4", "Sketch-only+Gate"),
          ("mcs5", "Raw-only"),
          ("mcs6", "Matte-CNN-only")]

# 측정값 로딩
data = defaultdict(dict)
with open(ROOT / "custom_results/T2_v1/eval/per_image.csv") as f:
    for r in csv.DictReader(f):
        data[(r["stem"], r["sparsity"], r["model"])] = (
            float(r["edge_iou"]), float(r["chroma"]))


def render_stem(stem):
    out = []
    out.append(f"#### {stem}\n")

    # 이미지 grid (6 rows × 3 cols, 모델 × sparsity)
    out.append("| 모델 | dense | mid | sparse |")
    out.append("|---|:---:|:---:|:---:|")
    out.append(f"| sketch | <img src=\"../outputs/T2_v1/staging/sketch_dense/{stem}.png\" width=\"140\"> "
               f"| <img src=\"../outputs/T2_v1/staging/sketch_mid/{stem}.png\" width=\"140\"> "
               f"| <img src=\"../outputs/T2_v1/staging/sketch_sparse/{stem}.png\" width=\"140\"> |")
    for mcs, label in MODELS:
        cells = [f"<img src=\"../outputs/T2_v1/{mcs}_{sk}/{stem}.png\" width=\"140\">"
                 for sk in SPARSITY]
        out.append(f"| **{mcs} ({label})** | " + " | ".join(cells) + " |")
    out.append("")

    # 측정 표
    out.append("| 모델 | dense (IoU / Chroma) | mid (IoU / Chroma) | sparse (IoU / Chroma) |")
    out.append("|------|:---:|:---:|:---:|")
    for mcs, label in MODELS:
        cells = []
        for sk in SPARSITY:
            ei, C = data[(stem, sk, mcs)]
            cells.append(f"{ei:.4f} / {C:.2f}")
        out.append(f"| {mcs} ({label}) | " + " | ".join(cells) + " |")
    out.append("")
    return "\n".join(out)


def main():
    for s in STEMS:
        print(render_stem(s))
        print()


if __name__ == "__main__":
    main()
