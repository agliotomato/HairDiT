"""T6_v? per-stem 섹션 자동 생성 (--version 1 또는 2)."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
STEMS = ["CM_1005", "CM_1033", "CM_1067", "CM_1068",
         "CM_1077", "CM_1082", "CM_1101", "CM_1106"]
VARIANTS = ["origin", "warm", "cool", "dim"]
MODELS = [("mcs1", "Ours"),
          ("mcs2", "Ours+Gate"),
          ("mcs3", "Sketch-only"),
          ("mcs4", "Sketch-only+Gate"),
          ("mcs5", "Raw-only"),
          ("mcs6", "Matte-CNN-only")]


def render(version: str):
    folder = f"T6_v{version}"
    data = defaultdict(dict)
    with open(ROOT / f"custom_results/{folder}/eval/tracking_ratio.csv") as f:
        for r in csv.DictReader(f):
            data[(r["stem"], r["model"], r["variant"])] = (
                float(r["tracking_L"]),
                float(r["tracking_b"]),
                float(r["tracking_C"]))

    out = []
    for stem in STEMS:
        out.append(f"#### {stem}\n")

        # 이미지 grid (7 rows × 4 cols, target+models × variants)
        out.append("| | origin | warm | cool | dim |")
        out.append("|---|:---:|:---:|:---:|:---:|")
        out.append(f"| target | "
                   + " | ".join(
                       f"<img src=\"../outputs/{folder}/staging/face_{v}/{stem}.png\" width=\"140\">"
                       for v in VARIANTS) + " |")
        for mcs, label in MODELS:
            cells = [f"<img src=\"../outputs/{folder}/{mcs}_{v}/{stem}.png\" width=\"140\">"
                     for v in VARIANTS]
            out.append(f"| **{mcs} ({label})** | " + " | ".join(cells) + " |")
        out.append("")

        # 측정 표 (variant 3개 × L/b/C)
        out.append("| 모델 | warm L/b/C | cool L/b/C | dim L/b/C |")
        out.append("|------|:---:|:---:|:---:|")
        for mcs, label in MODELS:
            cells = []
            for v in ["warm", "cool", "dim"]:
                L, b, C = data[(stem, mcs, v)]
                cells.append(f"{L:+.2f} / {b:+.2f} / {C:+.2f}")
            out.append(f"| {mcs} ({label}) | " + " | ".join(cells) + " |")
        out.append("")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", choices=["1", "2"], required=True)
    args = p.parse_args()
    print(render(args.version))


if __name__ == "__main__":
    main()
