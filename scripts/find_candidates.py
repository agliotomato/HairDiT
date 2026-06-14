"""SHS unbraid 후보 자동 검색.

기준:
  1. matte coverage (흰 픽셀 비율, 머리 영역 풍성)
  2. 정수리 덮음 (상단 50% 영역 매트 채움)
  3. hole 적음 (matte fill - matte 차이, hollow 영역 비율)
  4. sketch 색 다양성 (unique 색 개수, T6 재사용용)

종합 점수 = coverage * top_fill * (1 - hole_penalty) * sketch_diversity

상위 N개 stem 출력 (사용자 시각 검수용).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent


def load_l(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("L"))


def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))


def score_stem(stem: str, split: str = "unbraid") -> dict | None:
    matte_p  = ROOT / f"data/{split}/matte/test/{stem}.png"
    sketch_p = ROOT / f"data/{split}/sketch/test/{stem}.png"
    img_p    = ROOT / f"data/{split}/img_ori/test/{stem}.png"
    if not (matte_p.exists() and sketch_p.exists() and img_p.exists()):
        return None

    m = load_l(matte_p)
    s = load_rgb(sketch_p)
    h, w = m.shape

    bin_m = m > 127
    coverage = bin_m.mean()

    # 정수리 덮음: 상단 50% 영역의 matte 비율
    top_half = bin_m[:h // 2]
    top_fill = top_half.mean()

    # hole: matte close 후 차이
    closed = cv2.morphologyEx(
        bin_m.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        np.ones((25, 25), np.uint8),
    ) > 0
    hole = (closed & ~bin_m).sum()
    hole_ratio = hole / max(bin_m.sum(), 1)

    # sketch 색 다양성 (T6 후보용): 5-bit quantize 후 unique 색 수
    sq = (s >> 3) << 3
    flat = sq.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    n_colors = sum(1 for c in uniq if not (c[0] == 0 and c[1] == 0 and c[2] == 0))

    # 종합 점수
    score = (
        coverage
        * top_fill
        * max(0.0, 1.0 - 0.5 * hole_ratio)
        * min(1.0, n_colors / 8.0)
    )

    return {
        "stem":       stem,
        "coverage":   float(coverage),
        "top_fill":   float(top_fill),
        "hole_ratio": float(hole_ratio),
        "n_colors":   int(n_colors),
        "score":      float(score),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="unbraid")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()

    matte_dir = ROOT / f"data/{args.split}/matte/test"
    stems = sorted([p.stem for p in matte_dir.glob("*.png")])
    print(f"total stems: {len(stems)}")

    rows = []
    for s in stems:
        r = score_stem(s, args.split)
        if r is not None:
            rows.append(r)

    rows.sort(key=lambda r: -r["score"])
    print(f"\n=== top {args.top} candidates (score 내림차순) ===")
    print(f"{'stem':<14}{'coverage':>10}{'top_fill':>10}{'hole_ratio':>12}"
          f"{'n_colors':>10}{'score':>10}")
    print("-" * 66)
    for r in rows[: args.top]:
        print(f"{r['stem']:<14}{r['coverage']:>10.3f}{r['top_fill']:>10.3f}"
              f"{r['hole_ratio']:>12.3f}{r['n_colors']:>10d}{r['score']:>10.4f}")

    # CM_1005, CM_1101 reference 확인
    print("\n=== reference (현재 선정 OK 이미지) ===")
    print(f"{'stem':<14}{'coverage':>10}{'top_fill':>10}{'hole_ratio':>12}"
          f"{'n_colors':>10}{'score':>10}")
    for ref in ("CM_1005", "CM_1101"):
        r = next((r for r in rows if r["stem"] == ref), None)
        if r:
            rank = rows.index(r) + 1
            print(f"{r['stem']:<14}{r['coverage']:>10.3f}{r['top_fill']:>10.3f}"
                  f"{r['hole_ratio']:>12.3f}{r['n_colors']:>10d}{r['score']:>10.4f}"
                  f"   (rank {rank}/{len(rows)})")


if __name__ == "__main__":
    main()
