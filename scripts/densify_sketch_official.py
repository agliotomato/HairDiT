"""SHS 공개 auto-completion 코드(unbraid_completion.py) 기반 densification.

reports/20260804shssourcediffparametercorrection.md 의 권고:
자체 재구현(densify_sketch.py) 대신 SHS 가 공개한 `getSketchCompletion()` 을 그대로 쓴다.
원본에서 유일하게 바꾼 것은 하드코딩된 `threshold = 15` 를 함수 인자로 뺀 것(unbraid_completion.py)
뿐이고, 그 외 알고리즘은 전부 원본 그대로다. SHS 코드는 이진 마스크만 반환하므로(하류에서 색
부여는 SHS 논문에도 없음), 우리 sketch가 색으로 헤어 색을 인코딩하는 것에 맞춰 색 전파
(`_propagate_color`, densify_sketch.py 재사용) + blend(원본 우선)만 추가한다.

threshold 의미: SHS 원본 stroke 로부터의 Euclidean 거리(px) — 이 값보다 먼 영역만 gap 후보.
SHS 기본값 15. 우리 기존 K sweep(K=25/15/11, L∞ 반경 12/7/5)은 전부 이 기본값(15)보다
공격적이었음(§2-1) — 그래서 sweep 을 {15(SHS 기본), 12, 9, 6}으로 다시 잡는다.

  python scripts/densify_sketch_official.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from densify_sketch import _propagate_color, stroke_mask_from_sketch, viz_densified
from unbraid_completion import getSketchCompletion


def densify_sketch_shs(sketch_bgr, matte_u8, threshold=15):
    """SHS 공개 코드 그대로 + 색 전파/blend 만 추가.

    sketch_bgr: (H,W,3) uint8, 검은 배경 + 컬러 stroke
    matte_u8  : (H,W) uint8 grayscale, 흰색=hair 영역 (SHS 코드가 내부에서 >230 으로 이진화)
    반환      : (densified_sketch_bgr, info_dict)
    """
    sk_gray = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
    added_stroke, _ = getSketchCompletion(sk_gray, matte_u8, threshold=threshold)  # 이진 마스크
    new_mask = added_stroke > 0

    # 우리 추가분: 색 전파 + blend (원본 우선) — SHS 코드에는 없음(이진 마스크만 반환)
    src = stroke_mask_from_sketch(sketch_bgr)
    out = _propagate_color(sketch_bgr, src, new_mask)
    out[src] = sketch_bgr[src]

    m = matte_u8 > 230
    info = {
        "threshold": threshold,
        "orig_density": float(src.sum() / max(1, m.sum())),
        "new_density": float((src | new_mask).sum() / max(1, m.sum())),
        "added_px": int(new_mask.sum()),
    }
    return out, info


if __name__ == "__main__":
    IDS = ["CM_1067", "CM_1082"]
    THRESHOLDS = {"T15_shs_default": 15, "T12": 12, "T9": 9, "T6": 6}
    SKETCH_DIR = "data/test/sketch_gt"
    MATTE_DIR = "data/test/matt"

    os.makedirs("outputs/0804/densified_viz_shs", exist_ok=True)
    for name, thr in THRESHOLDS.items():
        out_dir = f"data/densified_shs/{name}"
        os.makedirs(out_dir, exist_ok=True)
        for i in IDS:
            sk = cv2.imread(f"{SKETCH_DIR}/{i}.png")
            mt = cv2.imread(f"{MATTE_DIR}/{i}.png", 0)
            out, info = densify_sketch_shs(sk, mt, threshold=thr)
            cv2.imwrite(f"{out_dir}/{i}.png", out)

            v = viz_densified(sk, out)
            cv2.imwrite(f"outputs/0804/densified_viz_shs/{name}_{i}.png", v)

            print(name, i, info)
