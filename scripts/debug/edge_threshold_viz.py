"""
edge_loss(SketchEdgeAlignmentLoss)의 stroke_threshold 검증용 시각화.

edge loss는 sketch_mask = (sketch.max(ch) > threshold) * matte 로 stroke 위치를 잡는다.
threshold가 0.1(구) → 1e-3(신)로 바뀌면서 마스크가 얼마나 넓어지는지,
그리고 "스케치 stroke만 깔끔히 걸리는지"를 눈으로 확인한다.

- 원본 스케치(raw)와, 학습 때 실제 들어가는 augmented 스케치(StrokeColorSampler+dark floor) 둘 다 처리.
- threshold 넘긴 픽셀만 남기고 나머지는 투명(alpha=0)인 RGBA PNG로 저장.
- test/edge_loss_threshold/{braid,unbraid}/ 에 저장.

실행:
  PYTHONPATH=. python3 scripts/debug/edge_threshold_viz.py
"""
from __future__ import annotations
import os, sys, glob, random
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from src.data.augmentation import StrokeColorSampler  # noqa: E402

THRESHOLDS = [1e-3, 0.1]         # 신(1e-3) vs 구(0.1)
BAND = (1e-3, 0.1)               # 두 threshold 사이 = 새로 포함되는 dark stroke 대역
N_PNG = 6                        # 세트별 PNG 저장 이미지 수
N_STATS = 150                    # 세트별 통계 집계 이미지 수
OUT_ROOT = "test/edge_loss_threshold"

SETS = {
    "braid":   "dataset/braid",
    "unbraid": "dataset/unbraid",
}
SPLIT = "train"


def load_u8(path, gray=False):
    im = Image.open(path).convert("L" if gray else "RGB")
    arr = np.asarray(im, dtype=np.uint8)
    if gray:
        arr = arr[None]
    else:
        arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr.copy())


def make_aug_sketch(stem, base):
    """학습 파이프라인과 동일하게 StrokeColorSampler 적용된 sketch(float [0,1], 3HW) 반환."""
    sketch_u8 = load_u8(f"{base}/sketch/{SPLIT}/{stem}.png")
    img_u8    = load_u8(f"{base}/img/{SPLIT}/{stem}.png")
    matte_u8  = load_u8(f"{base}/matte/{SPLIT}/{stem}.png", gray=True)
    sample = {
        "sketch":    sketch_u8.float() / 255.0,
        "sketch_u8": sketch_u8,
        "img_u8":    img_u8,
        "matte_u8":  matte_u8,
    }
    sampler = StrokeColorSampler(p=1.0)
    return sampler(sample)["sketch"], (matte_u8[0] > 0)


def thresh_rgba(sketch_f, th):
    """sketch_f: (3,H,W) float [0,1]. max_ch>th 픽셀만 불투명, 나머지 투명한 RGBA uint8 (H,W,4)."""
    mx = sketch_f.max(dim=0).values          # (H,W)
    keep = (mx > th).numpy()
    rgb = (sketch_f.clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    alpha = (keep * 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def coverage(sketch_f, matte_bool):
    mx = sketch_f.max(dim=0).values
    row = {}
    for th in THRESHOLDS:
        m = (mx > th)
        row[f"sk>{th:g}"] = 100 * m.float().mean().item()
        row[f"sk&matte>{th:g}"] = 100 * (m & matte_bool).float().mean().item()
    band = ((mx > BAND[0]) & (mx <= BAND[1]))
    row["band(0.001~0.1)"] = 100 * band.float().mean().item()
    row["band&matte"] = 100 * (band & matte_bool).float().mean().item()
    return row


def main():
    random.seed(0); torch.manual_seed(0)
    agg = {}
    for name, base in SETS.items():
        stems = sorted(os.path.splitext(os.path.basename(p))[0]
                       for p in glob.glob(f"{base}/sketch/{SPLIT}/*.png"))
        outdir = f"{OUT_ROOT}/{name}"
        os.makedirs(outdir, exist_ok=True)

        # --- PNG 저장 (raw & aug, threshold별) ---
        for stem in stems[:N_PNG]:
            raw = load_u8(f"{base}/sketch/{SPLIT}/{stem}.png").float() / 255.0
            aug, _ = make_aug_sketch(stem, base)
            for tag, sk in [("raw", raw), ("aug", aug)]:
                for th in THRESHOLDS:
                    rgba = thresh_rgba(sk, th)
                    Image.fromarray(rgba, "RGBA").save(f"{outdir}/{stem}_{tag}_th{th:g}.png")

        # --- 통계 집계 (raw / aug 각각) ---
        for tag in ("raw", "aug"):
            rows = []
            for stem in stems[:N_STATS]:
                if tag == "raw":
                    sk = load_u8(f"{base}/sketch/{SPLIT}/{stem}.png").float() / 255.0
                    mb = load_u8(f"{base}/matte/{SPLIT}/{stem}.png", gray=True)[0] > 0
                else:
                    sk, mb = make_aug_sketch(stem, base)
                rows.append(coverage(sk, mb))
            agg[(name, tag)] = {k: np.mean([r[k] for r in rows]) for k in rows[0]}

    # --- 리포트 ---
    print(f"\n=== 픽셀 커버리지 평균 (%), 세트별 {N_STATS}장 ===")
    hdr = list(next(iter(agg.values())).keys())
    print(f"{'set/tag':16s} " + " ".join(f"{h:>16s}" for h in hdr))
    for (name, tag), row in agg.items():
        print(f"{name+'/'+tag:16s} " + " ".join(f"{row[h]:16.3f}" for h in hdr))
    print(f"\nPNG 저장 위치: {OUT_ROOT}/{{braid,unbraid}}/  (raw/aug × th0.001/th0.1)")
    print("band(0.001~0.1) = threshold를 0.1→1e-3로 낮출 때 edge loss에 '새로 포함되는' dark stroke 비율.")
    print("이 값이 0에 가까우면 threshold 변경은 edge loss에 사실상 영향이 없다.")


if __name__ == "__main__":
    main()
