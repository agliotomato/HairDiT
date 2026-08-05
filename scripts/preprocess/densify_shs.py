"""SHS auto-completion 기반 stroke densification — 자립 단일 파일.

구버전 `densify_sketch_official.py` + `densify_sketch.py` 조합을 대체한다(둘 다 삭제됨).
우리 스크립트에 대한 의존이 없고, 외부 의존은 같은 디렉토리의 SHS 공개 코드
`unbraid_completion.getSketchCompletion` 하나뿐이다 — 이 함수는 **재구현하지 않고 그대로
호출**한다. 재구현하는 순간 "SHS 공개 구현을 그대로 사용"이라는 논문 방어가 무효화되므로,
`small_cc=240` · `matte>230` · `method='lee'` 는 그 함수 내부 값 그대로 유지되고 우리가
건드리는 인자는 `threshold` 하나뿐이다.

두 가지 산출물 (용도가 다름):

  masks   — 추가-stroke **이진 기하 마스크**. 학습용 오프라인 사전 생성
            (reports/2026-08-05-retrain-instructions-density-augmentation.md §2-3).
            기하는 색과 무관하므로 1회만 구우면 되고, 색 전파는 학습 로더가 매 스텝
            `StrokeColorSampler` 재착색 **이후**에 수행한다(§2-2 순서). 그래서 여기서는
            일부러 색을 입히지 않는다.

  colored — 색 전파까지 끝낸 **컬러 densified sketch**. 추론/평가용
            (`StrokeColorSampler` 가 돌지 않는 `infer_custom.py` 경로).
            `outputs/0805/densified_sketch/` 를 픽셀 단위로 재현한다.

사용:
  python scripts/preprocess/densify_shs.py masks \
      --sketch dataset/unbraid/sketch/train --matte dataset/unbraid/matte/train \
      --out    data/densify_masks --thresholds 21 15 12
  python scripts/preprocess/densify_shs.py colored \
      --sketch data/test/sketch_gt --matte data/test/matt \
      --out    data/densified_shs --thresholds 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))       # 같은 디렉토리의 SHS 원본을 참조

from unbraid_completion import getSketchCompletion   # SHS 공식 — threshold 인자화 외 수정 없음


# --------------------------------------------------------------------------- #
# 유틸
# --------------------------------------------------------------------------- #
def stroke_mask_from_sketch(sketch_bgr, thr=0):
    """검은 배경 위 컬러 stroke → 이진 마스크.

    thr=0 은 SHS 내부 기준(`sk_gray > 0`)과 맞춘 값이다. 예전 thr=20 은 어두운 헤어색
    stroke(예: CM_1084 — stroke 픽셀의 46.5% 가 max channel ≤20)를 stroke 로 못 잡아,
    색 복원도 추가-stroke 커버도 안 된 픽셀이 출력에서 검게 지워지는 버그가 있었다
    (2026-08-05 확인). 배경은 정확히 0 이라 thr=0 으로 낮춰도 부작용이 없다.
    """
    return sketch_bgr.max(axis=2) > thr


def propagate_color(sketch_bgr, src_mask, new_mask):
    """추가 stroke 에 **가장 가까운 원본 stroke 픽셀의 색**을 복사한다.

    SHS 코드는 이진 마스크만 반환하고(색 부여는 SHS 논문에도 없음), 우리 sketch 는 색으로
    헤어 색을 인코딩하므로 이 단계만 우리 추가분이다.
    """
    out = np.zeros_like(sketch_bgr)
    if not src_mask.any() or not new_mask.any():
        return out                                  # 전파할 시드가 없으면 빈 결과

    src_u8 = (~src_mask).astype(np.uint8)           # 0 = stroke(시드), 1 = 나머지
    _, labels = cv2.distanceTransformWithLabels(
        src_u8, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)

    ys, xs = np.nonzero(src_mask)
    lut = np.zeros(labels.max() + 1, dtype=np.int64)
    lut[labels[ys, xs]] = np.arange(len(ys))        # label → 시드 인덱스

    ny, nx = np.nonzero(new_mask)
    idx = lut[labels[ny, nx]]
    out[ny, nx] = sketch_bgr[ys[idx], xs[idx]]
    return out


# --------------------------------------------------------------------------- #
# 본체
# --------------------------------------------------------------------------- #
def densify_mask(sketch_bgr, matte_u8, threshold=15):
    """학습용: 추가-stroke 이진 기하 마스크만 반환. 색과 무관.

    반환: (mask (H,W) bool, info)
    """
    sk_gray = cv2.cvtColor(sketch_bgr, cv2.COLOR_BGR2GRAY)
    added_stroke, _ = getSketchCompletion(sk_gray, matte_u8, threshold=threshold)
    new_mask = added_stroke > 0

    src = stroke_mask_from_sketch(sketch_bgr)
    m = matte_u8 > 230                              # SHS 와 동일 기준으로 밀도 산출
    info = {
        "threshold": threshold,
        "orig_density": float(src.sum() / max(1, m.sum())),
        "new_density": float((src | new_mask).sum() / max(1, m.sum())),
        "added_px": int(new_mask.sum()),
    }
    return new_mask, info


def densify_colored(sketch_bgr, matte_u8, threshold=15):
    """추론/평가용: 색 전파 + blend 까지 끝낸 컬러 densified sketch.

    blend 규칙은 원본 우선 — 자동 stroke 는 **추가**만 하고 원본 stroke 를 대체하지 않는다
    (SHS §6.4: 수동 annotation 이 hair wisp junction 형성에 필수).

    반환: (densified_sketch_bgr, info)
    """
    new_mask, info = densify_mask(sketch_bgr, matte_u8, threshold)
    src = stroke_mask_from_sketch(sketch_bgr)

    if not src.any():                               # stroke 가 하나도 없으면 그대로 반환
        return sketch_bgr.copy(), info

    out = propagate_color(sketch_bgr, src, new_mask)
    out[src] = sketch_bgr[src]                      # 원본 stroke 보존
    return out, info


def viz_densified(orig_bgr, dens_bgr):
    """검증용: 원본 stroke = 그대로 / 추가 stroke = 초록 (SHS Fig.11 표기)."""
    src = stroke_mask_from_sketch(orig_bgr)
    new = stroke_mask_from_sketch(dens_bgr) & (~src)
    v = orig_bgr.copy()
    v[new] = (0, 255, 0)
    return v


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["masks", "colored"],
                    help="masks=학습용 이진 마스크 / colored=추론용 컬러 sketch")
    ap.add_argument("--sketch", required=True, help="sketch 디렉토리")
    ap.add_argument("--matte", required=True, help="matte 디렉토리 (sketch 와 같은 파일명)")
    ap.add_argument("--out", required=True, help="출력 루트 — 하위에 T{threshold}/ 로 저장")
    ap.add_argument("--thresholds", type=int, nargs="+", default=[21, 15, 12],
                    help="SHS threshold 목록 (클수록 약한 densification)")
    ap.add_argument("--viz-dir", default=None,
                    help="지정 시 추가 stroke 를 초록으로 칠한 검증 이미지도 저장")
    args = ap.parse_args()

    sk_dir, mt_dir, out_root = Path(args.sketch), Path(args.matte), Path(args.out)
    files = sorted(p for p in sk_dir.iterdir() if p.suffix.lower() in (".png", ".jpg"))
    if not files:
        sys.exit(f"sketch 파일 없음: {sk_dir}")

    for t in args.thresholds:
        (out_root / f"T{t}").mkdir(parents=True, exist_ok=True)
    if args.viz_dir:
        Path(args.viz_dir).mkdir(parents=True, exist_ok=True)

    densities = {t: [] for t in args.thresholds}
    for i, sp in enumerate(files, 1):
        sk = cv2.imread(str(sp))
        mp = mt_dir / f"{sp.stem}.png"
        mt = cv2.imread(str(mp), 0)
        if sk is None or mt is None:
            print(f"  [SKIP] 읽기 실패: {sp.name}")
            continue

        for t in args.thresholds:
            dst = out_root / f"T{t}" / f"{sp.stem}.png"
            if args.mode == "masks":
                mask, info = densify_mask(sk, mt, threshold=t)
                cv2.imwrite(str(dst), mask.astype(np.uint8) * 255)
            else:
                out, info = densify_colored(sk, mt, threshold=t)
                cv2.imwrite(str(dst), out)
                if args.viz_dir:
                    cv2.imwrite(str(Path(args.viz_dir) / f"T{t}_{sp.stem}.png"),
                                viz_densified(sk, out))
            densities[t].append(info["new_density"])

        if i % 200 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")

    print(f"\n완료: {out_root}/  ({args.mode}, {len(files)}장)")
    print(f"{'threshold':>10s}{'밀도 평균':>12s}{'std':>9s}")
    for t in args.thresholds:
        d = densities[t]
        print(f"{('T' + str(t)):>10s}{np.mean(d):12.4f}{np.std(d):9.4f}")


if __name__ == "__main__":
    main()
