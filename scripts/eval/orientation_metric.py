"""머릿결 방향 지표 — structure tensor (reports/2026-08-04-orientation-metric-implementation-guide.md).

국소 gradient 공분산(2x2)의 주고유벡터에 수직인 방향을 머릿결 방향 theta 로,
고유값 차이 비를 coherence 로 쓴다. theta 는 double-angle 로 비교한다.

집계 두 가지:
  1) GT 오차     — 생성 결과 vs GT 의 평균 각도차. 지침 §2 형식.
  2) seed 불일치 — 같은 run 안에서 seed 쌍끼리의 평균 각도차. GT 를 거치지 않는다.

  python scripts/eval/orientation_metric.py            # 검증 3종 + 표 (기본 sigma_i=3, erode_px=6)
  python scripts/eval/orientation_metric.py --validate # 검증 3종만
  python scripts/eval/orientation_metric.py --floor    # GT 내재 복잡도 진단

판정용 지표이며 loss 로 학습에 넣지 않는다.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent

RUNS = {
    "mcs2":   ROOT / "outputs/0803/seed_mcs2",
    "run2p1": ROOT / "outputs/0803/seed_run2",
    "run4":   ROOT / "outputs/0803/seed_run4",
}
SEEDS = [42, 1, 2, 3]
IMAGES = ["CM_1067", "CM_1082"]

GT_DIR = ROOT / "data/test/ori_image"      # hair 가 있는 원본 (bald 인 data/*/img 아님)
MATTE_DIR = ROOT / "data/test/matt"
OUT_DIR = ROOT / "outputs/0803/orientation"

SIGMA_I = 3.0      # §3 캘리브레이션 결과 (3/5/8 중 판별 gap 최대)
ERODE_PX = 6


# --------------------------------------------------------------------------- #
# core
# --------------------------------------------------------------------------- #
def hair_orientation(img, sigma_d=1.0, sigma_i=SIGMA_I):
    """img: (H,W,3) uint8 BGR 또는 (H,W) grayscale.

    반환: theta (H,W) 머릿결 방향 [0,pi), coherence (H,W) [0,1]
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    g = gray.astype(np.float64) / 255.0

    # (1) gradient (미분 스케일 sigma_d: 살짝 블러 후 Sobel)
    g = cv2.GaussianBlur(g, (0, 0), sigma_d)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)

    # (2) structure tensor 성분을 윈도(sigma_i)로 평활 — "국소 공분산"
    Jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_i)
    Jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_i)
    Jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_i)

    # (3) 고유분해 닫힌형: gradient 우세 방향 = 0.5*atan2(2Jxy, Jxx-Jyy)
    #     머릿결은 그 수직 → +pi/2
    theta = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) + np.pi / 2.0
    theta = np.mod(theta, np.pi)                      # [0, pi) — 180도 주기

    # (4) coherence = (lam1-lam2)/(lam1+lam2)
    lam_diff = np.sqrt((Jxx - Jyy) ** 2 + 4.0 * Jxy ** 2)
    coherence = lam_diff / (Jxx + Jyy + 1e-12)
    return theta, coherence


def angular_error_map(th_a, th_b):
    """double-angle 각도차 [0, pi/2] rad. theta = theta+180도 모호성 처리."""
    d2 = np.arctan2(np.sin(2 * (th_a - th_b)), np.cos(2 * (th_a - th_b)))
    return 0.5 * np.abs(d2)


def erode_matte(matte, erode_px=ERODE_PX):
    """matte 경계는 실루엣 gradient 가 오염시키므로 침식."""
    m = (matte > 0.5).astype(np.uint8)
    return cv2.erode(m, np.ones((erode_px, erode_px), np.uint8)).astype(bool)


def orientation_error(img_gen, img_gt, matte, sigma_i=SIGMA_I, erode_px=ERODE_PX):
    """반환: (평균 각도 오차 [deg], 생성영상 평균 coherence).

    matte: (H,W) [0,1] — GT hair 영역
    """
    th_g, co_g = hair_orientation(img_gen, sigma_i=sigma_i)
    th_t, co_t = hair_orientation(img_gt, sigma_i=sigma_i)
    m = erode_matte(matte, erode_px)

    # GT coherence 로 가중 — GT 가 평평/흐릿한 픽셀의 각도는 무의미
    w = co_t * m
    mean_err_deg = np.degrees((angular_error_map(th_g, th_t) * w).sum() / (w.sum() + 1e-12))
    return mean_err_deg, float(co_g[m].mean())


def seed_disagreement(gens, matte, sigma_i=SIGMA_I, erode_px=ERODE_PX):
    """같은 run 의 seed 쌍끼리 평균 각도차 [deg]. GT 방향을 거치지 않는다.

    std 는 seed 별 '오차 크기'의 변동만 재므로, 두 seed 가 서로 반대로 틀려도
    같은 값이 나온다. 이 지표는 방향 차이를 직접 재서 그 사각지대를 메운다.

    gens: {seed: img}. 반환: (쌍 평균, 쌍 std)
    """
    m = erode_matte(matte, erode_px)
    st = {s: hair_orientation(g, sigma_i=sigma_i) for s, g in gens.items()}
    out = []
    for a, b in itertools.combinations(sorted(gens), 2):
        # 두 쪽 다 방향이 또렷한 곳에서만 — 한쪽이 흐리면 각도차가 무의미
        w = np.minimum(st[a][1], st[b][1]) * m
        d = angular_error_map(st[a][0], st[b][0])
        out.append(np.degrees((d * w).sum() / (w.sum() + 1e-12)))
    return float(np.mean(out)), float(np.std(out, ddof=1))


def gt_intrinsic_floor(img_gt, matte, sigma_i=SIGMA_I, erode_px=ERODE_PX, sigma_ref=20.0):
    """GT 자신의 스케일 간 방향 흔들림 [deg].

    실사진 GT 는 웨이브·레이어 겹침 때문에 국소 방향이 stroke 를 따라 일방적으로
    흐르지 않는다. sigma_i 에서 잰 theta 와 sigma_ref 에서 잰 theta 의 차이가 그 양이고,
    매끄러운 예측이라면 무엇을 내놔도 이 아래로는 못 내려가는 바닥이 된다.
    orientation_error 의 절대값을 '모델 정확도' 로 읽으면 안 되는 이유.
    """
    th, co = hair_orientation(img_gt, sigma_i=sigma_i)
    th_ref, _ = hair_orientation(img_gt, sigma_i=sigma_ref)
    w = co * erode_matte(matte, erode_px)
    return np.degrees((angular_error_map(th, th_ref) * w).sum() / (w.sum() + 1e-12))


def viz(img, theta, coherence):
    """방향=색상, coherence=채도. 같은 결 방향 = 같은 색."""
    hsv = np.zeros((*theta.shape, 3), np.uint8)
    hsv[..., 0] = (theta / np.pi * 180).astype(np.uint8)   # 색상 = 방향
    hsv[..., 1] = (coherence * 255).astype(np.uint8)       # 채도 = 확신도
    hsv[..., 2] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #
def load_gt(img_id):
    for ext in (".png", ".jpg"):
        p = GT_DIR / f"{img_id}{ext}"
        if p.exists():
            return cv2.imread(str(p))
    raise FileNotFoundError(f"GT hair 원본 없음: {GT_DIR}/{img_id}.*")


def load_matte(img_id):
    m = cv2.imread(str(MATTE_DIR / f"{img_id}.png"), 0)
    if m is None:
        raise FileNotFoundError(f"matte 없음: {MATTE_DIR}/{img_id}.png")
    return m.astype(np.float64) / 255.0


def load_gen(run, seed, img_id):
    p = RUNS[run] / str(seed) / f"{img_id}.png"
    g = cv2.imread(str(p))
    if g is None:
        raise FileNotFoundError(f"렌더 없음: {p}")
    return g


# --------------------------------------------------------------------------- #
# 검증 3종 (표 만들기 전에 — 지침 §5)
# --------------------------------------------------------------------------- #
def validate(sigma_i=SIGMA_I, erode_px=ERODE_PX):
    """1) GT vs GT ~ 0도, 2) GT vs 90도 회전 ~ 90도, 3) 시각화 저장."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = True

    for img_id in IMAGES:
        gt, matte = load_gt(img_id), load_matte(img_id)
        m = erode_matte(matte, erode_px)

        e_same, _ = orientation_error(gt, gt, matte, sigma_i, erode_px)

        # 이미지를 돌리면 픽셀 대응이 깨지므로 theta 맵을 원좌표로 되돌린 뒤 비교한다.
        # (돌린 이미지를 그대로 orientation_error 에 넣으면 서로 다른 픽셀을 비교하게 됨)
        th_gt, co_gt = hair_orientation(gt, sigma_i=sigma_i)
        th_rot, _ = hair_orientation(np.ascontiguousarray(np.rot90(gt, 1)), sigma_i=sigma_i)
        w = co_gt * m
        e_rot = np.degrees((angular_error_map(np.rot90(th_rot, -1), th_gt) * w).sum()
                           / (w.sum() + 1e-12))

        p1, p2 = abs(e_same) < 0.5, abs(e_rot - 90.0) < 5.0
        ok = ok and p1 and p2
        print(f"  [{img_id}] GT vs GT        = {e_same:6.2f} deg (기대 ~0)   {'OK' if p1 else 'FAIL'}")
        print(f"  [{img_id}] GT vs 90도 회전 = {e_rot:6.2f} deg (기대 ~90)  {'OK' if p2 else 'FAIL'}")

    for img_id in IMAGES:
        gt, matte = load_gt(img_id), load_matte(img_id)
        m = erode_matte(matte, erode_px)[..., None]
        panels = []
        for label, im in [("GT", gt)] + [(r, load_gen(r, 42, img_id)) for r in RUNS]:
            th, co = hair_orientation(im, sigma_i=sigma_i)
            v = viz(im, th, co) * m                    # hair 영역 밖은 검게
            cv2.putText(v, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            panels.append(np.hstack([im, v]))
        out = OUT_DIR / f"viz_{img_id}_sigma{sigma_i:g}.png"
        cv2.imwrite(str(out), np.vstack(panels))
        print(f"  [{img_id}] 시각화 저장 → {out.relative_to(ROOT)}")

    print(f"\n  검증 결과: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# 표
# --------------------------------------------------------------------------- #
def measure_all(sigma_i=SIGMA_I, erode_px=ERODE_PX):
    """반환: {(run, img_id, seed): (err_deg, coh)}, {(run, img_id): (불일치, 불일치std)}"""
    err, dis = {}, {}
    for img_id in IMAGES:
        gt, matte = load_gt(img_id), load_matte(img_id)
        for run in RUNS:
            gens = {s: load_gen(run, s, img_id) for s in SEEDS}
            for s in SEEDS:
                err[(run, img_id, s)] = orientation_error(gens[s], gt, matte, sigma_i, erode_px)
            dis[(run, img_id)] = seed_disagreement(gens, matte, sigma_i, erode_px)
    return err, dis


def calibrate(sigmas=(3.0, 5.0, 8.0), erode_px=ERODE_PX):
    """판별력 기준: mcs2(정렬) 와 run4 seed42(어긋남) 의 오차가 가장 크게 벌어지는 sigma_i."""
    rows = []
    for s in sigmas:
        e, _ = measure_all(s, erode_px)
        mcs2 = np.mean([e[("mcs2", i, sd)][0] for i in IMAGES for sd in SEEDS])
        run4_42 = np.mean([e[("run4", i, 42)][0] for i in IMAGES])
        rows.append((s, run4_42 - mcs2))
        print(f"  sigma_i={s:>3g}: mcs2={mcs2:5.2f}  run4 seed42={run4_42:5.2f}  "
              f"gap={run4_42 - mcs2:+5.2f} ({(run4_42 - mcs2) / mcs2 * 100:+.1f}%)")
    best = max(rows, key=lambda x: x[1])[0]
    print(f"  → 선택: sigma_i = {best:g}")
    return best


def print_table(err, dis):
    for img_id in IMAGES:
        print(f"\n  === {img_id} ===")
        print(f"  {'model':8s}" + "".join(f"{'seed' + str(s):>9s}" for s in SEEDS)
              + f"{'mean+-std':>15s}{'coh':>8s}{'seed불일치':>14s}")
        for run in RUNS:
            e = [err[(run, img_id, s)][0] for s in SEEDS]
            c = np.mean([err[(run, img_id, s)][1] for s in SEEDS])
            d, ds = dis[(run, img_id)]
            print(f"  {run:8s}" + "".join(f"{v:9.2f}" for v in e)
                  + f"  {np.mean(e):6.2f}+-{np.std(e, ddof=1):.2f}{c:8.3f}"
                  + f"  {d:7.2f}+-{ds:.2f}")


def print_floor(err, sigma_i=SIGMA_I, erode_px=ERODE_PX):
    for img_id in IMAGES:
        floor = gt_intrinsic_floor(load_gt(img_id), load_matte(img_id), sigma_i, erode_px)
        obs = np.mean([err[(r, img_id, s)][0] for r in RUNS for s in SEEDS])
        print(f"  {img_id}: GT 내재 바닥 {floor:5.2f} deg / 측정 오차 평균 {obs:5.2f} deg "
              f"= {floor / obs * 100:.0f}%")
    print("  → 절대값은 모델 정확도가 아니다. run 간 상대 비교로만 읽을 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="검증 3종만 실행")
    ap.add_argument("--floor", action="store_true", help="GT 내재 복잡도 진단도 출력")
    ap.add_argument("--sigma-i", type=float, default=SIGMA_I, help=f"sigma_i (기본값: {SIGMA_I:g})")
    ap.add_argument("--erode-px", type=int, default=ERODE_PX)
    args = ap.parse_args()

    print("[1] 검증 3종 (지침 §5)")
    if not validate(sigma_i=args.sigma_i, erode_px=args.erode_px):
        print("  검증 실패 — 표를 만들기 전에 파라미터/입력을 점검할 것", file=sys.stderr)
        return 1
    if args.validate:
        return 0

    sigma_i = args.sigma_i

    print(f"\n[2] 표 — sigma_i={sigma_i:g}, erode_px={args.erode_px}")
    err, dis = measure_all(sigma_i, args.erode_px)
    print_table(err, dis)

    if args.floor:
        print("\n[3] GT 내재 복잡도 진단")
        print_floor(err, sigma_i, args.erode_px)
    return 0


if __name__ == "__main__":
    sys.exit(main())
