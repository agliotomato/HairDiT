"""C″ 리라이팅 (T6 입력) — design.md L51-66 그대로.

Warm: gain (R,G,B) = (×1.18, ×1.03, ×0.82)  — 6500K → ~4300K, Lab Δb ≈ +12~15
Cool: gain         = (×0.84, ×0.98, ×1.18)  — 6500K → ~9500K, Lab Δb ≈ −12~15
Dim:  전 채널 ×0.55                          — ΔL ≈ −20~25

구현:
  1. sRGB → linear (표준 sRGB EOTF)
  2. ×0.95 pre-scale (warm R 클리핑 방지)
  3. gain 적용
  4. linear → sRGB
  5. clip [0, 1]

Usage:
  python scripts/relight_c.py \\
      --src   data/unbraid/img_ori/test \\
      --stems CM_1005 CM_1033 ... \\
      --out   staging/T6_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


TRANSFORMS_PI = {
    "warm": (1.18, 1.03, 0.82),
    "cool": (0.84, 0.98, 1.18),
    "dim":  (0.55, 0.55, 0.55),
}

# Strong: design.md 목표 Δb +12~15, ΔL −20~25 만족용. raw gain ≈ PI 표 × 1.5~2.
TRANSFORMS_STRONG = {
    "warm": (1.35, 1.05, 0.70),
    "cool": (0.70, 0.95, 1.35),
    "dim":  (0.40, 0.40, 0.40),
}

TRANSFORMS_SETS = {"pi": TRANSFORMS_PI, "strong": TRANSFORMS_STRONG}


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)


def apply_relight(img_uint8: np.ndarray, gain_rgb: tuple,
                   prescale: float = 0.95) -> np.ndarray:
    """sRGB uint8 → relight → sRGB uint8.

    linear 공간에서 gain 적용. warm R 클리핑 방지 위해 pre-scale.
    """
    x = img_uint8.astype(np.float64) / 255.0
    lin = srgb_to_linear(x)
    lin = lin * prescale                  # pre-scale (warm 포화 방지)
    g = np.array(gain_rgb, dtype=np.float64).reshape(1, 1, 3)
    lin = lin * g
    out = linear_to_srgb(lin)
    return (out * 255.0).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src",   type=Path, required=True,
                   help="face_A 디렉토리 (예: data/unbraid/img_ori/test)")
    p.add_argument("--stems", nargs="+", required=True)
    p.add_argument("--out",   type=Path, required=True,
                   help="staging/T6_v1 — 안에 face_origin/warm/cool/dim 4 폴더 생성")
    p.add_argument("--prescale", type=float, default=0.95)
    p.add_argument("--set", choices=["pi", "strong"], default="pi",
                   help="pi (design.md L53 표) 또는 strong (design.md 목표 시프트 달성용 강한 gain)")
    args = p.parse_args()
    TRANSFORMS = TRANSFORMS_PI if args.set == "pi" else TRANSFORMS_STRONG
    print(f"transforms = {args.set} : {TRANSFORMS}, prescale={args.prescale}")

    out_dirs = {
        "origin": args.out / "face_origin",
        "warm":   args.out / "face_warm",
        "cool":   args.out / "face_cool",
        "dim":    args.out / "face_dim",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    for stem in args.stems:
        src_p = args.src / f"{stem}.png"
        if not src_p.exists():
            print(f"  [SKIP] {src_p} 없음")
            continue
        img = np.array(Image.open(src_p).convert("RGB"))

        # origin = 원본 그대로 (deep copy 의미상 cp)
        Image.fromarray(img).save(out_dirs["origin"] / f"{stem}.png")

        for name, gain in TRANSFORMS.items():
            out = apply_relight(img, gain, prescale=args.prescale)
            Image.fromarray(out).save(out_dirs[name] / f"{stem}.png")

        print(f"  {stem}: origin + warm/cool/dim → {args.out}/face_*")


if __name__ == "__main__":
    main()
