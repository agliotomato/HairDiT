"""GT 헤어 색으로 sketch stroke를 recolor (학습 분포 일치).

infer_custom.recolor_sketch_from_gt 를 pre-processing으로 분리.
재추론마다 매번 다시 돌리지 않게 사전 변환 → 같은 stem에 sparse·dense 둘 다 적용 가능.

reference (gt_img) = 항상 face_A (헤어 있는 GT). face_B(bald)로 돌리면 두피색이 추출돼 무의미.

Usage (single file):
  python scripts/recolor_sketch.py \\
      --sketch data/unbraid/sketch/test/CM_1005.png \\
      --face   data/unbraid/img_ori/test/CM_1005.png \\
      --matte  data/unbraid/matte/test/CM_1005.png \\
      --out    staging/T3_v3/sketch_recolor/CM_1005.png

Usage (directory + stems):
  python scripts/recolor_sketch.py \\
      --sketch data/unbraid/sketch/test \\
      --face   data/unbraid/img_ori/test \\
      --matte  data/unbraid/matte/test \\
      --stems  CM_1005 CM_1007 \\
      --out    staging/T3_v3/sketch_recolor
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.infer_custom import (
    load_face,
    load_matte,
    load_sketch,
    recolor_sketch_from_gt,
    to_pil,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sketch", type=Path, required=True,
                   help="sketch 파일 또는 디렉토리")
    p.add_argument("--face",   type=Path, required=True,
                   help="face_A 파일 또는 디렉토리 (GT 헤어 색 추출용)")
    p.add_argument("--matte",  type=Path, required=True,
                   help="matte 파일 또는 디렉토리")
    p.add_argument("--out",    type=Path, required=True,
                   help="출력 파일 또는 디렉토리")
    p.add_argument("--stems",  nargs="+", default=None,
                   help="처리할 stems (디렉토리 모드일 때)")
    args = p.parse_args()

    is_dir = args.sketch.is_dir()
    if is_dir:
        args.out.mkdir(parents=True, exist_ok=True)
        stems = args.stems or sorted([p.stem for p in args.sketch.glob("*.png")])
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        stems = [args.sketch.stem]

    for s in stems:
        sk_p = (args.sketch / f"{s}.png") if is_dir else args.sketch
        fa_p = (args.face   / f"{s}.png") if args.face.is_dir() else args.face
        mt_p = (args.matte  / f"{s}.png") if args.matte.is_dir() else args.matte
        if not (sk_p.exists() and fa_p.exists() and mt_p.exists()):
            print(f"  [SKIP] {s}: 입력 일부 없음")
            continue
        sk = load_sketch(sk_p)
        fa = load_face(fa_p)
        mt = load_matte(mt_p)
        out_t = recolor_sketch_from_gt(sk, fa, mt)
        out_p = (args.out / f"{s}.png") if is_dir else args.out
        to_pil(out_t.cpu()).save(out_p)
        print(f"  recolored: {out_p}")


if __name__ == "__main__":
    main()
