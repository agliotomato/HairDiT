"""Cross-TYPE 실험 사전 준비 — 교차 매핑 + face symlink + GT-recolor sketch.

Cross-TYPE (sketch와 image를 반대 유형으로 교차):
  sketch = matte = 샘플 A  (해당 유형)
  face/img        = 샘플 B  (반대 유형의 이미지)
    braid  sketch(A) → unbraid 이미지(B)
    unbraid sketch(A) → braid  이미지(B)

개수 비대칭 처리 (중복 허용):
  braid(107) → unbraid(466) : 중복 없이 107개 샘플
  unbraid(466) → braid(107) : 풀이 107개뿐 → braid 이미지 ~4.4회 반복 사용 (균등 분배)

생성물 (out-root, 기본 experiment_cross_id/):
  tripletmap_{A_type}.json                          {A_id: B_id}  (A_type = sketch 유형)
  face_links/{A_type}/{A_id}.png  -> dataset/{B_type}/img/test/{B_id}.png  (심볼릭)
  gt_recolored_sketch/{A_type}/{A_id}.png           A 자신의 (img, matte)로 recolor

주의:
  - 매핑 시드(기본 42)로 유형별 1개만 생성 → mcs1~6·전 노이즈시드 동일 재사용 (공정 비교).
  - GT recolor 색은 **A 자신의 실제 헤어색** (A_img×A_matte), B 아님 / 배경 아님.
  - face symlink trick으로 infer_custom.py의 stem 매칭(출력 stem = A_id)을 그대로 활용.
    run/eval은 sketch 유형 기준으로 동작하므로 수정 불필요.

실행 (프로젝트 env, torch 필요):
  python scripts/preprocess/gen_cross_id_map.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.infer_custom import (  # noqa: E402
    load_sketch, load_matte, load_face, recolor_sketch_from_gt, to_pil,
)

# (A_type = sketch, B_type = image) — 반대 유형으로 교차
DIRECTIONS = [("braid", "unbraid"), ("unbraid", "braid")]


def cross_map(a_ids: list[str], b_ids: list[str], seed: int) -> dict[str, str]:
    """A(sketch) → B(image, 반대 유형). |B|<|A|면 중복 허용·균등 분배.

    A_type ≠ B_type 이므로 stem 접두어가 달라 A==B 자기충돌 불가.
    """
    rng = random.Random(seed)
    if len(b_ids) >= len(a_ids):
        pool = b_ids[:]
        rng.shuffle(pool)
        return {a: pool[i] for i, a in enumerate(a_ids)}
    # 중복 허용: b_ids를 타일링→셔플→앞부분 사용 (각 B를 floor/ceil회 균등)
    reps = (len(a_ids) + len(b_ids) - 1) // len(b_ids)
    pool = b_ids * reps
    rng.shuffle(pool)
    return {a: pool[i] for i, a in enumerate(a_ids)}


def list_stems(d: Path) -> list[str]:
    return sorted(p.stem for p in d.glob("*.png"))


def process_direction(a_type: str, b_type: str, dataset_root: Path,
                      out_root: Path, seed: int):
    a_sketch = dataset_root / a_type / "sketch" / "test"
    a_img    = dataset_root / a_type / "img"    / "test"
    a_matte  = dataset_root / a_type / "matte"  / "test"
    b_img    = dataset_root / b_type / "img"    / "test"
    for d in (a_sketch, a_img, a_matte, b_img):
        if not d.is_dir():
            raise FileNotFoundError(d)

    # A: gt recolor에 자기 img+matte 필요 → 셋 다 존재하는 stem만
    a_ids = [s for s in list_stems(a_sketch)
             if (a_img / f"{s}.png").exists() and (a_matte / f"{s}.png").exists()]
    b_ids = list_stems(b_img)
    print(f"\n[{a_type} sketch → {b_type} image] A={len(a_ids)}  B풀={len(b_ids)}")

    mapping = cross_map(a_ids, b_ids, seed)
    used = len(set(mapping.values()))
    print(f"  B 사용: {used}/{len(b_ids)}개"
          + (f"  (중복 허용, 평균 {len(a_ids)/len(b_ids):.1f}회)" if len(b_ids) < len(a_ids) else "  (중복 없음)"))

    # 1) tripletmap json (sketch 유형 이름으로 저장)
    out_root.mkdir(parents=True, exist_ok=True)
    map_path = out_root / f"tripletmap_{a_type}.json"
    map_path.write_text(json.dumps(mapping, indent=2))
    print(f"  매핑 저장: {map_path}")

    # 2) face symlink: {A_id}.png -> abs(dataset/{B_type}/img/test/{B_id}.png)
    link_dir = out_root / "face_links" / a_type
    link_dir.mkdir(parents=True, exist_ok=True)
    for a_id, b_id in mapping.items():
        link = link_dir / f"{a_id}.png"
        target = (b_img / f"{b_id}.png").resolve()
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    print(f"  face symlink {len(mapping)}개 → {b_type} img: {link_dir}")

    # 3) GT-recolor sketch (A 자신의 img×matte로)
    gt_dir = out_root / "gt_recolored_sketch" / a_type
    gt_dir.mkdir(parents=True, exist_ok=True)
    for a_id in tqdm(a_ids, desc=f"  recolor[{a_type}]"):
        sketch = load_sketch(a_sketch / f"{a_id}.png")
        img    = load_face(a_img      / f"{a_id}.png")
        matte  = load_matte(a_matte   / f"{a_id}.png")
        recolored = recolor_sketch_from_gt(sketch, img, matte)
        to_pil(recolored.cpu()).save(gt_dir / f"{a_id}.png")
    print(f"  gt_recolored_sketch {len(a_ids)}개: {gt_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    p.add_argument("--out-root",     type=Path, default=ROOT / "experiment_cross_id")
    p.add_argument("--seed",         type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    for a_type, b_type in DIRECTIONS:
        process_direction(a_type, b_type, args.dataset_root, args.out_root, args.seed)
    print("\n완료.")


if __name__ == "__main__":
    main()
