"""체크포인트에서 추론용 ControlNet 가중치만 추출 → epoch_40_infer.pth 저장 (1회성).

full 체크포인트(~20GB)는 optimizer/EMA 등 학습 캐시를 포함한다. 추론엔 ControlNet
state_dict(~3GB)만 필요하므로, 이를 미리 뽑아 `*_infer.pth`로 저장해두면:
  - infer_custom.py가 `_infer.pth`를 우선 로드 (WARNING 사라짐)
  - CPU RAM 로드량 20GB → 3GB, 로딩 시간 단축

전부 CPU에서 동작 (GPU 불필요). 파일당 ~20GB를 잠깐 RAM에 올리므로 순차 처리.

사용 (torch 있는 프로젝트 env):
  python scripts/preprocess/strip_checkpoint.py
  python scripts/preprocess/strip_checkpoint.py --glob 'checkpoints/mcs*/epoch_40.pth' --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def strip_one(ckpt_path: Path, force: bool):
    out_path = ckpt_path.with_name(ckpt_path.stem + "_infer.pth")
    if out_path.exists() and not force:
        print(f"  [SKIP] 이미 존재: {out_path.name}")
        return
    # mmap=True: 20GB를 통째로 RAM에 올리지 않고 메모리맵 → controlnet 텐서만 실제 materialize.
    # (RAM 15GB 머신에서 full 로드 시 OOM 방지)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    if "controlnet" not in ckpt:
        print(f"  [WARN] 'controlnet' 키 없음 → 건너뜀: {ckpt_path}  (keys={list(ckpt)[:6]})")
        return
    state = ckpt["controlnet"]
    del ckpt
    torch.save({"controlnet": state}, out_path)
    size_gb = out_path.stat().st_size / 1e9
    print(f"  저장: {out_path.name}  ({size_gb:.2f} GB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob",  default="checkpoints/mcs*/epoch_40.pth",
                   help="ROOT 기준 체크포인트 glob 패턴")
    p.add_argument("--force", action="store_true", help="기존 _infer.pth 덮어쓰기")
    args = p.parse_args()

    paths = sorted(ROOT.glob(args.glob))
    if not paths:
        print(f"매칭된 체크포인트 없음: {args.glob}")
        return
    print(f"{len(paths)}개 체크포인트 처리\n")
    for cp in paths:
        print(f"[{cp.parent.name}/{cp.name}]")
        strip_one(cp, args.force)
    print("\n완료.")


if __name__ == "__main__":
    main()
