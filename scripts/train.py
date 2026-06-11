"""
Training entry point.

Usage:
  # Phase 1 (unbraid pretraining):
  python scripts/train.py --config configs/phase1_unbraid.yaml

  # Phase 2 (braid fine-tuning):
  python scripts/train.py --config configs/phase2_braid.yaml

  # Multi-GPU:
  accelerate launch scripts/train.py --config configs/phase1_unbraid.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import Trainer


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base_path = cfg.pop("base", None)
    if base_path:
        base_cfg = load_config(str(Path(config_path).parent / base_path))
        cfg = deep_merge(base_cfg, cfg)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume training from")
    parser.add_argument("--start_epoch", type=int, default=None, help="Resume epoch override (체크포인트에 epoch 키 없을 때)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.resume:
        cfg["training"]["resume"] = args.resume
    if args.start_epoch is not None:
        cfg["training"]["start_epoch"] = args.start_epoch
    print(f"Config loaded: phase={cfg['training']['phase']}, dataset={cfg['training']['dataset']}")

    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
