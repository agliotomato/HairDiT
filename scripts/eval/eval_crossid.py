"""Cross-identity 평가 driver — 기존 eval_metrics.py × 3 노이즈시드 집계.

지표:
  ① Sketch LPIPS ↓  (구조,     paired vs sketch, hair 영역)   → macro        [전 버전]
  ② Edge IoU ↑      (구조 보조, paired vs sketch, hair 영역)   → macro        [전 버전]
  ③ Hair FID ↓      (리얼리즘,  unpaired,         hair 크롭)   → combined 573 [전 버전]
  ④ LPIPS (GT) ↓    (외형,      paired vs GT,     hair-masked) → macro        [전 버전]
  + PSNR (hair) ↑   (화질,      paired vs GT,     hair-masked) → macro        [전 버전]
  + ΔE2000(GT,hair)↓(색상,      paired vs GT,     hair mask)   → macro        [gt_sketch만]

  → ΔE는 GT-recolor(sketch가 A 실제색) 버전에서만 의미 있음: 출력 헤어색이
    A 실제 헤어색으로 얼마나 정확히 가는가(누설 시 B쪽으로 틀어짐).

cross-id에서 출력 stem = A_id 이므로 eval_metrics가 자동으로 잡는
GT/sketch/matte가 전부 A 기준 → "B 배경에서 생성했어도 출력 헤어가
A 실제 외형(sketch·GT)에 얼마나 충실한가"를 그대로 측정. (외형 누설 ↓ = 충실 ↑)

흐름:
  (mcs, version)별 × seed{0,1,2}:
    eval_metrics.py --split combined --pred <braid> <unbraid> --metrics ...
  → seed별 summary.csv 읽어 mean ± std (시드 간 분산 = robustness)

전제: scripts/run_cross_id_experiment.sh 로 outputs/crossid/ 생성 완료.

사용:
  python scripts/eval/eval_crossid.py
  python scripts/eval/eval_crossid.py --seeds 0 1 2 --mcs 1 2 3 4 5 6
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# 버전별 eval_metrics --metrics 세트 (ΔE는 gt_sketch에만)
METRICS_BY_VERSION = {
    "original_sketch": "sketch_lpips,edge_iou,hair_fid,lpips_gt,psnr",
    "gt_sketch":       "sketch_lpips,edge_iou,hair_fid,lpips_gt,psnr,delta_e_gt",
}
# 표시 순서: (summary.csv label, 집계 컬럼) — delta_e는 gt_sketch에서만 값, 나머진 N/A
REPORT = [
    ("Sketch LPIPS ↓",       "macro"),
    ("Edge IoU ↑",           "macro"),
    ("Hair FID ↓",           "combined573"),
    ("LPIPS (GT) ↓",         "macro"),
    ("PSNR (hair) ↑",        "macro"),
    ("ΔE2000 (GT, hair) ↓",  "macro"),
]
MODELS = [("mcs1", "Ours"), ("mcs2", "Ours+Gate"),
          ("mcs3", "Sketch-only"), ("mcs4", "Sketch-only+Gate"),
          ("mcs5", "Raw-only"), ("mcs6", "Matte-CNN-only")]
VERSIONS = ["gt_sketch"]


def read_summary(path: Path) -> dict[str, dict]:
    """label -> row dict."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["metric"]] = r
    return out


def run_eval_one(mkey: str, version: str, seed: int, out_root: Path,
                 eval_root: Path, force: bool) -> Path | None:
    braid   = out_root / "braid"   / f"{mkey}_{version}" / f"seed{seed}"
    unbraid = out_root / "unbraid" / f"{mkey}_{version}" / f"seed{seed}"
    if not (braid.is_dir() and unbraid.is_dir()):
        print(f"  [SKIP] {mkey}_{version} seed{seed}: 출력 폴더 없음")
        return None
    out_prefix = eval_root / f"{mkey}_{version}_seed{seed}"
    summ = Path(f"{out_prefix}_summary.csv")
    if summ.exists() and not force:
        return summ
    cmd = [sys.executable, str(ROOT / "scripts/eval_metrics.py"),
           "--split", "combined", "--pred", str(braid), str(unbraid),
           "--metrics", METRICS_BY_VERSION[version], "--out", str(out_prefix),
           "--tag", f"{mkey}_{version}_s{seed}"]
    print(f"  eval: {mkey}_{version} seed{seed}")
    subprocess.run(cmd, check=True)
    return summ if summ.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root",  type=Path, default=ROOT / "outputs/crossid")
    p.add_argument("--eval-root", type=Path, default=ROOT / "eval_results/crossid")
    p.add_argument("--seeds",     nargs="+", type=int, default=[42, 1, 2])
    p.add_argument("--mcs",       nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--force",     action="store_true", help="기존 seed summary 재계산")
    args = p.parse_args()

    args.eval_root.mkdir(parents=True, exist_ok=True)
    models = [(f"mcs{n}", dict(MODELS)[f"mcs{n}"]) for n in args.mcs]

    agg_rows = []
    for mkey, mlabel in models:
        for ver in VERSIONS:
            # seed별 summary 수집
            per_seed = []
            for s in args.seeds:
                summ = run_eval_one(mkey, ver, s, args.out_root, args.eval_root, args.force)
                if summ:
                    per_seed.append(read_summary(summ))
            if not per_seed:
                continue
            row = {"model": mlabel, "version": ver, "n_seed": len(per_seed)}
            for label, col in REPORT:
                vals = []
                for sdict in per_seed:
                    v = sdict.get(label, {}).get(col)
                    if v not in (None, "", "N/A", "—"):
                        try:
                            vals.append(float(v))
                        except ValueError:
                            pass
                if vals:
                    row[f"{label}|mean"] = float(np.mean(vals))
                    row[f"{label}|std"]  = float(np.std(vals))
                else:
                    row[f"{label}|mean"] = row[f"{label}|std"] = None
            agg_rows.append(row)

    # 콘솔 표
    print(f"\n{'='*100}\n  Cross-ID 3-seed 집계 (mean ± std)\n{'='*100}")
    hdr = f"{'model':<18}{'version':<16}"
    for label, _ in REPORT:
        hdr += f"{label:>18}"
    print(hdr + f"{'seed':>6}")
    print("-" * 112)
    for r in agg_rows:
        line = f"{r['model']:<18}{r['version']:<16}"
        for label, _ in REPORT:
            m, sd = r.get(f"{label}|mean"), r.get(f"{label}|std")
            line += f"{(f'{m:.3f}±{sd:.3f}' if m is not None else 'N/A'):>18}"
        print(line + f"{r['n_seed']:>6}")

    # CSV
    out_csv = args.eval_root / "summary_3seed.csv"
    fields = ["model", "version", "n_seed"] + \
             [f"{label}|{k}" for label, _ in REPORT for k in ("mean", "std")]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(agg_rows)
    print(f"\ncsv: {out_csv}")


if __name__ == "__main__":
    main()
