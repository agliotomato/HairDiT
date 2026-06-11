"""
Split-aware evaluation script.

Usage:
  python3 scripts/eval_metrics.py \\
      --split   braid                          \\
      --pred    custom_results/dit/exp1        \\
      [--pred-suffix _full]                    \\
      [--out    eval_results/exp1_braid]       \\
      [--tag    "Exp1"]

Dataset paths are derived automatically from the split:
  dataset/{split}/img/test    — GT images
  dataset/{split}/matte/test  — GT mattes (grayscale)
  dataset/{split}/sketch/test — sketch images

Metrics (최종 7개 — 스펙 확정):
  per-image (braid/unbraid/macro 분리 보고, braid는 ±CI95):
    ① Sketch LPIPS ↓  (구조,     paired vs sketch, hair 영역)
    ② Edge IoU ↑      (구조 보조, paired vs sketch, hair 영역)
    ④ LPIPS (GT) ↓    (외형,     paired vs GT,     hair-masked)
    ⑥ Boundary LPIPS ↓(경계 보조, paired vs GT,     경계 band)
  FID (통합 573만 보고, dims=2048):
    ③ Hair FID ↓          (리얼리즘,   hair 마스킹/크롭)
    ⑤ Boundary FID ↓      (경계,       경계 band)
    ⑦ Full-portrait FID ↓ (전역 조화,  whole-image)

  보고 단위 규칙:
    - FID(③⑤⑦)는 2048-dim 공분산 → n=107 braid 단독은 rank-deficient. 통합(573)만 보고.
    - per-image(①②④⑥)는 unbraid/braid/macro 분리. braid는 ±CI95.

Outputs:
  <out>_per_image.csv  (stem, split, ①②④⑥)
  <out>_summary.csv    (metric, axis, paired, unit, braid, braid_ci95, unbraid, macro, combined573)
  console table
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_METRICS = frozenset({
    "sketch_lpips", "sketch_delta_e", "edge_iou",  # vs sketch (구조/색상)
    "lpips_gt", "bnd_lpips",                        # vs GT (외형/경계)
    "delta_e_gt", "psnr",                           # vs GT (색상/화질)
    "hair_fid", "bnd_fid", "full_fid",             # FID (리얼리즘)
})

SPLITS = {
    "braid": {
        "img":    ROOT / "dataset/braid/img/test",
        "matte":  ROOT / "dataset/braid/matte/test",
        "sketch": ROOT / "dataset/braid/sketch/test",
    },
    "unbraid": {
        "img":    ROOT / "dataset/unbraid/img/test",
        "matte":  ROOT / "dataset/unbraid/matte/test",
        "sketch": ROOT / "dataset/unbraid/sketch/test",
    },
    "combined": None,  # braid + unbraid 합산 — --pred braid_dir unbraid_dir 순서로 2개 지정
}


# ---------------------------------------------------------------------------
# LPIPS
# NOTE: compute_lpips는 항상 RGB 3채널 컬러 입력을 받는다.
#   - lpips_gt, bnd_lpips: pred/GT 모두 full-color RGB crop → 컬러 비교 ✓
#   - sketch_lpips: pred 측은 Canny edge map (grayscale→3ch 복제)으로 변환 후 비교.
#     구조(선 패턴) 유사도가 목적이므로 컬러 비교가 아니다.
#     컬러 충실도 비교는 sketch_delta_e(ΔE) 지표가 담당한다.
# ---------------------------------------------------------------------------
_lpips_fn = None


def get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
    return _lpips_fn


def _to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(arr).float().permute(2, 0, 1) / 127.5 - 1.0
    return t.unsqueeze(0).to(DEVICE)


def compute_lpips(a: np.ndarray, b: np.ndarray) -> float:
    fn = get_lpips()
    with torch.no_grad():
        ta, tb = _to_lpips_tensor(a), _to_lpips_tensor(b)
        if ta.shape[-1] < 64 or ta.shape[-2] < 64:
            ta = F.interpolate(ta, (64, 64), mode="bilinear", align_corners=False)
            tb = F.interpolate(tb, (64, 64), mode="bilinear", align_corners=False)
        return float(fn(ta, tb).item())


# ---------------------------------------------------------------------------
# Pixel metrics
# ---------------------------------------------------------------------------

def compute_psnr(a: np.ndarray, b: np.ndarray,
                 alpha: np.ndarray | None = None) -> float:
    """MSE → PSNR. alpha (H,W) float [0,1]: alpha-weighted MSE."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    if alpha is None:
        err = float(np.mean((a64 - b64) ** 2))
    else:
        aw = alpha[..., None].astype(np.float64)   # (H,W,1) broadcast over C
        w  = float(aw.sum()) * (a64.shape[-1] if a64.ndim == 3 else 1)
        if w < 1e-6:
            return float("nan")
        err = float(((a64 - b64) ** 2 * aw).sum() / w)
    return 10 * math.log10(255.0 ** 2 / err) if err > 0 else float("inf")


def _rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """RGB uint8 → CIE Lab float64 (L∈[0,100], a,b∈[-128,127])."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)
    lab[:, :, 0] = lab[:, :, 0] * 100.0 / 255.0
    lab[:, :, 1] -= 128.0
    lab[:, :, 2] -= 128.0
    return lab


def compute_delta_e_hue(a: np.ndarray, b: np.ndarray, mask: np.ndarray,
                         alpha: np.ndarray | None = None) -> float:
    """CIEDE2000 between shade-normalized mean Lab colors within mask.

    Shade normalization: L is fixed to 50 before averaging so the result
    reflects hue/chroma difference only, not lighting variation.
    alpha (H,W) float [0,1]: soft-alpha weights for the mean.
    """
    from skimage.color import deltaE_ciede2000
    pixels = mask > 0
    if pixels.sum() < 10:
        return float("nan")
    lab_a = _rgb_to_lab(a)[pixels].copy()   # (N, 3)
    lab_b = _rgb_to_lab(b)[pixels].copy()
    lab_a[:, 0] = 50.0                       # shade normalization
    lab_b[:, 0] = 50.0
    if alpha is not None:
        w = alpha[pixels].astype(np.float64)
        w_sum = w.sum()
        if w_sum > 1e-8:
            mean_a = (lab_a * w[:, None]).sum(0) / w_sum
            mean_b = (lab_b * w[:, None]).sum(0) / w_sum
        else:
            mean_a, mean_b = lab_a.mean(0), lab_b.mean(0)
    else:
        mean_a, mean_b = lab_a.mean(axis=0), lab_b.mean(axis=0)
    return float(deltaE_ciede2000(mean_a, mean_b))


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _alpha_blend_crop(img: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """img * alpha[...,None], returns uint8. alpha ∈ [0,1] float."""
    return (img.astype(np.float64) * alpha[..., None]).clip(0, 255).astype(np.uint8)


def extract_region_crop(img: np.ndarray, matte: np.ndarray, min_px: int = 64):
    hair = matte > 0
    if hair.sum() < min_px:
        return None
    y0, y1, x0, x1 = _bbox(hair)
    crop    = img  [y0:y1, x0:x1].copy()
    alpha_c = matte[y0:y1, x0:x1].astype(np.float64) / 255.0
    crop    = _alpha_blend_crop(crop, alpha_c)
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return None
    return cv2.resize(crop, (128, 128))


def get_boundary_mask(matte: np.ndarray) -> np.ndarray:
    return (matte >= 25) & (matte <= 230)


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def canny_edges(img: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 50, 150) > 0


def edge_iou(pred_e: np.ndarray, sk_e: np.ndarray, matte: np.ndarray) -> float:
    hair = matte > 0
    a, b = pred_e & hair, sk_e & hair
    union = (a | b).sum()
    return float((a & b).sum() / union) if union > 0 else 0.0


def sketch_lpips(pred: np.ndarray, sketch: np.ndarray, matte: np.ndarray) -> float:
    hair = matte > 0
    if hair.sum() == 0:
        return float("nan")
    y0, y1, x0, x1 = _bbox(hair)
    alpha_c  = matte[y0:y1, x0:x1].astype(np.float64) / 255.0
    edge_rgb = np.stack([canny_edges(pred).astype(np.uint8) * 255] * 3, axis=-1)
    p_crop   = _alpha_blend_crop(edge_rgb[y0:y1, x0:x1].copy(), alpha_c)
    s_crop   = _alpha_blend_crop(sketch  [y0:y1, x0:x1].copy(), alpha_c)
    if p_crop.shape[0] < 8 or p_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(p_crop, s_crop)


def hair_masked_lpips(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    hair = matte > 0
    if hair.sum() < 64:
        return float("nan")
    y0, y1, x0, x1 = _bbox(hair)
    alpha_c = matte[y0:y1, x0:x1].astype(np.float64) / 255.0
    p_crop  = _alpha_blend_crop(pred[y0:y1, x0:x1].copy(), alpha_c)
    g_crop  = _alpha_blend_crop(gt  [y0:y1, x0:x1].copy(), alpha_c)
    if p_crop.shape[0] < 8 or p_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(p_crop, g_crop)


def boundary_lpips(pred: np.ndarray, gt: np.ndarray, matte: np.ndarray) -> float:
    bnd = get_boundary_mask(matte)
    if bnd.sum() < 64:
        return float("nan")
    y0, y1, x0, x1 = _bbox(bnd)
    bnd_alpha = np.where(bnd, matte.astype(np.float64) / 255.0, 0.0)
    alpha_c   = bnd_alpha[y0:y1, x0:x1]
    p_crop = _alpha_blend_crop(pred[y0:y1, x0:x1].copy(), alpha_c)
    g_crop = _alpha_blend_crop(gt  [y0:y1, x0:x1].copy(), alpha_c)
    if p_crop.shape[0] < 8 or p_crop.shape[1] < 8:
        return float("nan")
    return compute_lpips(p_crop, g_crop)


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(real_imgs: list, fake_imgs: list, dims: int = 2048) -> float:
    """표준 InceptionV3 FID. dims=2048 (pool3 2048-dim 공분산).

    주의: 2048-dim 공분산은 표본 수 n < 2048 이면 rank-deficient → 통합(573)에서만
    안정적. braid 단독(n=107) 분리 보고 금지 (스펙 보고 단위 규칙).
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        print("[WARN] pytorch_fid 없음 — FID는 NaN. pip install pytorch-fid")
        return float("nan")
    with tempfile.TemporaryDirectory() as tmp:
        real_dir, fake_dir = Path(tmp) / "real", Path(tmp) / "fake"
        real_dir.mkdir()
        fake_dir.mkdir()
        for i, img in enumerate(real_imgs):
            if img is not None:
                Image.fromarray(img).save(real_dir / f"{i:05d}.png")
        for i, img in enumerate(fake_imgs):
            if img is not None:
                Image.fromarray(img).save(fake_dir / f"{i:05d}.png")
        nr = len(list(real_dir.glob("*.png")))
        nf = len(list(fake_dir.glob("*.png")))
        if nr < 2 or nf < 2:
            return float("nan")
        return float(fid_score.calculate_fid_given_paths(
            [str(real_dir), str(fake_dir)],
            batch_size=32, device=str(DEVICE), dims=dims, num_workers=0,
        ))


# ---------------------------------------------------------------------------
# Summary / output
# ---------------------------------------------------------------------------

# 최종 정량 지표 7개 (스펙 확정).
#   (label, per_image_key, fid_key, higher_better, axis, paired, unit)
#     unit = "split"    → per-image. braid/unbraid/macro 분리 보고 (braid는 ±CI)
#     unit = "combined" → FID. 통합(573)만 보고 (2048-dim 공분산, braid n=107 rank-deficient)
SPEC_METRICS = [
    ("Sketch LPIPS ↓",      "sketch_lpips",   None,       False, "구조",        "paired(vs sketch)", "split"),
    ("Sketch ΔE2000 ↓",      "sketch_delta_e", None,       False, "색상(vs sk)", "paired(vs sketch)", "split"),
    ("Edge IoU ↑",          "edge_iou",       None,       True,  "구조(보조)",  "paired(vs sketch)", "split"),
    ("Hair FID ↓",          None,             "hair_fid", False, "리얼리즘",    "unpaired",          "combined"),
    ("LPIPS (GT) ↓",        "lpips",          None,       False, "외형",        "paired(vs GT)",     "split"),
    ("ΔE2000 (GT, hair) ↓",  "delta_e_gt",     None,       False, "색상(vs GT)", "paired(vs GT)",     "split"),
    ("PSNR (hair) ↑",       "psnr",           None,       True,  "화질",        "paired(vs GT)",     "split"),
    ("Boundary FID ↓",      None,             "bnd_fid",  False, "경계",        "unpaired",          "combined"),
    ("Boundary LPIPS ↓",    "bnd_lpips",      None,       False, "경계(보조)",  "paired(vs GT)",     "split"),
    ("Full-portrait FID ↓", None,             "full_fid", False, "전역 조화",   "unpaired",          "combined"),
]

PER_IMAGE_KEYS = ["sketch_lpips", "sketch_delta_e", "edge_iou", "lpips", "bnd_lpips", "delta_e_gt", "psnr"]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.4f}"


def _safe_mean(vals):
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(vs) / len(vs) if vs else None


def _ci95(vals):
    """95% 신뢰구간 half-width (1.96·SD/√n). 표본<2면 None."""
    vs = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(vs)
    if n < 2:
        return None
    m = sum(vs) / n
    var = sum((v - m) ** 2 for v in vs) / (n - 1)
    return 1.96 * math.sqrt(var) / math.sqrt(n)


def build_summary(rows_braid, rows_unbraid, fid: dict) -> list[dict]:
    """SPEC_METRICS 각 지표를 보고단위별로 정리.
    combined 모드: rows_braid, rows_unbraid 둘 다 지정. single split: 한쪽만(나머지 None)."""
    out = []
    for label, pk, fk, higher, axis, paired, unit in SPEC_METRICS:
        r = {"label": label, "axis": axis, "paired": paired, "unit": unit,
             "braid": None, "braid_ci95": None, "unbraid": None, "unbraid_ci95": None,
             "macro": None, "macro_ci95": None, "combined": None}
        if unit == "combined":
            r["combined"] = fid.get(fk)
        else:
            if rows_braid is not None:
                r["braid"]      = _safe_mean([x.get(pk) for x in rows_braid])
                r["braid_ci95"] = _ci95([x.get(pk) for x in rows_braid])
            if rows_unbraid is not None:
                r["unbraid"]      = _safe_mean([x.get(pk) for x in rows_unbraid])
                r["unbraid_ci95"] = _ci95([x.get(pk) for x in rows_unbraid])
            if r["braid"] is not None and r["unbraid"] is not None:
                r["macro"] = (r["braid"] + r["unbraid"]) / 2
                # 독립 두 그룹 평균의 평균 → CI = ½·√(CI_b² + CI_u²)
                cb, cu = r["braid_ci95"], r["unbraid_ci95"]
                if cb is not None and cu is not None:
                    r["macro_ci95"] = 0.5 * math.sqrt(cb ** 2 + cu ** 2)
        out.append(r)
    return out


def _fmt_ci(v, ci) -> str:
    s = _fmt(v)
    return f"{s}±{ci:.4f}" if (v is not None and ci is not None) else s


def print_summary(summary: list[dict], tag: str, n_b: int, n_u: int):
    print(f"\n{'='*80}")
    print(f"  tag={tag}   braid n={n_b}   unbraid n={n_u}   combined N={n_b + n_u}")
    print("=" * 80)
    print(f"  {'Metric':<18}{'축':<10}{'braid(±CI95)':>22}{'unbraid(±CI95)':>22}{'macro(±CI95)':>22}{'comb573':>10}")
    print("-" * 100)
    for r in summary:
        if r["unit"] == "combined":
            braid = unbraid = macro = "—"
            comb = _fmt(r["combined"])
        else:
            braid   = _fmt_ci(r["braid"],   r["braid_ci95"])
            unbraid = _fmt_ci(r["unbraid"], r["unbraid_ci95"])
            macro   = _fmt_ci(r["macro"],   r["macro_ci95"])
            comb    = "—"
        print(f"  {r['label']:<18}{r['axis']:<10}{braid:>22}{unbraid:>22}{macro:>22}{comb:>10}")
    print("=" * 100)
    print("  · FID(③⑤⑦)=통합573만  · per-image(①②④⑥)=braid/unbraid/macro (모두 ±CI95)")
    print("  · run 간 유의차 확정은 scripts/stats_compare.py (paired t-test + Wilcoxon)")


def write_summary_csv(summary: list[dict], tag: str, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "axis", "paired", "unit",
                    "braid", "braid_ci95", "unbraid", "unbraid_ci95",
                    "macro", "macro_ci95", "combined573"])
        for r in summary:
            w.writerow([
                r["label"], r["axis"], r["paired"], r["unit"],
                _fmt(r["braid"]),   _fmt(r["braid_ci95"]),
                _fmt(r["unbraid"]), _fmt(r["unbraid_ci95"]),
                _fmt(r["macro"]),   _fmt(r["macro_ci95"]),
                _fmt(r["combined"]),
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_stems(pred_dir: Path, gt_dir: Path, suffix: str) -> list[str]:
    pred_stems = {
        (p.stem[:-len(suffix)] if suffix and p.stem.endswith(suffix) else p.stem)
        for p in pred_dir.glob("*.png")
    }
    gt_stems = {p.stem for p in gt_dir.glob("*.png")}
    common = pred_stems & gt_stems
    if not common:
        print("[ERROR] pred와 GT 사이에 공통 stem이 없습니다.")
        sys.exit(1)
    skipped = gt_stems - common
    if skipped:
        print(f"[INFO] GT에만 있어 제외된 stem: {len(skipped)}개")
    return sorted(common)


def _process_split(split_name: str, pred_dir: Path, suffix: str,
                   sketch_dir: Path | None = None,
                   metrics: frozenset = ALL_METRICS) -> dict:
    """단일 split 평가. per-image rows(split 라벨 포함) + FID용 이미지 리스트 dict 반환."""
    ds     = SPLITS[split_name]
    gt_dir = ds["img"]
    mt_dir = ds["matte"]
    sk_dir = sketch_dir if sketch_dir is not None else ds["sketch"]

    need_sketch = "sketch_lpips" in metrics or "edge_iou" in metrics or "sketch_delta_e" in metrics
    need_gt     = "lpips_gt" in metrics or "bnd_lpips" in metrics or "delta_e_gt" in metrics or "psnr" in metrics

    stems = discover_stems(pred_dir, gt_dir, suffix)
    print(f"  [{split_name}] {len(stems)}개  metrics={sorted(metrics)}")

    rows = []
    hair_r, hair_f = [], []
    bnd_r,  bnd_f  = [], []
    full_r, full_f = [], []

    for stem in tqdm(stems, desc=f"[{split_name}]"):
        pred_name = f"{stem}{suffix}.png" if suffix else f"{stem}.png"
        pred_path = pred_dir / pred_name

        gt    = np.array(Image.open(gt_dir / f"{stem}.png").convert("RGB"))
        matte = np.array(Image.open(mt_dir / f"{stem}.png").convert("L"))
        sk    = np.array(Image.open(sk_dir / f"{stem}.png").convert("RGB")) if need_sketch else None

        h, w = gt.shape[:2]
        if matte.shape[:2] != (h, w):
            matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
        if sk is not None and sk.shape[:2] != (h, w):
            sk = cv2.resize(sk, (w, h), interpolation=cv2.INTER_LINEAR)

        if not pred_path.exists():
            rows.append({"stem": stem, "split": split_name, **{k: None for k in PER_IMAGE_KEYS}})
            continue

        pred = np.array(Image.open(pred_path).convert("RGB"))
        if pred.shape[:2] != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)

        alpha     = matte.astype(np.float64) / 255.0
        hair      = matte > 0
        bnd       = get_boundary_mask(matte)
        bnd_matte = (matte * bnd.astype(np.uint8))   # boundary zone only, with matte values

        pred_e = canny_edges(pred) if need_sketch else None
        sk_e   = canny_edges(sk)   if need_sketch else None

        rows.append({
            "stem":           stem,
            "split":          split_name,
            "sketch_lpips":   sketch_lpips(pred, sk, matte)                                          if "sketch_lpips"   in metrics else None,
            "sketch_delta_e": compute_delta_e_hue(pred, sk, (sk.min(axis=-1) < 240) & hair, alpha)  if "sketch_delta_e" in metrics else None,
            "edge_iou":       edge_iou(pred_e, sk_e, matte)                                          if "edge_iou"       in metrics else None,
            "lpips":          hair_masked_lpips(pred, gt, matte)                                     if "lpips_gt"       in metrics else None,
            "bnd_lpips":      boundary_lpips(pred, gt, matte)                                        if "bnd_lpips"      in metrics else None,
            "delta_e_gt":     compute_delta_e_hue(pred, gt, hair, alpha)                             if "delta_e_gt"     in metrics else None,
            "psnr":           compute_psnr(pred, gt, alpha)                                          if "psnr"           in metrics else None,
        })

        if "hair_fid" in metrics:
            hair_r.append(extract_region_crop(gt,   matte))
            hair_f.append(extract_region_crop(pred, matte))
        if "bnd_fid" in metrics:
            bnd_r.append(extract_region_crop(gt,   bnd_matte, min_px=16))
            bnd_f.append(extract_region_crop(pred, bnd_matte, min_px=16))
        if "full_fid" in metrics:
            full_r.append(gt)
            full_f.append(pred)

    return {"rows": rows, "hair_r": hair_r, "hair_f": hair_f,
            "bnd_r": bnd_r, "bnd_f": bnd_f, "full_r": full_r, "full_f": full_f}


def _compute_fids(parts: dict, metrics: frozenset = ALL_METRICS) -> dict:
    """통합 FID 계산 (dims=2048). metrics에 포함된 FID만 계산."""
    fid_targets = [k for k in ("hair_fid", "bnd_fid", "full_fid") if k in metrics]
    if not fid_targets:
        return {"hair_fid": None, "bnd_fid": None, "full_fid": None}
    print("\nFID 계산 중 (통합, dims=2048)...")
    key_map = {
        "hair_fid": ("hair_r", "hair_f"),
        "bnd_fid":  ("bnd_r",  "bnd_f"),
        "full_fid": ("full_r", "full_f"),
    }
    fid = {"hair_fid": None, "bnd_fid": None, "full_fid": None}
    for k in fid_targets:
        rk, fk = key_map[k]
        fid[k] = compute_fid(
            [x for x in parts[rk] if x is not None],
            [x for x in parts[fk] if x is not None],
        )
        print(f"  {k}: {_fmt(fid[k])}")
    return fid


def _merge_parts(pb: dict, pu: dict) -> dict:
    return {k: pb[k] + pu[k] for k in ("hair_r", "hair_f", "bnd_r", "bnd_f", "full_r", "full_f")}


def _n_valid(rows):
    return sum(1 for r in rows if r.get("sketch_lpips") is not None)


def _write_per_image(rows, path: Path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "split"] + PER_IMAGE_KEYS)
        w.writeheader()
        w.writerows(rows)


def _save_and_report(summary, rows, tag, out, n_b, n_u):
    print_summary(summary, tag, n_b, n_u)
    per_path = Path(f"{out}_per_image.csv")
    sum_path = Path(f"{out}_summary.csv")
    _write_per_image(rows, per_path)
    write_summary_csv(summary, tag, sum_path)
    print(f"\n저장 완료:\n  {per_path}\n  {sum_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate braid/unbraid/combined split (7-metric spec).")
    parser.add_argument("--split",       required=True, choices=list(SPLITS.keys()),
                        help="'braid', 'unbraid', 'combined'")
    parser.add_argument("--pred",        required=True, nargs="+",
                        help="생성 이미지 디렉토리. combined는 'braid_dir unbraid_dir' 순서로 2개 지정")
    parser.add_argument("--pred-suffix", default="",
                        help="pred 파일명 접미사 (예: _full  →  stem_full.png)")
    parser.add_argument("--out",         default=None,
                        help="출력 경로 prefix (기본: eval_results/{split}_{pred_dir_name})")
    parser.add_argument("--tag",         default=None,
                        help="결과 테이블 컬럼 레이블")
    parser.add_argument("--sketch-dir",  default=None,
                        help="sketch 디렉토리 오버라이드 (기본: SPLITS[split]['sketch']). "
                             "combined 모드에서는 braid split에 적용.")
    parser.add_argument("--sketch-dir-unbraid", default=None,
                        help="unbraid split용 sketch 디렉토리 오버라이드. "
                             "combined 모드 전용. 미지정 시 --sketch-dir 값 사용.")
    parser.add_argument("--metrics",     default=None,
                        help="계산할 지표 (쉼표 구분). 기본: 전체. "
                             f"선택 가능: {','.join(sorted(ALL_METRICS))}")
    args = parser.parse_args()

    suffix             = args.pred_suffix
    sketch_dir         = Path(args.sketch_dir)         if args.sketch_dir         else None
    sketch_dir_unbraid = Path(args.sketch_dir_unbraid) if args.sketch_dir_unbraid else sketch_dir
    metrics            = frozenset(args.metrics.split(",")) if args.metrics else ALL_METRICS
    unknown            = metrics - ALL_METRICS
    if unknown:
        print(f"[ERROR] 알 수 없는 metric: {unknown}. 가능: {ALL_METRICS}")
        sys.exit(1)

    if args.split == "combined":
        if len(args.pred) == 1:
            braid_dir = unbraid_dir = Path(args.pred[0])
            if not braid_dir.exists():
                print(f"[ERROR] pred 디렉토리 없음: {braid_dir}")
                sys.exit(1)
        elif len(args.pred) == 2:
            braid_dir, unbraid_dir = Path(args.pred[0]), Path(args.pred[1])
            for p in (braid_dir, unbraid_dir):
                if not p.exists():
                    print(f"[ERROR] pred 디렉토리 없음: {p}")
                    sys.exit(1)
        else:
            print("[ERROR] --split combined은 --pred를 1개(단일 폴더) 또는 2개(braid unbraid) 지정하세요")
            sys.exit(1)

        tag = args.tag or braid_dir.name
        out = Path(args.out) if args.out else ROOT / "eval_results" / f"combined_{braid_dir.name}"
        out.parent.mkdir(parents=True, exist_ok=True)

        print("split  : combined (braid + unbraid)")
        print(f"pred   : {braid_dir}" + (f"  |  {unbraid_dir}" if braid_dir != unbraid_dir else " (단일 폴더, 자동 분리)"))
        print(f"suffix : '{suffix}'")
        print(f"out    : {out}_*\n")

        print("braid 처리 중...")
        pb = _process_split("braid",   braid_dir,   suffix, sketch_dir,            metrics)
        print("unbraid 처리 중...")
        pu = _process_split("unbraid", unbraid_dir, suffix, sketch_dir_unbraid,    metrics)

        fid = _compute_fids(_merge_parts(pb, pu), metrics)
        summary = build_summary(pb["rows"], pu["rows"], fid)
        _save_and_report(summary, pb["rows"] + pu["rows"], tag,
                         out, _n_valid(pb["rows"]), _n_valid(pu["rows"]))

    else:
        if len(args.pred) != 1:
            print("[ERROR] braid/unbraid split은 --pred를 1개만 지정하세요")
            sys.exit(1)
        pred_dir = Path(args.pred[0])
        if not pred_dir.exists():
            print(f"[ERROR] pred 디렉토리 없음: {pred_dir}")
            sys.exit(1)
        if args.split == "braid":
            print("[WARN] braid 단독 FID는 2048-dim에서 rank-deficient(n=107<2048). "
                  "FID는 통합(573, --split combined)으로 보고 권장.")

        tag = args.tag or pred_dir.name
        out = Path(args.out) if args.out else ROOT / "eval_results" / f"{args.split}_{pred_dir.name}"
        out.parent.mkdir(parents=True, exist_ok=True)

        ds = SPLITS[args.split]
        print(f"split  : {args.split}")
        print(f"pred   : {pred_dir}")
        print(f"suffix : '{suffix}'")
        print(f"gt     : {ds['img']}")
        print(f"out    : {out}_*\n")

        parts = _process_split(args.split, pred_dir, suffix, sketch_dir, metrics)
        fid = _compute_fids(parts, metrics)
        rb, ru = (parts["rows"], None) if args.split == "braid" else (None, parts["rows"])
        summary = build_summary(rb, ru, fid)
        n = _n_valid(parts["rows"])
        _save_and_report(summary, parts["rows"], tag, out,
                         n if args.split == "braid" else 0,
                         n if args.split == "unbraid" else 0)


if __name__ == "__main__":
    main()
