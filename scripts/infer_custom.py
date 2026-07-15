"""
임의 스케치/matte 파일로 hair region 추론 (학습 데이터 불필요)

Usage:
  # 단일 파일
  python scripts/infer_custom.py \
    --sketch    custom/my_sketch.png \
    --matte     custom/my_matte.png \
    --checkpoint checkpoints/mcs1_phase2/final.pth \
    --config    configs/mcs1_phase2.yaml \
    --output_dir custom_results/

  # face 이미지 포함 → latent 합성 full image 출력
  python scripts/infer_custom.py \
    --sketch    custom/my_sketch.png \
    --matte     custom/my_matte.png \
    --face      custom/my_face.png \
    --checkpoint checkpoints/mcs1_phase2/final.pth \
    --config    configs/mcs1_phase2.yaml \
    --output_dir custom_results/

  # 폴더 일괄 처리 (sketch/*.png 와 matte/*.png 파일명 매칭)
  python scripts/infer_custom.py \
    --sketch    custom/sketches/ \
    --matte     custom/mattes/ \
    --face      custom/faces/ \
    --checkpoint checkpoints/mcs1_phase2/final.pth \
    --config    configs/mcs1_phase2.yaml \
    --output_dir custom_results/

출력 (face 없을 때):  {name}.png  — 생성된 hair patch
출력 (face 있을 때):  {name}.png  — face + hair 합성 full image
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from torchvision import transforms

from src.models.controlnet_sd35 import HairControlNet, gate_block_samples
from src.models.vae_wrapper import VAEWrapper


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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
        # base 경로는 cwd(프로젝트 루트) 기준. 단, 없으면 config_path 옆에서 fallback.
        bp = Path(base_path)
        if not bp.exists():
            bp = Path(config_path).parent / base_path
        base_cfg = load_config(str(bp))
        cfg = deep_merge(base_cfg, cfg)
    return cfg


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

_to_tensor_512 = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


def load_sketch(path: Path) -> torch.Tensor:
    """Returns (1, 3, 512, 512) float [0,1]."""
    return _to_tensor_512(Image.open(path).convert("RGB")).unsqueeze(0)


def load_matte(path: Path) -> torch.Tensor:
    """Returns (1, 1, 512, 512) float [0,1]."""
    return _to_tensor_512(Image.open(path).convert("L")).unsqueeze(0)


def load_face(path: Path) -> torch.Tensor:
    """Returns (1, 3, 512, 512) float [0,1]."""
    return _to_tensor_512(Image.open(path).convert("RGB")).unsqueeze(0)


def sketch_to_matte(sketch: torch.Tensor) -> torch.Tensor:
    """스케치 비-배경 영역(비검정)을 matte로 사용. (1,1,512,512)"""
    fg = (sketch.squeeze(0).max(dim=0).values > 0.05).float()
    return fg.unsqueeze(0).unsqueeze(0)


def recolor_sketch(sketch: torch.Tensor, hair_color: tuple[int, int, int]) -> torch.Tensor:
    """스케치 stroke 색을 지정 색으로 교체. (1,3,512,512)"""
    color = torch.tensor([c / 255.0 for c in hair_color], dtype=sketch.dtype)
    s = sketch.squeeze(0)
    brightness = s.mean(dim=0)
    is_stroke = (brightness > 0.05) & (brightness < 0.95)
    result = s.clone()
    result[:, is_stroke] = color.unsqueeze(1).expand(-1, is_stroke.sum())
    return result.unsqueeze(0)


def recolor_sketch_from_gt(
    sketch: torch.Tensor,
    gt_img: torch.Tensor,
    matte: torch.Tensor,
    min_pixels: int = 10,
    quantize_bits: int | None = None,
) -> torch.Tensor:
    """GT 헤어색으로 stroke를 칠한다 (flat-color raw uint8 정책, deterministic mean only).

    각 exact RGB stroke group에 대해 matte 내부 GT 이미지 평균색을 계산해
    matte 내부만 칠하고, matte 바깥은 검정으로 지운다. valid 픽셀이
    min_pixels 미만인 group은 전체를 검정으로 제거한다.

    Args:
        sketch: (1,3,512,512) [0,1]
        gt_img: (1,3,512,512) [0,1] — GT 원본 이미지 (test set은 face와 동일)
        matte:  (1,1,512,512) [0,1]
    """
    _ = quantize_bits  # Backward-compatible argument; exact RGB grouping ignores quantization.
    s = sketch.squeeze(0)                          # (3,H,W)
    sketch_u8 = (s * 255).round().to(torch.uint8)
    img_u8 = (gt_img.squeeze(0) * 255).round().to(torch.uint8)
    matte_u8 = (matte.squeeze(0) * 255).round().to(torch.uint8)

    flat = sketch_u8.view(3, -1).T
    unique_colors = torch.unique(flat, dim=0)
    out = s.clone()
    matte_mask = matte_u8[0] > 0
    floor_color = torch.tensor([7.0, 7.0, 10.0], dtype=torch.float32, device=s.device) / 255.0
    floor_threshold = floor_color.max().item()

    for color in unique_colors:
        r, g, b = color.tolist()
        if r == 0 and g == 0 and b == 0:
            continue

        group_mask = (
            (sketch_u8[0] == r) &
            (sketch_u8[1] == g) &
            (sketch_u8[2] == b)
        )
        valid_mask = group_mask & matte_mask
        outside_mask = group_mask & (~matte_mask)

        if valid_mask.sum().item() < min_pixels:
            out[:, group_mask] = 0.0
            continue

        hair_pixels = img_u8[:, valid_mask].float() / 255.0
        mean_color = hair_pixels.mean(dim=1)
        if mean_color.max().item() < floor_threshold:
            mean_color = floor_color.to(mean_color)

        out[:, valid_mask] = mean_color.unsqueeze(1)
        out[:, outside_mask] = 0.0

    return out.unsqueeze(0)


def to_pil(t: torch.Tensor) -> Image.Image:
    """(1,3,H,W) float [0,1] → PIL Image."""
    arr = t.squeeze(0).float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype("uint8"))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sampling(
    controlnet, transformer, scheduler, schedule,
    sketch, matte, num_steps, device,
    vae=None, face=None, bld_mode="full", bld_soft_steps=None,
    gate_alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Euler sampling → (hair_latent, face_latent | None).

    bld_mode (face 있을 때만 의미):
      full  — 매 스텝 matte 바깥을 face noised latent로 블렌딩 (최종 하드 pin 없음).
      final — 매 스텝 블렌딩 없이, 마지막 decode 직전 단 한 번만 latent에서 원본
              img(face)로 matte 바깥을 합성 (디코더 직전 합성).
      off   — 배경 블렌딩/합성 전부 없음 (face 무시, 생성 latent 그대로).
    face_latent(x0_bg)는 인코딩한 배경 latent이며 composite_full에서 재사용된다.
    """
    scheduler.set_timesteps(num_steps, device=device)

    noise   = torch.randn(1, 16, 64, 64, device=device, dtype=torch.bfloat16)
    latents = noise.clone()

    sketch_bf = sketch.to(device=device, dtype=torch.bfloat16)
    matte_bf  = matte.to(device=device, dtype=torch.bfloat16)

    # 배경 합성(BLD) 사전 준비. do_bg=최종 합성 여부, per_step=매 스텝 블렌딩 여부.
    do_bg    = face is not None and vae is not None and bld_mode != "off"
    per_step = do_bg and bld_mode == "full"
    if do_bg:
        x0_bg    = vae.encode(face.to(device=device, dtype=torch.bfloat16))  # (1,16,64,64)
        mask_lat = F.interpolate(matte_bf, size=(64, 64), mode="area")       # (1,1,64,64)

    for i, t in enumerate(tqdm(scheduler.timesteps, desc="steps", leave=False)):
        sigmas_1d = scheduler.sigmas[i].to(device=device, dtype=torch.bfloat16).view(1)

        # BLD(full): matte 바깥을 face의 noised latent로 블렌딩.
        #   bld_soft_steps — 순수 배경(mask==0)은 매 스텝 계속, soft 경계(0<mask<1)는
        #                    앞 N스텝만 블렌딩(이후 경계는 자유 생성).
        if per_step:
            noised_bg = (1.0 - sigmas_1d) * x0_bg + sigmas_1d * noise
            m = mask_lat
            if bld_soft_steps is not None and i >= bld_soft_steps:
                m = (mask_lat > 1e-4).to(mask_lat.dtype)   # 경계는 keep(1), 순수배경만 복원
            latents = m * latents + (1.0 - m) * noised_bg

        block_samples, null_enc_hs, null_pooled = controlnet(
            noisy_latent=latents,
            sketch=sketch_bf,
            matte=matte_bf,
            sigmas=sigmas_1d,
        )
        block_samples = [s.to(dtype=torch.bfloat16) for s in block_samples]

        if schedule != "none":
            block_samples = gate_block_samples(block_samples, matte_bf, schedule, gate_alpha=gate_alpha)

        v_pred = transformer(
            hidden_states=latents,
            encoder_hidden_states=null_enc_hs.to(dtype=torch.bfloat16),
            pooled_projections=null_pooled.to(dtype=torch.bfloat16),
            timestep=sigmas_1d,
            block_controlnet_hidden_states=block_samples,
            return_dict=False,
        )[0]

        latents = scheduler.step(v_pred, t, latents, return_dict=False)[0]

    # final 모드에서만: 디코더 직전 단 한 번 matte 바깥을 노이즈 없는 source latent
    # x0_bg로 합성. full 모드는 매스텝 블렌딩만 하고 이 하드 고정은 하지 않는다.
    if do_bg and bld_mode == "final":
        latents = mask_lat * latents + (1.0 - mask_lat) * x0_bg

    return latents, (x0_bg if do_bg else None)


@torch.no_grad()
def decode_hair(vae, hair_latent: torch.Tensor) -> torch.Tensor:
    """hair latent → (1,3,512,512) [0,1]."""
    image = vae.decode(hair_latent)
    return (image.float().clamp(-1, 1) + 1) / 2


@torch.no_grad()
def composite_full(
    vae,
    hair_latent: torch.Tensor,
    matte: torch.Tensor,
    face: torch.Tensor,
    device: torch.device,
    face_latent: torch.Tensor | None = None,
) -> torch.Tensor:
    """VAE decode 만 수행. Decoder 직전 latent BLD / Decode 직후 pixel BLD 모두 적용 안 함."""
    return (vae.decode(hair_latent).float().clamp(-1, 1) + 1) / 2


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def collect_pairs(
    sketch_arg: str,
    matte_arg: str | None,
    face_arg: str | None,
) -> list[tuple[Path, Path | None, Path | None, str]]:
    sketch_path = Path(sketch_arg)
    if sketch_path.is_dir():
        files = sorted(sketch_path.glob("*_sketch.png")) or \
                sorted(sketch_path.glob("*.png")) + sorted(sketch_path.glob("*.jpg"))
        pairs = []
        for sf in files:
            bare = sf.stem.replace("_sketch", "")
            mf = ff = None
            if matte_arg:
                mp = Path(matte_arg)
                for ext in (".png", ".jpg"):
                    for cand_stem in (bare + "_matte", bare, sf.stem):
                        c = mp / (cand_stem + ext)
                        if c.exists():
                            mf = c
                            break
                    if mf:
                        break
            if face_arg:
                fp = Path(face_arg)
                for ext in (".png", ".jpg"):
                    for cand_stem in (bare, sf.stem):
                        c = fp / (cand_stem + ext)
                        if c.exists():
                            ff = c
                            break
                    if ff:
                        break
            pairs.append((sf, mf, ff, bare))
        return pairs
    else:
        mf = Path(matte_arg) if matte_arg else None
        ff = Path(face_arg)  if face_arg  else None
        bare = sketch_path.stem.replace("_sketch", "")
        return [(sketch_path, mf, ff, bare)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sketch",      required=True)
    parser.add_argument("--matte",       default=None)
    parser.add_argument("--face",        default=None,
                        help="얼굴 이미지 (있으면 latent 합성 full image 출력)")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--num_steps",   type=int, default=20)
    parser.add_argument("--output_dir",  default="custom_results")
    parser.add_argument("--hair_color",  default=None,
                        help="stroke 색 교체 (R,G,B 0-255). 예: 139,90,43")
    parser.add_argument("--recolor_from_gt", action="store_true",
                        help="GT 이미지(=matte 영역 평균색)로 stroke 재착색 (SHS 방식). "
                             "matte와 face(GT img) 필요. --hair_color보다 우선.")
    parser.add_argument("--bld_mode",      default="full", choices=["full", "final", "off"],
                        help="배경 합성 방식 (face 있을 때만). full=매스텝 블렌딩만, "
                             "final=디코더 직전 1회만 원본 img 합성, off=합성 안 함")
    parser.add_argument("--bld_soft_steps", type=int, default=None,
                        help="순수 배경(mask==0)은 매 스텝, soft 경계(0<mask<1)만 앞 N 스텝만 "
                             "블렌딩 (이후 경계는 자유 생성).")
    parser.add_argument("--device",        default=None,
                        help="cuda / cpu (기본: cuda if available)")
    parser.add_argument("--seed",          type=int, default=None,
                        help="고정 시 매 이미지마다 동일한 noise 사용 (A/B 같은 controlled compare용)")
    parser.add_argument("--throttle",      type=float, default=0.0,
                        help="이미지마다 N초 sleep — GPU를 중간중간 쉬게 해 in-run 발열/전력 피크 완화 "
                             "(WSL에서 전력제한 불가할 때 소프트웨어 스로틀). 예: 1.0")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model_id         = cfg["model"]["model_id"]
    local_files_only = cfg.get("local_files_only", False)
    schedule         = cfg["training"].get("schedule", "none")
    gate_alpha       = cfg["training"].get("gate_alpha", 1.0)   # PDF alpha gate (Eq. 9)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading VAE...")
    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=torch.bfloat16, local_files_only=local_files_only,
    ).to(device).eval()

    print("Loading Transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    ).to(device).eval()

    print("Loading ControlNet...")
    controlnet = HairControlNet(
        model_id=model_id,
        vae=vae,
        num_layers=cfg["model"].get("num_controlnet_layers", 12),
        local_files_only=local_files_only,
        zero_matte_cond=cfg["model"].get("zero_matte_cond", False),
        zero_matte_feat=cfg["model"].get("zero_matte_feat", False),
        zero_raw_matte=cfg["model"].get("zero_raw_matte", False),
        num_extra_conditioning_channels=cfg["model"].get("num_extra_conditioning_channels", 16),
        matte_bias_zero_init=cfg["model"].get("matte_bias_zero_init", True),
        use_matte_scale=cfg["model"].get("use_matte_scale", True),
    )
    ckpt_path = Path(args.checkpoint)
    infer_path = ckpt_path.with_name(ckpt_path.stem + "_infer.pth")
    load_path = infer_path if infer_path.exists() else ckpt_path
    if load_path == ckpt_path:
        print(f"[WARNING] infer checkpoint not found ({infer_path.name}), loading full checkpoint (requires large RAM)")
    ckpt = torch.load(load_path, map_location="cpu", weights_only=True)
    state = ckpt["controlnet"] if "controlnet" in ckpt else ckpt
    controlnet.load_state_dict(state)
    del ckpt
    controlnet = controlnet.to(device=device, dtype=torch.bfloat16).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", local_files_only=local_files_only,
    )

    pairs = collect_pairs(args.sketch, args.matte, args.face)
    print(f"\n{len(pairs)}개 스케치 처리 (schedule={schedule})\n")

    hair_color = None
    if args.hair_color:
        hair_color = tuple(int(x) for x in args.hair_color.split(","))
        print(f"Stroke 색 교체: RGB{hair_color}")

    for sketch_file, matte_file, face_file, stem in tqdm(pairs, desc="Generating"):
        if args.seed is not None:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        sketch = load_sketch(sketch_file)

        if matte_file and matte_file.exists():
            matte = load_matte(matte_file)
        else:
            if matte_file:
                print(f"  [WARNING] matte 없음: {matte_file} → 스케치 fg 영역으로 대체")
            matte = sketch_to_matte(sketch)

        face = load_face(face_file) if face_file and face_file.exists() else None

        if args.recolor_from_gt:
            if face is None:
                print(f"  [WARNING] {stem}: GT(face) 없음 → recolor 생략, 원본 sketch 사용")
            else:
                sketch = recolor_sketch_from_gt(sketch, face, matte)
        elif hair_color:
            sketch = recolor_sketch(sketch, hair_color)
        hair_latent, face_latent_bld = run_sampling(
            controlnet, transformer, scheduler, schedule,
            sketch, matte, args.num_steps, device,
            vae=vae, face=face, bld_mode=args.bld_mode,
            bld_soft_steps=args.bld_soft_steps,
            gate_alpha=gate_alpha,
        )

        if face is not None:
            result = composite_full(vae, hair_latent, matte, face, device, face_latent=face_latent_bld)
        else:
            if face_file:
                print(f"  [WARNING] face 없음: {face_file} → hair patch만 저장")
            result = decode_hair(vae, hair_latent)

        to_pil(result.cpu()).save(output_dir / f"{stem}.png")

        if args.throttle > 0:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            time.sleep(args.throttle)

    print(f"\n완료: {output_dir}/")


if __name__ == "__main__":
    main()
