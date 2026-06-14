"""
Web demo for HairDiT — sketch + matte 브러싱 → 추론

Usage:
  python demo/server.py \\
    --checkpoint checkpoints/mcs1_phase2/final.pth \\
    --config     configs/mcs1_phase2.yaml \\
    --host 0.0.0.0 --port 7860
"""

import argparse
import io
import sys
import threading
from pathlib import Path

import torch
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel  # noqa: E402

from scripts.infer_custom import (  # noqa: E402
    composite_full,
    decode_hair,
    deep_merge,
    recolor_sketch,
    run_sampling,
    sketch_to_matte,
    to_pil,
)
from src.models.controlnet_sd35 import HairControlNet  # noqa: E402
from src.models.vae_wrapper import VAEWrapper  # noqa: E402


def load_config(config_path: str) -> dict:
    """Load YAML config with `base:` inheritance.

    Resolves `base:` against the config's parent dir first, then against the
    repo root — covers both conventions used in this codebase.
    """
    path = Path(config_path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = cfg.pop("base", None)
    if base:
        candidates = [path.parent / base, REPO_ROOT / base, Path(base)]
        resolved = next((c for c in candidates if c.exists()), None)
        if resolved is None:
            raise FileNotFoundError(
                f"base config not found: {base} (tried {[str(c) for c in candidates]})"
            )
        base_cfg = load_config(str(resolved))
        cfg = deep_merge(base_cfg, cfg)
    return cfg

STATIC_DIR = Path(__file__).parent / "static"

_STATE: dict = {}
_GPU_LOCK = threading.Lock()  # 동시 추론 직렬화 (단일 GPU)

_to_tensor_512 = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
])


def _tensor_from_bytes(data: bytes, mode: str) -> torch.Tensor:
    img = Image.open(io.BytesIO(data)).convert(mode)
    return _to_tensor_512(img).unsqueeze(0)


def _png_response(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_models(config_path: str, checkpoint_path: str, device: torch.device) -> dict:
    cfg = load_config(config_path)
    model_id = cfg["model"]["model_id"]
    local_files_only = cfg.get("local_files_only", False)
    schedule = cfg["training"].get("schedule", "none")

    print("[demo] Loading VAE...")
    vae = VAEWrapper.from_pretrained(
        model_id=model_id, torch_dtype=torch.bfloat16, local_files_only=local_files_only,
    ).to(device).eval()

    print("[demo] Loading Transformer...")
    transformer = SD3Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    ).to(device).eval()

    print("[demo] Loading ControlNet...")
    controlnet = HairControlNet(
        model_id=model_id, vae=vae,
        num_layers=cfg["model"].get("num_controlnet_layers", 12),
        local_files_only=local_files_only,
        zero_matte_cond=cfg["model"].get("zero_matte_cond", False),
        zero_matte_feat=cfg["model"].get("zero_matte_feat", False),
        zero_raw_matte=cfg["model"].get("zero_raw_matte", False),
    )
    ckpt_path = Path(checkpoint_path)
    infer_path = ckpt_path.with_name(ckpt_path.stem + "_infer.pth")
    load_path = infer_path if infer_path.exists() else ckpt_path
    if load_path == ckpt_path:
        print(f"[demo][WARN] infer checkpoint 없음 ({infer_path.name}) — full checkpoint 로드")
    ckpt = torch.load(load_path, map_location="cpu", weights_only=True)
    state = ckpt["controlnet"] if "controlnet" in ckpt else ckpt
    controlnet.load_state_dict(state)
    del ckpt
    controlnet = controlnet.to(device=device, dtype=torch.bfloat16).eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_id, subfolder="scheduler", local_files_only=local_files_only,
    )

    return {
        "vae": vae, "transformer": transformer, "controlnet": controlnet,
        "scheduler": scheduler, "schedule": schedule, "device": device,
        "model_id": model_id,
    }


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="HairDiT Demo")


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "ready": bool(_STATE),
        "device": str(_STATE.get("device", "")),
        "schedule": _STATE.get("schedule", ""),
        "model_id": _STATE.get("model_id", ""),
    }


@app.post("/api/sketch2matte")
async def api_sketch2matte(sketch: UploadFile = File(...)):
    sketch_t = _tensor_from_bytes(await sketch.read(), "RGB")
    matte_t = sketch_to_matte(sketch_t)  # (1,1,512,512), [0,1]
    arr = (matte_t.squeeze() * 255).clamp(0, 255).byte().cpu().numpy()
    return _png_response(Image.fromarray(arr, mode="L"))


@app.post("/api/infer")
async def api_infer(
    sketch: UploadFile = File(...),
    matte: UploadFile = File(...),
    face: UploadFile | None = File(None),
    num_steps: int = Form(20),
    composite: bool = Form(True),
    hair_color: str | None = Form(None),
):
    if not _STATE:
        return Response("Models not loaded yet", status_code=503)

    sketch_t = _tensor_from_bytes(await sketch.read(), "RGB")
    matte_t = _tensor_from_bytes(await matte.read(), "L")
    face_t = _tensor_from_bytes(await face.read(), "RGB") if face is not None else None

    if hair_color:
        try:
            rgb = tuple(int(x) for x in hair_color.split(","))
            if len(rgb) == 3:
                sketch_t = recolor_sketch(sketch_t, rgb)
        except ValueError:
            pass

    device = _STATE["device"]
    with _GPU_LOCK:
        hair_latent, face_latent_bld = run_sampling(
            _STATE["controlnet"], _STATE["transformer"], _STATE["scheduler"],
            _STATE["schedule"], sketch_t, matte_t, num_steps, device,
            vae=_STATE["vae"], face=face_t,
        )
        if face_t is not None and composite:
            result = composite_full(
                _STATE["vae"], hair_latent, matte_t, face_t, device,
                face_latent=face_latent_bld,
            )
        else:
            result = decode_hair(_STATE["vae"], hair_latent)

    return _png_response(to_pil(result.cpu()))


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _STATE.update(_load_models(args.config, args.checkpoint, device))
    print(f"[demo] Ready on {device} — http://{args.host}:{args.port}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
