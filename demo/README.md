# HairDiT Web Demo

Interactive web demo for HairDiT — paint the hair matte, sketch colored hair
strokes, and run inference from your browser. Supports mouse, pen, and touch
with tablet pressure sensitivity (PointerEvents).

<!-- Add a UI screenshot to demo/assets/ui.png and uncomment:
![HairDiT demo UI](assets/ui.png)
-->

## Quick start

You need Python 3.10+ and a CUDA-capable GPU. No Node.js / npm step is
required — the frontend is plain HTML / CSS / JS and is served as static files
by the backend.

```
# from the repo root
pip install -r requirements.txt
pip install -r demo/requirements.txt

python demo/server.py \
  --checkpoint checkpoints/mcs1_phase2/final.pth \
  --config     configs/mcs1_phase2.yaml \
  --port 7860
```

then open http://0.0.0.0:7860 in your browser.
(*If http://0.0.0.0:7860 does not work, try http://localhost:7860.*)

The server loads the VAE, transformer, and ControlNet once at startup, then
holds them in GPU memory for the lifetime of the process.

## Requirements

- Python 3.10+
- PyTorch with CUDA (see the root `requirements.txt`)
- ~12 GB of GPU memory for inference at 512×512 (bfloat16, 20 sampling steps)
- A modern browser supporting [PointerEvents](https://developer.mozilla.org/docs/Web/API/Pointer_events) (Chrome, Edge, Safari, Firefox)

## Installation

Two install steps — first the project's main dependencies, then the demo
backend dependencies:

```
pip install -r requirements.txt          # torch / diffusers / ...
pip install -r demo/requirements.txt     # fastapi / uvicorn / python-multipart
```

If a `{checkpoint}_infer.pth` (ControlNet-only) file exists next to the full
checkpoint, the server loads it preferentially. Otherwise the full training
checkpoint is loaded, which requires significantly more host RAM.

## Usage

### CLI arguments

| arg            | description                                                                       |
| -------------- | --------------------------------------------------------------------------------- |
| `--checkpoint` | ControlNet checkpoint path (`{name}.pth` or `{name}_infer.pth`) **(required)**    |
| `--config`     | YAML config used at training time (e.g. `configs/mcs1_phase2.yaml`) **(required)** |
| `--device`     | `cuda` / `cpu` (default: auto)                                                    |
| `--host`       | bind host (default `0.0.0.0`)                                                     |
| `--port`       | bind port (default `7860`)                                                        |

### Example flow

1. **Upload a face image** (optional). Without one, only the hair patch is
   returned; with one, the result is a latent-composited full image.

   <!-- ![Face upload](assets/01_face.png) -->

2. **Brush the matte.** Select *Matte brush* and paint over the hair region.
   Use *Matte erase* to clean up. The brush width is constant (mask
   semantics).

   <!-- ![Matte brushing](assets/02_matte.png) -->

3. **Draw sketch strokes.** Switch to *Sketch pen*, pick a hair color, and
   draw flow lines on top of the face. With a tablet, line width follows pen
   pressure automatically (toggle from the toolbar).

   <!-- ![Sketch lines](assets/03_sketch.png) -->

4. **(Optional) Derive a matte from the sketch.** Click *Sketch → Matte* to
   auto-generate the matte from the foreground of your sketch
   (same as `scripts/infer_custom.sketch_to_matte`).

5. **Run inference.** Click *Inference*. The result appears on the right and
   can be downloaded as PNG.

   <!-- ![Result](assets/04_result.png) -->

### Features

- **Matte brushing** — paint the hair mask directly (white brush / eraser).
- **Sketch lines** — draw colored strokes for hair (mouse / pen / touch).
- **Pen pressure** — tablet pressure (Wacom, etc.) modulates the sketch line
  width. Toggleable from the toolbar.
- **Sketch → Matte** — derives a matte from the sketch foreground.
- **Inference** — face-composited full image, or hair patch only.
- **Layer toggles** — show / hide face, matte, and sketch independently.
- **PNG download** — save the result with one click.

## API

The backend exposes three HTTP endpoints. All image bodies are PNG.

| method   | endpoint               | form fields                                                                                                          | response                                |
| -------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `GET`    | `/api/health`          | —                                                                                                                    | `{ready, device, schedule, model_id}`   |
| `POST`   | `/api/sketch2matte`    | `sketch` (PNG, RGB)                                                                                                  | matte PNG (grayscale)                   |
| `POST`   | `/api/infer`           | `sketch`, `matte`, optional `face`, `num_steps` (default 20), `composite` (default true), `hair_color` (`"R,G,B"`)   | result PNG                              |

`sketch` must be RGB with a black background and colored strokes;
`matte` must be grayscale with white = hair region. Both conventions match
`scripts/infer_custom.py`.

### Example: `curl`

```
curl -F sketch=@my_sketch.png \
     -F matte=@my_matte.png \
     -F face=@my_face.png \
     -F num_steps=20 \
     http://localhost:7860/api/infer \
     -o result.png
```

## Troubleshooting

- **`http://0.0.0.0:7860` shows nothing.** Try `http://localhost:7860`
  instead. Some browsers do not resolve `0.0.0.0` as a client address.
- **Status badge stays on `loading…`.** Model loading takes a minute on the
  first run. Check the server log — the message `[demo] Ready on cuda` means
  the models are loaded.
- **Status badge shows `offline`.** The frontend cannot reach the backend.
  Check that the server is still running and that the port matches.
- **CUDA out-of-memory.** Reduce the number of sampling steps from the
  toolbar (`Steps`) or run on a GPU with at least 12 GB of memory.
- **Inference is slow.** Sampling cost is dominated by the SD3 transformer;
  20 steps at 512×512 is the default. Lower `Steps` for faster previews.
- **Pen pressure has no effect.** Confirm `pressure` is checked in the
  toolbar. Touch input without a pressure-capable digitizer reports a
  constant 0.5; this is browser behavior, not a bug.
- **Sketch brushed area looks faded after `Sketch → Matte`.** The endpoint
  returns a grayscale matte; the client maps luminance to alpha. Anti-aliased
  edges are intentional and pass through to the model as soft mattes.

## Notes and limitations

- The canvas resolution is fixed at 512×512 to match the model input.
- On a single GPU, concurrent inference requests are serialized with a lock.
- The matte brush uses constant width (mask semantics). Only the sketch pen
  reacts to pressure.
- This demo is for interactive exploration. For batch inference, use
  `scripts/infer_custom.py` instead.
