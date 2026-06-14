/* HairDiT demo client.
 * Layered canvases: face (RGB), matte (white-on-transparent), sketch (color-on-transparent).
 * Drawing uses PointerEvents (mouse / pen / touch) with optional pressure.
 */

const W = 512, H = 512;

const display = document.getElementById('display');
const dctx    = display.getContext('2d');
const result  = document.getElementById('result');

const faceLayer   = makeOffscreen();   // RGB image, transparent if none
const matteLayer  = makeOffscreen();   // white strokes, transparent bg → exported on black = grayscale matte
const sketchLayer = makeOffscreen();   // colored strokes, transparent bg → exported on black = RGB sketch

let hasFace = false;
let tintedMatteCache = null;            // recomputed lazily

const state = {
  tool: 'matte-brush',
  color: '#8b5a2b',
  size: 20,
  pressure: true,
  showMatte: true,
  showSketch: true,
  showFace: true,
};

// --------------------------------------------------------------------------
// Setup

function makeOffscreen() {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  return c;
}

function bindToolbar() {
  document.querySelectorAll('#tool-seg button').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#tool-seg button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.tool = btn.dataset.tool;
    });
  });

  document.getElementById('color').addEventListener('input', e => {
    state.color = e.target.value;
  });
  const sizeInput = document.getElementById('size');
  const sizeVal   = document.getElementById('size-val');
  sizeInput.addEventListener('input', e => {
    state.size = +e.target.value;
    sizeVal.textContent = state.size;
  });
  document.getElementById('pressure').addEventListener('change', e => {
    state.pressure = e.target.checked;
  });

  document.getElementById('clear-matte').addEventListener('click', () => {
    clearLayer(matteLayer); tintedMatteCache = null; redraw();
  });
  document.getElementById('clear-sketch').addEventListener('click', () => {
    clearLayer(sketchLayer); redraw();
  });
  document.getElementById('clear-face').addEventListener('click', () => {
    clearLayer(faceLayer); hasFace = false; redraw();
  });
  document.getElementById('face-input').addEventListener('change', onFaceUpload);

  document.getElementById('sketch2matte').addEventListener('click', onSketch2Matte);
  document.getElementById('infer').addEventListener('click', onInfer);
  document.getElementById('save-result').addEventListener('click', onSaveResult);

  for (const [id, key] of [['show-matte', 'showMatte'], ['show-sketch', 'showSketch'], ['show-face', 'showFace']]) {
    document.getElementById(id).addEventListener('change', e => {
      state[key] = e.target.checked; redraw();
    });
  }
}

function clearLayer(c) {
  c.getContext('2d').clearRect(0, 0, W, H);
}

// --------------------------------------------------------------------------
// Drawing

let drawing = false;
let lastX = 0, lastY = 0;

function getXY(e) {
  const rect = display.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (W / rect.width),
    y: (e.clientY - rect.top)  * (H / rect.height),
  };
}

function strokeOn(layer, x0, y0, x1, y1, lineWidth, color, erase) {
  const ctx = layer.getContext('2d');
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = Math.max(1, lineWidth);
  if (erase) {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.strokeStyle = 'rgba(0,0,0,1)';
  } else {
    ctx.strokeStyle = color;
  }
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
  ctx.restore();
}

function applyStroke(x, y, pressure) {
  const isMatte = state.tool.startsWith('matte');
  const isErase = state.tool.endsWith('erase');
  const layer = isMatte ? matteLayer : sketchLayer;
  // Matte = mask; pressure modulation would make it too soft → constant size.
  // Sketch = stroke; pressure modulates linewidth.
  let lw = state.size;
  if (!isMatte && state.pressure) {
    lw = state.size * Math.max(0.25, pressure || 0.5);
  }
  const color = isMatte ? '#ffffff' : state.color;
  strokeOn(layer, lastX, lastY, x, y, lw, color, isErase);
  if (isMatte) tintedMatteCache = null;
  lastX = x; lastY = y;
}

display.addEventListener('pointerdown', e => {
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  drawing = true;
  display.setPointerCapture(e.pointerId);
  const { x, y } = getXY(e);
  lastX = x; lastY = y;
  // Single tap = small dot.
  applyStroke(x + 0.01, y + 0.01, e.pressure || 0.5);
  redraw();
  e.preventDefault();
});

display.addEventListener('pointermove', e => {
  if (!drawing) return;
  const { x, y } = getXY(e);
  applyStroke(x, y, e.pressure);
  redraw();
  e.preventDefault();
});

['pointerup', 'pointercancel', 'pointerleave'].forEach(ev => {
  display.addEventListener(ev, () => { drawing = false; });
});

// --------------------------------------------------------------------------
// Composite display

function tintLayer(srcCanvas, color) {
  const out = makeOffscreen();
  const ctx = out.getContext('2d');
  ctx.drawImage(srcCanvas, 0, 0);
  ctx.globalCompositeOperation = 'source-in';
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, W, H);
  return out;
}

function redraw() {
  dctx.clearRect(0, 0, W, H);
  // background
  dctx.fillStyle = '#161922';
  dctx.fillRect(0, 0, W, H);
  if (hasFace && state.showFace) dctx.drawImage(faceLayer, 0, 0);

  if (state.showMatte) {
    if (!tintedMatteCache) tintedMatteCache = tintLayer(matteLayer, '#00d4ff');
    dctx.save();
    dctx.globalAlpha = 0.4;
    dctx.drawImage(tintedMatteCache, 0, 0);
    dctx.restore();
  }
  if (state.showSketch) dctx.drawImage(sketchLayer, 0, 0);
}

// --------------------------------------------------------------------------
// Export helpers (layer → black-bg PNG blob)

function flattenOnBlack(layer) {
  const tmp = makeOffscreen();
  const ctx = tmp.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, W, H);
  ctx.drawImage(layer, 0, 0);
  return new Promise(res => tmp.toBlob(res, 'image/png'));
}

function faceBlob() {
  return new Promise(res => faceLayer.toBlob(res, 'image/png'));
}

// --------------------------------------------------------------------------
// Actions

function onFaceUpload(e) {
  const f = e.target.files && e.target.files[0];
  if (!f) return;
  const img = new Image();
  img.onload = () => {
    const ctx = faceLayer.getContext('2d');
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(img, 0, 0, W, H);
    hasFace = true;
    redraw();
    URL.revokeObjectURL(img.src);
  };
  img.src = URL.createObjectURL(f);
}

async function onSketch2Matte() {
  if (!isSketchEmpty()) {
    const blob = await flattenOnBlack(sketchLayer);
    const fd = new FormData();
    fd.append('sketch', blob, 'sketch.png');
    setStatus('sketch → matte…');
    const r = await fetch('/api/sketch2matte', { method: 'POST', body: fd });
    if (!r.ok) { setStatus('sketch2matte failed: ' + r.status); return; }
    const b = await r.blob();
    await loadMatteFromGrayscaleBlob(b);
    setStatus('');
  }
}

function isSketchEmpty() {
  const data = sketchLayer.getContext('2d').getImageData(0, 0, W, H).data;
  for (let i = 3; i < data.length; i += 4) if (data[i] !== 0) return false;
  return true;
}

function loadMatteFromGrayscaleBlob(blob) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const ctx = matteLayer.getContext('2d');
      ctx.clearRect(0, 0, W, H);
      // Draw image temporarily, then convert luminance → alpha (white pixels with alpha = L).
      const tmp = makeOffscreen();
      const tctx = tmp.getContext('2d');
      tctx.drawImage(img, 0, 0, W, H);
      const id = tctx.getImageData(0, 0, W, H);
      for (let i = 0; i < id.data.length; i += 4) {
        const v = id.data[i];                            // grayscale R == L
        id.data[i] = id.data[i + 1] = id.data[i + 2] = 255;
        id.data[i + 3] = v;
      }
      ctx.putImageData(id, 0, 0);
      tintedMatteCache = null;
      redraw();
      URL.revokeObjectURL(img.src);
      resolve();
    };
    img.src = URL.createObjectURL(blob);
  });
}

async function onInfer() {
  const sketchBlob = await flattenOnBlack(sketchLayer);
  const matteBlob  = await flattenOnBlack(matteLayer);
  const fd = new FormData();
  fd.append('sketch', sketchBlob, 'sketch.png');
  fd.append('matte', matteBlob, 'matte.png');
  if (hasFace) {
    const fb = await faceBlob();
    if (fb) fd.append('face', fb, 'face.png');
  }
  fd.append('num_steps', document.getElementById('steps').value);
  fd.append('composite', document.getElementById('composite').checked ? 'true' : 'false');

  setBusy(true);
  const t0 = performance.now();
  try {
    const r = await fetch('/api/infer', { method: 'POST', body: fd });
    if (!r.ok) {
      setStatus('infer failed: ' + r.status);
      return;
    }
    const blob = await r.blob();
    const img = new Image();
    img.onload = () => {
      const ctx = result.getContext('2d');
      ctx.clearRect(0, 0, W, H);
      ctx.drawImage(img, 0, 0, W, H);
      document.getElementById('save-result').disabled = false;
      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(blob);
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`done — ${dt}s`);
  } catch (err) {
    setStatus('error: ' + err.message);
  } finally {
    setBusy(false);
  }
}

function onSaveResult() {
  result.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `hairdit-${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
  }, 'image/png');
}

// --------------------------------------------------------------------------
// UI status

function setStatus(s) {
  document.getElementById('infer-status').textContent = s;
}
function setBusy(b) {
  document.getElementById('infer').disabled = b;
  document.getElementById('sketch2matte').disabled = b;
  if (b) setStatus('running…');
}

async function pingHealth() {
  try {
    const r = await fetch('/api/health');
    if (!r.ok) throw new Error(r.status);
    const j = await r.json();
    const el = document.getElementById('status');
    el.textContent = j.ready ? `ready · ${j.device} · ${j.schedule || 'no-schedule'}` : 'loading…';
    el.classList.toggle('ready', j.ready);
  } catch {
    document.getElementById('status').textContent = 'offline';
  }
}

// --------------------------------------------------------------------------

bindToolbar();
redraw();
pingHealth();
setInterval(pingHealth, 5000);
