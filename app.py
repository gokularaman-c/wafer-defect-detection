# app.py — FastAPI inference for WaferNet (DenseNet121+ECA, GAP+GMP head)
# Entry point for wafer defect detection application

import os
import io
import base64
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from gradcam import GradCAM
from image_preprocess import load_input_any
from model_utils import get_gradcam_target_wafernet
from overlay import overlay_cam_on_wafer

from image_preview import render_png_bytes

# ─── Labels (match training order) ──────────────────────────────────────────────
CLASS_NAMES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Random", "Scratch", "Near-Full", "none"]
N_CLASSES = len(CLASS_NAMES)

# ─── Model definition (embedded) ────────────────────────────────────────────────
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # x: [B,C,H,W]
        y = self.avg(x)  # [B,C,1,1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y


class WaferNet(nn.Module):
    def __init__(self, n_classes=9, in_ch=2, use_gmp=True, dropout_p=0.25, use_eca=True, k_size=3):
        super().__init__()
        m = models.densenet121(weights=None)
        # current stem: 7x7 stride-2
        m.features.conv0 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = m.features
        self.norm = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=True)

        self.use_eca = use_eca
        self.eca = ECA(1024, k_size=k_size) if use_eca else nn.Identity()

        self.use_gmp = use_gmp
        feat_dim = 1024
        head_dim = feat_dim * (2 if use_gmp else 1)
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p else nn.Identity()
        self.cls = nn.Linear(head_dim, n_classes)

    @staticmethod
    def masked_gap(feat, mask):
        masked = feat * mask
        denom = mask.sum((2, 3)).clamp_min(1e-6)
        return masked.sum((2, 3)) / denom

    @staticmethod
    def masked_gmp(feat, mask):
        # SAFE sentinel to avoid extreme values propagating to the linear head
        very_neg = feat.new_tensor(-1e4)
        masked = torch.where(mask > 0, feat, very_neg)
        gmp = masked.amax((2, 3))
        return torch.nan_to_num(gmp, nan=0.0, posinf=1e4, neginf=-1e4)

    def forward(self, x):  # x = [B,in_ch,96,96]; channel-1 is mask
        mask = x[:, 1:2]
        feat = self.features(x)  # [B,1024,h,w]
        feat = self.relu(self.norm(feat))
        feat = self.eca(feat)

        mask_ds = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")

        # If any sample has an empty mask after downsampling, fall back to all-ones
        empty = (mask_ds.sum((2, 3)) == 0)
        if empty.any():
            mask_ds = mask_ds.clone()
            mask_ds[empty] = 1.0

        gap = self.masked_gap(feat, mask_ds)
        if self.use_gmp:
            gmp = self.masked_gmp(feat, mask_ds)
            h = torch.cat([gap, gmp], dim=1)
        else:
            h = gap
        h = torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        h = self.dropout(h)
        return self.cls(h)  # logits [B,9]


# ─── Logging / device / lazy model loading ────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("wafernet.api")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)
BASE_DIR = Path(__file__).resolve().parent
CHANNELS_LAST = DEVICE.type == "cuda"


def resolve_weights_path() -> Optional[Path]:
    """Try to find weights file. Returns a Path if found, else None."""
    # 1) Explicit env var
    env_p = os.getenv("WAFERNET_WEIGHTS")
    if env_p:
        p = Path(env_p)
        if not p.is_absolute():
            p = BASE_DIR / p
        if p.exists():
            return p
        logger.error("WAFERNET_WEIGHTS points to '%s', but it does not exist.", p)

    # 2) Common local paths (include your file with a space in the name)
    candidates = [
        BASE_DIR / "weights" / "wafernet_best_4th sep.pth",
    ]
    # 3) Notebooks subfolders
    candidates += [
        BASE_DIR / "notebooks" / "weights" / "wafernet_best_4th sep.pth",

    ]
    for c in candidates:
        if c.exists():
            return c

    # 4) Best-effort search
    nb_dir = BASE_DIR / "notebooks"
    if nb_dir.exists():
        patterns = ["wafernet*best*.pth", "wafernet*_final*.pth", "wafernet*.pth"]
        for pat in patterns:
            for p in nb_dir.rglob(pat):
                if p.is_file():
                    return p
    return None


WEIGHTS_PATH: Optional[Path] = None
_model: Optional[nn.Module] = None
_model_load_error: Optional[str] = None


def _extract_state_dict(obj: Union[dict, "OrderedDict", torch.Tensor]) -> dict:
    """Extract a state dict from common checkpoint formats."""
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break
        sd = obj
    else:
        sd = obj
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def load_model(force: bool = False) -> Optional[nn.Module]:
    """Load the model once; cache globally."""
    global _model, WEIGHTS_PATH, _model_load_error
    if _model is not None and not force:
        return _model

    try:
        if WEIGHTS_PATH is None:
            WEIGHTS_PATH = resolve_weights_path()
        if WEIGHTS_PATH is None:
            raise FileNotFoundError(
                "Could not find model weights. Set WAFERNET_WEIGHTS or place a file in ./weights or ./notebooks/weights."
            )

        logger.info("Loading WaferNet weights from: %s", WEIGHTS_PATH)
        state_obj = torch.load(WEIGHTS_PATH, map_location="cpu")
        state = _extract_state_dict(state_obj)

        model = WaferNet(n_classes=N_CLASSES, in_ch=2, use_gmp=True, dropout_p=0.25, use_eca=True, k_size=3)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys when loading state_dict: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading state_dict: %s", unexpected)

        model.eval().to(DEVICE)
        if DEVICE.type == "cuda":
            model.to(memory_format=torch.channels_last)
            torch.backends.cudnn.benchmark = True

        _model = model
        _model_load_error = None
        return _model
    except Exception as e:
        _model = None
        _model_load_error = f"{type(e).__name__}: {e}"
        logger.exception("Failed to load model: %s", _model_load_error)
        return None


# (Optional) temperature and per-class bias for calibration
try:
    TEMPERATURE = float(os.getenv("WAFERNET_TEMPERATURE", "1.0"))
except Exception:
    logger.warning("Invalid WAFERNET_TEMPERATURE; defaulting to 1.0")
    TEMPERATURE = 1.0

BIAS_VEC = os.getenv("WAFERNET_BIAS_VEC", None)
bias_tensor = None
if BIAS_VEC:
    try:
        vals = [float(v.strip()) for v in BIAS_VEC.split(",")]
        if len(vals) != N_CLASSES:
            raise ValueError(f"WAFERNET_BIAS_VEC must have {N_CLASSES} values (got {len(vals)}).")
        bias_tensor = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
    except Exception as e:
        logger.warning("Ignoring invalid WAFERNET_BIAS_VEC: %s", e)
        bias_tensor = None

# ─── FastAPI setup ─────────────────────────────────────────────────────────────
app = FastAPI(title="WaferNet Inference API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class JsonPredictRequest(BaseModel):
    wafer_map: List[List[float]]             # 2D list; values in {0,1,2} or {0.0,0.5,1.0}
    mask: Optional[List[List[float]]] = None # optional; if missing, derived as (wafer_map > 0)
    resize_to: Optional[int] = 96            # default 96
    return_topk: Optional[int] = 3

class PredictResponse(BaseModel):
    predicted_index: int
    predicted_label: str
    probabilities: List[float]               # size 9
    topk: List[Tuple[str, float]]            # [(label, prob)]
    logits: Optional[List[float]] = None

# ─── Preprocess helpers ────────────────────────────────────────────────────────
def _to_96(x: np.ndarray, size: int = 96) -> np.ndarray:
    """Center-pad to size×size; downscale with nearest if larger."""
    h, w = x.shape[-2:]
    if h == size and w == size:
        return x.astype(np.float32, copy=False)
    if h <= size and w <= size:
        out = np.zeros((size, size), np.float32)
        top = (size - h) // 2
        left = (size - w) // 2
        out[top:top+h, left:left+w] = x.astype(np.float32)
        return out
    # larger -> nearest-neighbor downscale (preserves discrete values)
    img = Image.fromarray(x.astype(np.float32), mode="F")
    img = img.resize((size, size), resample=Image.NEAREST)
    return np.array(img, dtype=np.float32)

def _normalize_wafer_image(w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float32)
    u = np.unique(w)
    mx = float(w.max())

    # Already normalized floats (0, 0.5, 1)
    if mx <= 1.0 + 1e-6:
        return np.clip(w, 0.0, 1.0)

    # Stored as 0/1/2 (or 0/1/2/3 with rare noise)
    if mx <= 3.5 and set(np.round(u).tolist()) <= {0, 1, 2, 3}:
        return np.clip(w / 2.0, 0.0, 1.0)

    # 8-bit 0/128/255
    if mx <= 255.0 + 1e-6:
        w = np.where(w >= 192, 2.0, np.where(w >= 64, 1.0, 0.0))
        return np.clip(w / 2.0, 0.0, 1.0)

    # Fallback: scale to [0,1] then quantize to {0,0.5,1}
    w = w / (mx + 1e-8)
    w = np.where(w > 0.75, 1.0, np.where(w > 0.25, 0.5, 0.0))
    return w

def _prep_from_arrays(wafer: np.ndarray, mask: Optional[np.ndarray], size: int = 96) -> torch.Tensor:
    """
    wafer: HxW in {0,1,2} or {0.0,0.5,1.0}
    mask:  HxW in {0,1} (optional)
    returns tensor [1,2,96,96] float32 on DEVICE
    """
    from fastapi import HTTPException

    wafer = _normalize_wafer_image(wafer)
    wafer = np.clip(wafer, 0.0, 1.0)

    if mask is None:
        mask = (wafer > 0.0).astype(np.float32)
    else:
        mask = (mask > 0.5).astype(np.float32)

    wafer = _to_96(wafer, size)
    mask  = _to_96(mask, size)

    # Early validation of inputs
    uniq = set(np.unique(np.round(wafer, 3)).tolist())
    if not uniq.issubset({0.0, 0.5, 1.0}):
        raise HTTPException(
            status_code=400,
            detail=f"wafer_map must contain only {{0,0.5,1.0}}, found {sorted(list(uniq))[:8]}"
        )
    if float(mask.sum()) < 1.0:
        raise HTTPException(status_code=400, detail="Mask has no on-wafer pixels (all zeros).")

    x = np.stack([wafer, mask], axis=0).astype(np.float32)   # [2,H,W]
    x = np.expand_dims(x, axis=0)                            # [1,2,H,W]
    t = torch.from_numpy(x).to(DEVICE, non_blocking=True)
    if CHANNELS_LAST:
        t = t.to(memory_format=torch.channels_last)
    return t

def _softmax_with_calibration(logits: torch.Tensor) -> torch.Tensor:
    """NaN/Inf-proof calibrated softmax."""
    z = logits
    if bias_tensor is not None:
        z = z - bias_tensor.view(1, -1)
    if TEMPERATURE != 1.0:
        z = z / max(TEMPERATURE, 1e-6)
    z = torch.nan_to_num(z, neginf=-1e9, posinf=1e9)
    p = F.softmax(z, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return p

# ─── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return (
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>WaferNet · Inference API</title>
            <style>
              :root{
                --bg:#000;
                --fg:#fff;
                --surface:#1e1e1e;
                --surface-2:#232323;
                --muted:rgba(255,255,255,.62);
                --muted-2:rgba(255,255,255,.40);
                --ring:rgba(255,255,255,.20);
              }
              *{box-sizing:border-box}
              html,body{height:100%}
              body{
                margin:0;
                font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;
                background:var(--bg);
                color:var(--fg);
                -webkit-font-smoothing:antialiased;
                line-height:1.5;
              }
              header{
                display:flex;
                align-items:center;
                justify-content:space-between;
                gap:16px;
                padding:16px 20px;
                position:sticky;
                top:0;
                background:rgba(0,0,0,.78);
                backdrop-filter:blur(6px);
                border-bottom:1px solid rgba(255,255,255,.08);
                z-index:10;
              }
              .brand{
                display:flex;
                align-items:center;
                gap:12px;
                color:var(--fg);
                text-decoration:none;
              }
              .logo{
                width:32px;
                height:32px;
                border-radius:10px;
                background:linear-gradient(140deg,var(--surface),var(--surface-2));
                display:grid;
                place-items:center;
                box-shadow:0 0 0 1px rgba(255,255,255,.06) inset,0 10px 26px rgba(0,0,0,.45);
              }
              .logo svg{width:18px;height:18px;opacity:.9}
              nav a{
                color:var(--muted);
                text-decoration:none;
                margin-left:16px;
                font-size:14px;
                padding:8px 12px;
                border-radius:10px;
                border:1px solid transparent;
                transition:all .12s ease;
              }
              nav a:hover{color:var(--fg);border-color:var(--ring);background:var(--surface-2);}
              .container{
                max-width:1040px;
                margin:0 auto;
                padding:36px 20px 64px;
              }
              h1{font-size:clamp(30px,4vw,46px);margin:24px 0 6px;letter-spacing:-0.02em;}
              .sub{color:var(--muted);max-width:60ch;}
              .grid{
                display:grid;
                gap:18px;
                margin-top:28px;
              }
              @media(min-width:960px){.grid{grid-template-columns:1.08fr .92fr;}}
              .card{
                background:linear-gradient(175deg,var(--surface),var(--surface-2));
                border:1px solid rgba(255,255,255,.08);
                border-radius:18px;
                padding:22px;
                box-shadow:0 12px 32px rgba(0,0,0,.40);
              }
              .card h2{margin:0 0 12px;font-size:18px;letter-spacing:-0.01em;}
              .drop{
                display:grid;
                place-items:center;
                text-align:center;
                padding:26px;
                border:1.6px dashed rgba(255,255,255,.18);
                border-radius:16px;
                background:#050505;
                transition:border-color .16s ease,transform .12s ease;
              }
              .drop.dragover{border-color:var(--fg);transform:scale(1.01);}
              .drop button{
                appearance:none;
                border:none;
                cursor:pointer;
                padding:10px 14px;
                border-radius:12px;
                background:#fff;
                color:#000;
                font-weight:600;
                margin:6px 4px 0;
                box-shadow:0 10px 22px rgba(0,0,0,.35);
                transition:transform .08s ease,box-shadow .18s ease;
              }
              .drop button:hover{transform:translateY(-1px);box-shadow:0 16px 30px rgba(0,0,0,.45);}
              .drop button:active{transform:translateY(0);}
              .file-meta{font-size:12px;color:var(--muted-2);margin-top:6px;}
              .actions{display:grid;gap:10px;margin-top:18px;}
              @media(min-width:500px){.actions{grid-template-columns:1fr 1fr;}}
              .params{display:grid;gap:12px;margin-top:16px;}
              @media(min-width:500px){.params{grid-template-columns:1fr 1fr;}}
              .params label{display:flex;flex-direction:column;gap:6px;font-size:13px;color:var(--muted);}
              .params input{
                padding:10px 12px;
                border-radius:10px;
                border:1px solid rgba(255,255,255,.18);
                background:#050505;
                color:var(--fg);
              }
              .params input:focus{outline:none;border-color:var(--ring);box-shadow:0 0 0 3px rgba(255,255,255,.08);}
              .btn{
                appearance:none;
                border:none;
                cursor:pointer;
                padding:14px 16px;
                border-radius:14px;
                font-weight:600;
                font-size:15px;
                transition:transform .08s ease,box-shadow .18s ease,opacity .18s ease;
              }
              .btn.primary{
                background:#fff;
                color:#000;
                box-shadow:0 12px 26px rgba(0,0,0,.36);
              }
              .btn.secondary{
                background:transparent;
                color:var(--fg);
                border:1px solid rgba(255,255,255,.22);
              }
              .btn:disabled{opacity:.55;cursor:not-allowed;transform:none;box-shadow:none;}
              .btn:not(:disabled):hover{transform:translateY(-1px);box-shadow:0 18px 32px rgba(0,0,0,.45);}
              .hint{font-size:12px;color:var(--muted-2);margin-top:10px;min-height:16px;}
              .preview{display:none;margin-top:16px;border-radius:14px;overflow:hidden;border:1px solid rgba(255,255,255,.12);}
              .preview img{display:block;width:100%;height:auto;}
              .prediction{
                display:none;
              }
              .prediction ul{list-style:none;padding:0;margin:12px 0 0;display:grid;gap:6px;}
              .prediction li{
                display:flex;
                justify-content:space-between;
                font-size:14px;
                padding:10px 12px;
                border-radius:10px;
                background:rgba(255,255,255,.04);
                border:1px solid rgba(255,255,255,.08);
              }
              footer{color:var(--muted-2);font-size:12px;margin-top:28px;text-align:center;}
            </style>
          </head>
          <body>
            <header>
              <a class="brand" href="/">
                <div class="logo" aria-hidden="true">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 8h16v8H4z"/><path d="M8 12h8"/><path d="M12 8v8"/></svg>
                </div>
                <strong>WaferNet</strong>
              </a>
              <nav aria-label="Quick links">
                <a href="/docs">API Docs</a>
                <a href="/health">Health</a>
                <a href="/classes">Classes</a>
              </nav>
            </header>
            <main class="container">
              <h1>Inference Console</h1>
              <p class="sub">Upload a wafer map and get two actions: generate a Grad-CAM explanation or retrieve the probability distribution.</p>
              <div class="grid">
                <section class="card">
                  <h2>Upload wafer</h2>
                  <div class="drop" id="drop-zone">
                    <p>Drag & drop wafer maps or use the buttons below.</p>
                    <div>
                      <button type="button" id="wafer-browse">Select wafer</button>
                      <button type="button" id="mask-browse">Select mask (optional)</button>
                    </div>
                  <p class="file-meta" id="wafer-meta">No wafer file selected</p>
                  <p class="file-meta" id="mask-meta">No mask file selected</p>
                </div>
                <div class="params">
                  <label>
                    Resize (px)
                    <input type="number" id="resize-input" min="16" max="256" value="96" step="1" />
                  </label>
                  <label>
                    Top-k classes
                    <input type="number" id="topk-input" min="1" max="9" value="3" step="1" />
                  </label>
                </div>
                <input type="file" id="wafer-input" accept=".png,.jpg,.jpeg,.npy" hidden />
                  <input type="file" id="mask-input" accept=".png,.jpg,.jpeg,.npy" hidden />
                <div class="actions">
                  <button type="button" class="btn primary" id="explain-btn">Generate Grad-CAM</button>
                  <button type="button" class="btn secondary" id="predict-btn">Predict</button>
                </div>
                  <p class="hint" id="upload-hint">Accepted: PNG/JPG/NPY wafer plus optional mask.</p>
                  <div class="preview" id="explain-preview">
                    <img id="explain-image" alt="Grad-CAM overlay" />
                  </div>
                  <p class="hint" id="explain-hint"></p>
                </section>
                <section class="card prediction" id="predict-section">
                  <h2>Prediction</h2>
                  <p id="predict-label"></p>
                  <ul id="predict-list"></ul>
                  <p class="hint" id="predict-hint"></p>
                </section>
              </div>
              <footer>
                © 2025 WaferNet · DenseNet121 + Grad-CAM
              </footer>
             </main>
             <script>
               document.addEventListener('DOMContentLoaded', () => {
                 const CLASS_NAMES = ["Center","Donut","Edge-Loc","Edge-Ring","Loc","Random","Scratch","Near-Full","none"];
                 const dropZone = document.getElementById('drop-zone');
                 const waferInput = document.getElementById('wafer-input');
                 const maskInput = document.getElementById('mask-input');
                 const waferMeta = document.getElementById('wafer-meta');
                 const maskMeta = document.getElementById('mask-meta');
                 const waferBrowse = document.getElementById('wafer-browse');
                 const maskBrowse = document.getElementById('mask-browse');
                 const explainBtn = document.getElementById('explain-btn');
                 const predictBtn = document.getElementById('predict-btn');
                 const resizeInput = document.getElementById('resize-input');
                 const topkInput = document.getElementById('topk-input');
                 const explainPreview = document.getElementById('explain-preview');
                 const explainImage = document.getElementById('explain-image');
                 const explainHint = document.getElementById('explain-hint');
                 const predictSection = document.getElementById('predict-section');
                 const predictLabel = document.getElementById('predict-label');
                 const predictList = document.getElementById('predict-list');
                 const predictHint = document.getElementById('predict-hint');

                 function updateMeta() {
                   waferMeta.textContent = waferInput.files && waferInput.files.length ? `Wafer: ${waferInput.files[0].name}` : 'No wafer file selected';
                   maskMeta.textContent = maskInput.files && maskInput.files.length ? `Mask: ${maskInput.files[0].name}` : 'No mask file selected';
                 }

                 function resetExplain() {
                   explainPreview.style.display = 'none';
                   explainImage.removeAttribute('src');
                   explainHint.textContent = '';
                 }

                 function resetPredict() {
                   predictSection.style.display = 'none';
                   predictLabel.textContent = '';
                   predictList.innerHTML = '';
                   predictHint.textContent = '';
                 }

                 function clampResize() {
                   const value = parseInt(resizeInput.value, 10);
                   const clamped = Number.isFinite(value) ? Math.min(256, Math.max(16, value)) : 96;
                   resizeInput.value = String(clamped);
                   return clamped;
                 }

                 function clampTopk() {
                   const value = parseInt(topkInput.value, 10);
                   const clamped = Number.isFinite(value) ? Math.min(CLASS_NAMES.length, Math.max(1, value)) : 3;
                   topkInput.value = String(clamped);
                   return clamped;
                 }

                 function toFileList(files) {
                   if (!files || !files.length) return null;
                   if (typeof DataTransfer === 'undefined') return null;
                   const dt = new DataTransfer();
                   files.forEach(file => dt.items.add(file));
                   return dt.files;
                 }

                 function assignFiles(list) {
                   if (!list || !list.length) return;
                   const files = Array.from(list);
                   const waferFile = files[0];
                   const maskFile = files.length > 1 ? files[1] : null;
                   const waferList = toFileList([waferFile]);
                   if (waferList) {
                     waferInput.files = waferList;
                   }
                   if (maskFile) {
                     const maskList = toFileList([maskFile]);
                     if (maskList) maskInput.files = maskList;
                   }
                   updateMeta();
                   resetExplain();
                   resetPredict();
                 }

                 updateMeta();

                 ['dragenter','dragover'].forEach(eventName => {
                   dropZone.addEventListener(eventName, evt => {
                     evt.preventDefault();
                     evt.stopPropagation();
                     dropZone.classList.add('dragover');
                   });
                 });
                 ['dragleave','drop'].forEach(eventName => {
                   dropZone.addEventListener(eventName, evt => {
                     evt.preventDefault();
                     evt.stopPropagation();
                     dropZone.classList.remove('dragover');
                   });
                 });
                 dropZone.addEventListener('drop', evt => {
                   if (evt.dataTransfer && evt.dataTransfer.files) {
                     assignFiles(evt.dataTransfer.files);
                   }
                 });

                 waferBrowse.addEventListener('click', () => waferInput.click());
                 maskBrowse.addEventListener('click', () => maskInput.click());
                 waferInput.addEventListener('change', () => {
                   updateMeta();
                   resetExplain();
                   resetPredict();
                 });
                 maskInput.addEventListener('change', () => {
                   updateMeta();
                   resetExplain();
                   resetPredict();
                 });
                 resizeInput.addEventListener('change', clampResize);
                 topkInput.addEventListener('change', clampTopk);

                 clampResize();
                 clampTopk();

                 function ensureWafer() {
                   if (!waferInput.files || !waferInput.files.length) {
                     explainHint.textContent = 'Select a wafer file before running inference.';
                     explainHint.style.color = '#ff8080';
                     return false;
                   }
                   explainHint.style.color = 'var(--muted)';
                   return true;
                 }

                async function runExplain() {
                  if (!ensureWafer()) return;
                  const resizeVal = clampResize();
                  const formData = new FormData();
                  const waferFile = waferInput.files[0];
                  formData.append('wafer', waferFile, waferFile.name);
                  if (maskInput.files && maskInput.files.length) {
                    const maskFile = maskInput.files[0];
                    formData.append('mask', maskFile, maskFile.name);
                  }
                  formData.append('resize', String(resizeVal));

                  explainBtn.disabled = true;
                  explainBtn.textContent = 'Generating...';
                  explainHint.textContent = `Calculating Grad-CAM (resize=${resizeVal}).`;
                  explainHint.style.color = 'var(--muted)';

                  try {
                     const res = await fetch('/predict/explain', { method: 'POST', body: formData });
                     const payload = await res.json().catch(() => null);
                     if (!res.ok || !payload || !payload.overlay_png_base64) {
                       const detail = payload && payload.detail ? payload.detail : 'Failed to generate explanation.';
                       explainHint.textContent = detail;
                       explainHint.style.color = '#ff8080';
                       resetExplain();
                       return;
                     }
                     explainImage.src = payload.overlay_png_base64;
                     explainPreview.style.display = 'block';
                     explainHint.textContent = 'Grad-CAM overlay generated.';
                     explainHint.style.color = 'var(--muted-2)';
                   } catch (err) {
                     explainHint.textContent = 'Network error while generating Grad-CAM.';
                     explainHint.style.color = '#ff8080';
                     resetExplain();
                   } finally {
                     explainBtn.disabled = false;
                     explainBtn.textContent = 'Generate Prediction';
                   }
                 }

                async function runPredict() {
                  if (!ensureWafer()) return;
                  const resizeVal = clampResize();
                  const topkVal = clampTopk();
                  const formData = new FormData();
                  const waferFile = waferInput.files[0];
                  formData.append('wafer_file', waferFile, waferFile.name);
                  if (maskInput.files && maskInput.files.length) {
                    const maskFile = maskInput.files[0];
                    formData.append('mask_file', maskFile, maskFile.name);
                  }
                  formData.append('resize_to', String(resizeVal));
                  formData.append('return_topk', String(topkVal));

                  predictBtn.disabled = true;
                  predictBtn.textContent = 'Predicting...';
                  predictHint.textContent = `Running inference (k=${topkVal}, resize=${resizeVal}).`;
                  predictHint.style.color = 'var(--muted)';

                   try {
                     const res = await fetch('/predict/file', { method: 'POST', body: formData });
                     const payload = await res.json().catch(() => null);
                     if (!res.ok || !payload) {
                       const detail = payload && payload.detail ? payload.detail : 'Prediction failed.';
                       predictHint.textContent = detail;
                       predictHint.style.color = '#ff8080';
                       predictSection.style.display = 'block';
                       predictLabel.textContent = '';
                       predictList.innerHTML = '';
                       return;
                     }
                    const probs = Array.isArray(payload.probabilities) ? payload.probabilities : [];
                    const topkRaw = Array.isArray(payload.topk) ? payload.topk : [];
                    const topkPairs = topkRaw.map(entry => {
                      if (Array.isArray(entry) && entry.length >= 2) {
                        return { label: String(entry[0]), prob: Number(entry[1]) };
                      }
                      if (entry && typeof entry === 'object' && 'label' in entry && 'prob' in entry) {
                        return { label: String(entry.label), prob: Number(entry.prob) };
                      }
                      return null;
                    }).filter(item => item && Number.isFinite(item.prob));
                    const fallbackPairs = probs.map((prob, idx) => ({
                      label: CLASS_NAMES[idx] || `Class ${idx}`,
                      prob: Number(prob),
                    })).filter(item => Number.isFinite(item.prob));
                    const listSource = topkPairs.length ? topkPairs.slice(0) : fallbackPairs.slice().sort((a, b) => b.prob - a.prob);
                    const rows = listSource.slice(0, topkVal);
                    predictList.innerHTML = '';
                    rows.forEach(({ label, prob }) => {
                      const probVal = Number(prob);
                      const probText = Number.isFinite(probVal) ? probVal.toFixed(6) : '0.000000';
                      const pctText = Number.isFinite(probVal) ? (probVal * 100).toFixed(2) : '0.00';
                      const li = document.createElement('li');
                      li.innerHTML = `<span>${label}</span><span>${probText} (${pctText}%)</span>`;
                      predictList.appendChild(li);
                    });
                    const topRow = rows[0];
                    predictLabel.textContent = topRow ? `Predicted class: ${topRow.label} (p=${topRow.prob.toFixed(4)})` : `Predicted class: ${payload.predicted_label}`;
                    predictSection.style.display = 'block';
                    predictHint.textContent = `Top ${rows.length} classes (k=${topkVal}) at resize ${resizeVal} from /predict/file.`;
                    predictHint.style.color = 'var(--muted-2)';
                   } catch (err) {
                     predictHint.textContent = 'Network error while predicting.';
                     predictHint.style.color = '#ff8080';
                     predictSection.style.display = 'block';
                     predictLabel.textContent = '';
                     predictList.innerHTML = '';
                   } finally {
                     predictBtn.disabled = false;
                     predictBtn.textContent = 'Predict';
                   }
                 }

                 explainBtn.addEventListener('click', runExplain);
                 predictBtn.addEventListener('click', runPredict);
               });
             </script>
           </body>
         </html>
         """
     )


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    model_ok = load_model() is not None
    return {
        "status": "ok" if model_ok else "error",
        "device": str(DEVICE),
        "weights_path": str(WEIGHTS_PATH) if WEIGHTS_PATH else None,
        "classes": CLASS_NAMES,
        "error": _model_load_error,
    }

@app.get("/classes")
def classes():
    return {"classes": CLASS_NAMES}

@app.post("/predict/json", response_model=PredictResponse)
def predict_json(req: JsonPredictRequest):
    mdl = load_model()
    if mdl is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_model_load_error}")

    wafer = np.array(req.wafer_map, dtype=np.float32)
    mask = None if req.mask is None else np.array(req.mask, dtype=np.float32)

    # Basic validation
    from fastapi import HTTPException
    if wafer.ndim != 2:
        raise HTTPException(status_code=400, detail=f"wafer_map must be 2D (got shape {list(wafer.shape)})")
    if mask is not None and mask.ndim != 2:
        raise HTTPException(status_code=400, detail=f"mask must be 2D (got shape {list(mask.shape)})")
    if mask is not None and mask.shape != wafer.shape:
        raise HTTPException(status_code=400, detail=f"mask shape {list(mask.shape)} must match wafer_map shape {list(wafer.shape)}")
    size = int(req.resize_to or 96)
    if size <= 0:
        raise HTTPException(status_code=400, detail=f"resize_to must be positive (got {size})")

    x = _prep_from_arrays(wafer, mask, size=size)

    with torch.inference_mode():
        logits = mdl(x)
        probs = _softmax_with_calibration(logits)

    # Sanitize outputs before JSON
    logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)
    probs  = torch.nan_to_num(probs,  nan=0.0)

    probs_np = probs[0].detach().cpu().numpy().tolist()
    logits_np = logits[0].detach().cpu().numpy().tolist()
    idx = int(np.argmax(probs_np))
    label = CLASS_NAMES[idx]
    k = max(1, min(req.return_topk or 3, N_CLASSES))
    topk_idx = np.argsort(probs_np)[::-1][:k]
    topk = [(CLASS_NAMES[i], float(probs_np[i])) for i in topk_idx]
    return PredictResponse(predicted_index=idx, predicted_label=label, probabilities=probs_np, topk=topk, logits=logits_np)

@app.post("/predict/file", response_model=PredictResponse)
async def predict_file(
    wafer_file: UploadFile = File(..., description="Grayscale PNG/JPG or .npy; values {0,1,2} or {0,0.5,1.0}"),
    mask_file: Optional[UploadFile] = File(None, description="Optional mask PNG/JPG or .npy in {0,1}"),
    resize_to: int = Form(96),
    return_topk: int = Form(3),
):
    async def load_any(f: UploadFile) -> np.ndarray:
        name = (f.filename or "").lower()
        raw = await f.read()
        if name.endswith(".npy"):
            arr = np.load(io.BytesIO(raw))
        else:
            img = Image.open(io.BytesIO(raw)).convert("F")  # 32-bit float grayscale
            arr = np.array(img, dtype=np.float32)
        return arr

    mdl = load_model()
    if mdl is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_model_load_error}")

    try:
        wafer = await load_any(wafer_file)
        mask = await load_any(mask_file) if mask_file is not None else None
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Failed to parse inputs: {e}")

    # Basic validation
    from fastapi import HTTPException
    if wafer.ndim != 2:
        raise HTTPException(status_code=400, detail=f"wafer image must be 2D (got shape {list(wafer.shape)})")
    if mask is not None and mask.ndim != 2:
        raise HTTPException(status_code=400, detail=f"mask image must be 2D (got shape {list(mask.shape)})")
    if mask is not None and mask.shape != wafer.shape:
        raise HTTPException(status_code=400, detail=f"mask shape {list(mask.shape)} must match wafer image shape {list(wafer.shape)}")
    size = int(resize_to or 96)
    if size <= 0:
        raise HTTPException(status_code=400, detail=f"resize_to must be positive (got {size})")

    x = _prep_from_arrays(wafer, mask, size=size)

    with torch.inference_mode():
        logits = mdl(x)
        probs = _softmax_with_calibration(logits)

    # Sanitize outputs before JSON
    logits = torch.nan_to_num(logits, neginf=-1e9, posinf=1e9)
    probs  = torch.nan_to_num(probs,  nan=0.0)

    probs_np = probs[0].detach().cpu().numpy().tolist()
    logits_np = logits[0].detach().cpu().numpy().tolist()
    idx = int(np.argmax(probs_np))
    label = CLASS_NAMES[idx]
    k = max(1, min(return_topk or 3, N_CLASSES))
    topk_idx = np.argsort(probs_np)[::-1][:k]
    topk = [(CLASS_NAMES[i], float(probs_np[i])) for i in topk_idx]
    return PredictResponse(predicted_index=idx, predicted_label=label, probabilities=probs_np, topk=topk, logits=logits_np)


@app.post("/preview/file")
async def preview_file(
    wafer_file: UploadFile = File(..., description="Wafer .npy in (H,W) or (2,H,W) float32/float64"),
    mask_file: Optional[UploadFile] = File(None, description="Optional mask .npy in {0,1}"),
):
    from fastapi import HTTPException

    wafer_bytes = await wafer_file.read()
    if not wafer_bytes:
        raise HTTPException(status_code=400, detail="wafer_file is empty.")

    try:
        wafer_arr = np.load(io.BytesIO(wafer_bytes), allow_pickle=False)
    except Exception as e:  # pragma: no cover - numpy error messages are varied
        raise HTTPException(status_code=400, detail=f"Failed to load wafer_file: {e}") from e

    mask_arr: Optional[np.ndarray] = None
    if mask_file is not None:
        mask_bytes = await mask_file.read()
        if mask_bytes:
            try:
                mask_arr = np.load(io.BytesIO(mask_bytes), allow_pickle=False)
            except Exception as e:  # pragma: no cover - numpy error messages are varied
                raise HTTPException(status_code=400, detail=f"Failed to load mask_file: {e}") from e

    def _squeeze(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[0] == 1:
            return arr[0]
        return arr

    wafer_arr = np.asarray(wafer_arr)
    wafer_from_stack: Optional[np.ndarray] = None

    if wafer_arr.ndim == 2:
        wafer_np = wafer_arr.astype(np.float32)
    elif wafer_arr.ndim == 3 and wafer_arr.shape[0] == 2:
        wafer_np = wafer_arr[0].astype(np.float32)
        wafer_from_stack = wafer_arr[1].astype(np.float32)
    elif wafer_arr.ndim == 3 and wafer_arr.shape[0] == 1:
        wafer_np = wafer_arr[0].astype(np.float32)
    else:
        raise HTTPException(status_code=400, detail=f"Unexpected wafer array shape {wafer_arr.shape}. Expected (H,W) or (2,H,W).")

    if mask_arr is not None:
        mask_np = _squeeze(np.asarray(mask_arr)).astype(np.float32)
    elif wafer_from_stack is not None:
        mask_np = wafer_from_stack.astype(np.float32)
    else:
        mask_np = (wafer_np > 0.0).astype(np.float32)

    if mask_np.ndim != 2:
        raise HTTPException(status_code=400, detail=f"Mask must be 2D after processing, got shape {list(mask_np.shape)}")
    if wafer_np.shape != mask_np.shape:
        raise HTTPException(status_code=400, detail=f"Mask shape {list(mask_np.shape)} must match wafer shape {list(wafer_np.shape)}")

    mask_bin = (mask_np > 0.5).astype(np.float32)
    wafer_np = np.clip(np.nan_to_num(wafer_np, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    png_bytes = render_png_bytes(wafer_np, mask_bin)
    return Response(content=png_bytes, media_type="image/png")


@app.post("/predict/explain")
async def predict_explain(
    wafer: UploadFile = File(..., description="Wafer (.png/.jpg/.npy)"),
    mask: Optional[UploadFile] = File(None, description="Optional mask (.png/.jpg/.npy)"),
    resize: int = Form(96),
):
    from fastapi import HTTPException

    mdl = load_model()
    if mdl is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_model_load_error}")

    try:
        resize_val = int(resize)
        if resize_val <= 0:
            raise ValueError("resize must be positive")

        wafer_bytes = await wafer.read()
        mask_bytes = await mask.read() if mask is not None else None

        wafer_stack = load_input_any(
            wafer_bytes=wafer_bytes,
            wafer_filename=wafer.filename,
            mask_bytes=mask_bytes,
            mask_filename=(mask.filename if mask else None),
            resize=resize_val,
        )

        wafer_np = wafer_stack[0]
        mask_np = (wafer_stack[1] > 0).astype(np.float32)

        tensor_input = _prep_from_arrays(wafer_np, mask_np, size=resize_val)
        tensor_input = tensor_input.clone().requires_grad_(True)

        target = get_gradcam_target_wafernet(mdl, "db3")
        if target is None:
            raise RuntimeError("Could not locate Grad-CAM target layer for WaferNet")

        cam_engine = GradCAM(mdl, target)
        try:
            cam, logits, act_shape = cam_engine(
                tensor_input,
                class_idx=None,
                out_size=resize_val,
            )
        finally:
            cam_engine.remove()

        probs = torch.softmax(logits, dim=1)[0]
        explained_idx = int(probs.argmax().item())

        cam_np = cam[0, 0].detach().cpu().numpy().astype(np.float32)
        processed = tensor_input[0].detach().cpu().numpy()
        wafer_proc = np.clip(processed[0], 0.0, 1.0)
        wafer_area_mask = (processed[1] > 0).astype(np.float32)

        cam_np = np.nan_to_num(cam_np, nan=0.0, neginf=0.0, posinf=0.0)
        cam_np = np.maximum(cam_np, 0.0)

        # Weight Grad-CAM towards dies that are failing (wafer value≈1.0) while
        # still keeping a small signal on marginal regions so the heatmap stays visible.
        fail_weight = np.clip((wafer_proc - 0.25) / 0.75, 0.0, 1.0)
        cam_np *= fail_weight

        inside = cam_np[fail_weight > 0]
        if inside.size > 0:
            cmn, cmx = float(inside.min()), float(inside.max())
            if cmx > cmn:
                cam_np = (cam_np - cmn) / (cmx - cmn)

        cam_np = np.power(np.clip(cam_np, 0.0, 1.0), 0.6)
        cam_np *= wafer_area_mask

        png_bytes = overlay_cam_on_wafer(wafer_proc, wafer_area_mask, cam_np, alpha=0.70)
        b64 = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")

        return {
            "predicted_index": explained_idx,
            "predicted_label": CLASS_NAMES[explained_idx],
            "overlay_png_base64": b64,
            "activation_shape": list(act_shape),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explainability failed: {type(e).__name__}: {e}")

#uvicorn app:app --reload --host 127.0.0.1 --port 8000

#uvicorn app:app --reload --host 0.0.0.0 --port 8000
