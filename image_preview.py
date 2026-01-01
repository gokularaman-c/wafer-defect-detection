from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image

# ─── Palette and styling ───────────────────────────────────────────────────────
# Matches the discrete viridis anchors used throughout the project.
_PALETTE = np.array([
    [ 68,   1,  84],  # 0.0  -> purple  (background)
    [ 31, 158, 137],  # 0.5  -> teal    (pass)
    [253, 231,  37],  # 1.0  -> yellow  (fail)
], dtype=np.uint8)
_BACKGROUND_RGB = np.array([10, 10, 16], dtype=np.uint8)
_OUTSIDE_ALPHA = 96


def _discretize_wafer(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp to [0,1], snap to {0,0.5,1} and return (values, palette_index)."""
    w = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    w = np.clip(w, 0.0, 1.0)
    w = np.round(w * 2.0) / 2.0
    idx = (w * 2.0).astype(np.int64)  # 0,1,2 mapping into palette
    return w, idx


def render_png_bytes(wafer: np.ndarray, mask: np.ndarray, scale: int = 1) -> bytes:
    """Render wafer + mask into PNG bytes with nearest-neighbour scaling."""
    if wafer.shape != mask.shape:
        raise ValueError(f"wafer shape {wafer.shape} must match mask shape {mask.shape}")
    if scale < 1:
        raise ValueError("scale must be >= 1")

    wafer = wafer.astype(np.float32, copy=False)
    mask = (mask > 0.5).astype(np.uint8)

    _, palette_idx = _discretize_wafer(wafer)
    h, w = wafer.shape

    rgb = _PALETTE[palette_idx]
    alpha = np.where(mask == 1, 255, _OUTSIDE_ALPHA).astype(np.uint8)
    rgb[mask == 0] = _BACKGROUND_RGB

    rgba = np.concatenate([rgb, alpha.reshape(h, w, 1)], axis=2)
    img = Image.fromarray(rgba, mode="RGBA")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), resample=Image.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
