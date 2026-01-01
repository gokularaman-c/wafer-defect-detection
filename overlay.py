from __future__ import annotations

import io
import base64
from typing import Tuple

import numpy as np
from PIL import Image

# Reuse the discrete viridis palette from preview rendering.
_PALETTE = np.array([
    [ 68,   1,  84],  # background / 0.0
    [ 31, 158, 137],  # mid / 0.5
    [253, 231,  37],  # fail / 1.0
], dtype=np.uint8)
_BACKGROUND_RGB = np.array([10, 10, 16], dtype=np.uint8)

# Simple three-stop heatmap (black -> royal blue -> amber -> white)
_HEAT_R = np.array([0,  64, 255, 255], dtype=np.float32)
_HEAT_G = np.array([0,  72, 160, 255], dtype=np.float32)
_HEAT_B = np.array([0, 180,  40, 255], dtype=np.float32)
_HEAT_POS = np.array([0.0, 0.35, 0.7, 1.0], dtype=np.float32)


def _snap_wafer(values: np.ndarray) -> np.ndarray:
    w = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    w = np.clip(w, 0.0, 1.0)
    w = np.round(w * 2.0) / 2.0
    return w


def _wafer_to_rgb(wafer: np.ndarray, mask: np.ndarray) -> np.ndarray:
    wafer = _snap_wafer(wafer)
    idx = (wafer * 2).astype(np.int64)
    rgb = _PALETTE[idx]
    rgb = rgb.copy()
    rgb[mask <= 0] = _BACKGROUND_RGB
    return rgb


def _apply_heatmap(cam: np.ndarray, mask: np.ndarray, alpha_scale: float) -> np.ndarray:
    c = np.clip(cam.astype(np.float32), 0.0, 1.0)
    c *= mask.astype(np.float32)
    flat = c.reshape(-1)
    r = np.interp(flat, _HEAT_POS, _HEAT_R).reshape(c.shape)
    g = np.interp(flat, _HEAT_POS, _HEAT_G).reshape(c.shape)
    b = np.interp(flat, _HEAT_POS, _HEAT_B).reshape(c.shape)
    alpha = (c ** 0.7) * (alpha_scale * 255.0)
    heat = np.stack([r, g, b, alpha], axis=-1).astype(np.uint8)
    return heat


def overlay_cam_on_wafer(
    wafer: np.ndarray,
    mask: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
) -> bytes:
    if wafer.shape != mask.shape or wafer.shape != cam.shape:
        raise ValueError("wafer, mask, and cam must share the same spatial shape")

    mask_bin = (mask > 0.5).astype(np.float32)
    base_rgb = _wafer_to_rgb(wafer, mask_bin)
    base_rgba = np.concatenate([
        base_rgb,
        np.where(mask_bin > 0, 255, 0).astype(np.uint8)[..., None]
    ], axis=2)

    heat_rgba = _apply_heatmap(cam, mask_bin, alpha_scale=max(0.0, min(alpha, 1.0)))

    base_img = Image.fromarray(base_rgba, mode="RGBA")
    heat_img = Image.fromarray(heat_rgba, mode="RGBA")

    composed = Image.alpha_composite(base_img, heat_img)

    buf = io.BytesIO()
    composed.save(buf, format="PNG")
    return buf.getvalue()


def overlay_png_base64(wafer: np.ndarray, mask: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> str:
    png_bytes = overlay_cam_on_wafer(wafer, mask, cam, alpha=alpha)
    return base64.b64encode(png_bytes).decode("ascii")
