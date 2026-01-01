from __future__ import annotations
import io
import numpy as np
from PIL import Image

# Viridis anchor colors you see in your PNGs (purple, teal, yellow)
_VIRIDIS_RGB = np.array([
    [ 68,   1,  84],   # purple  -> background -> 0.0
    [ 31, 158, 137],   # teal    -> mid (good) -> 0.5
    [253, 231,  37],   # yellow  -> high (fail)-> 1.0
], dtype=np.float32)
_VIRIDIS_VALUES = np.array([0.0, 0.5, 1.0], dtype=np.float32)

def _resize_nn(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.NEAREST)

def _rgb_to_discrete_wafer(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Map each RGB pixel to the nearest of 3 viridis anchors (purple/teal/yellow)
    and return a wafer array with values in {0.0, 0.5, 1.0}.
    Works even if PNG has slight antialiasing: nearest-color snapping.
    """
    h, w, _ = arr_rgb.shape
    flat = arr_rgb.reshape(-1, 3).astype(np.float32)
    # squared distances to the 3 anchors, shape (N, 3)
    d2 = ((flat[:, None, :] - _VIRIDIS_RGB[None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)  # 0..2
    wafer = _VIRIDIS_VALUES[idx].reshape(h, w)
    return wafer

def decode_wafer_png(file_bytes: bytes, resize: int = 96) -> np.ndarray:
    """
    Returns x with shape (2, resize, resize): [wafer, mask] float32
    wafer ∈ {0.0, 0.5, 1.0}, mask ∈ {0.0, 1.0}
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = _resize_nn(img, resize)
    arr = np.array(img, dtype=np.uint8)
    wafer = _rgb_to_discrete_wafer(arr).astype(np.float32)
    mask  = (wafer > 0.0).astype(np.float32)   # background (purple=0.0) excluded
    return np.stack([wafer, mask], axis=0).astype(np.float32)

def decode_mask_png(file_bytes: bytes, resize: int = 96) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = _resize_nn(img, resize)
    m = (np.array(img, dtype=np.uint8) > 0).astype(np.float32)
    return m

def load_input_any(
    wafer_bytes: bytes, wafer_filename: str,
    mask_bytes: bytes | None, mask_filename: str | None,
    resize: int = 96
) -> np.ndarray:
    """
    Accepts wafer as .npy or .png/.jpg and optional mask as .npy or .png/.jpg.
    Always returns (2, resize, resize) float32.
    """
    import io, numpy as np
    fn = wafer_filename.lower()
    if fn.endswith(".npy"):
        arr = np.load(io.BytesIO(wafer_bytes), allow_pickle=False)
        if arr.ndim == 2:
            wafer = arr.astype(np.float32)
            mask  = (wafer > 0.0).astype(np.float32)
            x = np.stack([wafer, mask], axis=0)
        elif arr.ndim == 3 and arr.shape[0] in (1,2):
            if arr.shape[0] == 1:
                wafer = arr[0].astype(np.float32)
                mask  = (wafer > 0.0).astype(np.float32)
                x = np.stack([wafer, mask], axis=0)
            else:
                x = arr.astype(np.float32)
        else:
            raise ValueError(f"Unexpected NPY shape {arr.shape}. Expected (H,W), (1,H,W) or (2,H,W).")
        # Resize if needed
        if x.shape[-1] != resize or x.shape[-2] != resize:
            from PIL import Image
            def rz(a):  # nearest neighbor for both wafer and mask
                im = Image.fromarray((a*255).astype(np.uint8))
                im = _resize_nn(im, resize)
                return (np.array(im) / 255.0).astype(np.float32)
            x = np.stack([rz(x[0]), (rz(x[1]) > 0).astype(np.float32)], axis=0)
    else:
        x = decode_wafer_png(wafer_bytes, resize=resize)

    # Optional explicit mask override
    if mask_bytes is not None and mask_filename is not None:
        mf = mask_filename.lower()
        if mf.endswith(".npy"):
            m = np.load(io.BytesIO(mask_bytes), allow_pickle=False).astype(np.float32)
            if m.ndim == 3 and m.shape[0] == 1: m = m[0]
        else:
            m = decode_mask_png(mask_bytes, resize=resize)
        if m.shape != (resize, resize):
            from PIL import Image
            im = Image.fromarray((m > 0).astype(np.uint8)*255)
            im = _resize_nn(im, resize)
            m = (np.array(im) > 0).astype(np.float32)
        x[1] = (m > 0).astype(np.float32)
    return x.astype(np.float32)
