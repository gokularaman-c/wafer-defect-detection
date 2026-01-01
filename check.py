import numpy as np
from PIL import Image

def load_png_as_float(path):
    return np.array(Image.open(path).convert("F"), dtype=np.float32)

w_png = load_png_as_float("notebooks/test samples(2-per-class)/wafer_Center_01.png")
m_png = load_png_as_float("notebooks/test samples(2-per-class)/mask_Center_01.png")

print("wafer PNG unique:", np.unique(w_png))
print("mask  PNG unique:", np.unique(m_png))

# emulate server normalization exactly
wafer = w_png.copy()
if wafer.max() > 1.0:
    wafer = wafer / 2.0
wafer = wafer.clip(0.0, 1.0)

mask = (m_png > 0.5).astype(np.float32)

inside = mask > 0.5
frac_fail = (wafer[inside] == 1.0).mean()    # fraction of fails seen by the model
print("fail fraction inside wafer (after server norm):", frac_fail)