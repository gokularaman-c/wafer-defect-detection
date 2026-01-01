from __future__ import annotations

import torch.nn as nn


def get_gradcam_target_wafernet(net, variant: str = "db4"):
    """
    Returns the Conv2d module to hook for Grad-CAM.
    variant:
      - "db4": last conv in denseblock4  (~3x3 map on 96x96)
      - "db3": last conv in denseblock3  (~6x6 map on 96x96)
      - "auto": scan features and return the last Conv2d found
    """
    f = net.features  # torchvision densenet121 features
    if variant == "db4":
        return f.denseblock4.denselayer16.conv2
    if variant == "db3":
        return f.denseblock3.denselayer24.conv2
    last = None
    for _, m in f.named_modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last
