from __future__ import annotations

import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._save_act)
        self._bh = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _module, _inputs, output):
        self.activations = output.detach()

    def _save_grad(self, _module, _grad_inputs, grad_outputs):
        self.gradients = grad_outputs[0].detach()

    def remove(self):
        self._fh.remove()
        self._bh.remove()

    def __call__(
        self,
        x: torch.Tensor,
        class_idx: int | None = None,
        out_size: int = 96,
    ):
        """
        x: (B,2,96,96)
        returns: cam (B,1,out_size,out_size) in [0,1], logits, act_shape
        """
        self.activations = None
        self.gradients = None

        was_training = self.model.training
        self.model.eval()

        if not x.requires_grad:
            x = x.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()

        A = self.activations
        G = self.gradients
        if A is None or G is None:
            raise RuntimeError("GradCAM hooks failed to capture activations/gradients.")

        W = G.mean((2, 3), keepdim=True)
        cam = torch.relu((W * A).sum(1, keepdim=True))
        cam = torch.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
        cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)

        cam_min = cam.amin(dim=(-2, -1), keepdim=True)
        cam_max = cam.amax(dim=(-2, -1), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        if was_training:
            self.model.train()
        return cam, logits, A.shape
