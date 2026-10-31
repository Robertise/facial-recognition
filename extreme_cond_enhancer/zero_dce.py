"""Zero-DCE++ curve-estimation network and its zero-reference training losses.

Reference: Li, Guo, Loy, "Learning to Enhance Low-Light Image via Zero-Reference
Deep Curve Estimation", TPAMI 2021 (https://github.com/Li-Chongyi/Zero-DCE_extension).

The network predicts a per-pixel curve parameter map ``A`` in ``[-1, 1]`` and
applies the quadratic light-enhancement curve eight times:
``x <- x + A * (x**2 - x)``  (equivalently ``x - A * x * (1 - x)``).
Each step is monotonic and keeps ``x`` inside ``[0, 1]``, so it can only remap
exposure; it cannot invent facial structure. Layer names match the official
release, so its ``Epoch99.pth`` state dict loads directly (~10k parameters).
"""
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.depth_conv = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch)
        self.point_conv = nn.Conv2d(in_ch, out_ch, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.point_conv(self.depth_conv(x))


class ZeroDCEPP(nn.Module):
    """Returns ``(enhanced, curve_map)`` for RGB ``NCHW`` input in ``[0, 1]``."""

    def __init__(self, scale_factor: int = 1, features: int = 32) -> None:
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.iterations = 8
        self.e_conv1 = _DepthwiseSeparableConv(3, features)
        self.e_conv2 = _DepthwiseSeparableConv(features, features)
        self.e_conv3 = _DepthwiseSeparableConv(features, features)
        self.e_conv4 = _DepthwiseSeparableConv(features, features)
        self.e_conv5 = _DepthwiseSeparableConv(features * 2, features)
        self.e_conv6 = _DepthwiseSeparableConv(features * 2, features)
        self.e_conv7 = _DepthwiseSeparableConv(features * 2, 3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_down = x
        if self.scale_factor > 1:
            x_down = F.interpolate(x, scale_factor=1.0 / self.scale_factor, mode="bilinear",
                                   align_corners=False)
        x1 = F.relu(self.e_conv1(x_down))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        curve = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor > 1:
            curve = F.interpolate(curve, size=(x.shape[2], x.shape[3]), mode="bilinear",
                                  align_corners=True)
        enhanced = x
        for _ in range(self.iterations):
            enhanced = enhanced + curve * (torch.pow(enhanced, 2) - enhanced)
        return enhanced, curve


class ZeroDCEPPInference(nn.Module):
    """Image-only wrapper matching the TorchScript adapter contract."""

    def __init__(self, net: ZeroDCEPP) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enhanced, _ = self.net(x)
        return enhanced.clamp(0.0, 1.0)


def load_zero_dce_pp(path: Union[str, Path], scale_factor: int = 1,
                     map_location: Union[str, torch.device] = "cpu") -> ZeroDCEPP:
    """Load an official or locally trained Zero-DCE++ state dict (never downloads)."""
    state = torch.load(str(path), map_location=map_location, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    net = ZeroDCEPP(scale_factor=scale_factor)
    net.load_state_dict(state, strict=True)
    return net.eval()


def export_torchscript(net: ZeroDCEPP, out_path: Union[str, Path]) -> Path:
    """Script the image-only wrapper so it plugs into ``TorchScriptReconstructionModel``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(ZeroDCEPPInference(net.cpu().eval()))
    scripted.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# Zero-reference losses (no ground-truth "well lit" image is needed)
# ---------------------------------------------------------------------------

def color_constancy_loss(enhanced: torch.Tensor) -> torch.Tensor:
    """Grey-world prior: penalise differences between the channel means."""
    mr, mg, mb = torch.mean(enhanced, dim=(2, 3)).unbind(1)
    return torch.sqrt((mr - mg) ** 4 + (mr - mb) ** 4 + (mb - mg) ** 4 + 1e-12).mean()


def exposure_loss(enhanced: torch.Tensor, target: float = 0.6, patch: int = 16) -> torch.Tensor:
    """Pull the mean intensity of each ``patch x patch`` region towards ``target``."""
    mean = F.avg_pool2d(enhanced.mean(1, keepdim=True), patch)
    return torch.mean((mean - target) ** 2)


class SpatialConsistencyLoss(nn.Module):
    """Keep neighbouring-region contrast of the output equal to the input's."""

    def __init__(self) -> None:
        super().__init__()
        kernels = torch.zeros(4, 1, 3, 3)
        for i, (r, c) in enumerate(((1, 0), (1, 2), (0, 1), (2, 1))):  # left, right, up, down
            kernels[i, 0, 1, 1] = 1.0
            kernels[i, 0, r, c] = -1.0
        self.register_buffer("kernels", kernels)

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        org = F.avg_pool2d(original.mean(1, keepdim=True), 4)
        enh = F.avg_pool2d(enhanced.mean(1, keepdim=True), 4)
        d_org = F.conv2d(org, self.kernels, padding=1)
        d_enh = F.conv2d(enh, self.kernels, padding=1)
        return torch.mean(torch.sum((d_org - d_enh) ** 2, dim=1))


def illumination_smoothness_loss(curve: torch.Tensor) -> torch.Tensor:
    """Total variation of the curve map so neighbouring pixels get similar curves."""
    n, _, h, w = curve.shape
    h_tv = torch.pow(curve[:, :, 1:, :] - curve[:, :, :-1, :], 2).sum()
    w_tv = torch.pow(curve[:, :, :, 1:] - curve[:, :, :, :-1], 2).sum()
    return 2.0 * (h_tv / max((h - 1) * w, 1) + w_tv / max(h * (w - 1), 1)) / n


class ZeroReferenceLoss(nn.Module):
    """Weighted Zero-DCE++ objective; defaults match the official training script."""

    def __init__(self, exposure_target: float = 0.6, w_tv: float = 1600.0, w_spa: float = 1.0,
                 w_col: float = 5.0, w_exp: float = 10.0) -> None:
        super().__init__()
        self.spatial = SpatialConsistencyLoss()
        self.exposure_target = exposure_target
        self.weights = (w_tv, w_spa, w_col, w_exp)

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor,
                curve: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        w_tv, w_spa, w_col, w_exp = self.weights
        parts = {
            "tv": illumination_smoothness_loss(curve),
            "spa": self.spatial(original, enhanced),
            "col": color_constancy_loss(enhanced),
            "exp": exposure_loss(enhanced, self.exposure_target),
        }
        total = w_tv * parts["tv"] + w_spa * parts["spa"] + w_col * parts["col"] + w_exp * parts["exp"]
        return total, {k: float(v.detach()) for k, v in parts.items()}
