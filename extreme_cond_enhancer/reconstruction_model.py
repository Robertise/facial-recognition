"""Optional adapter for a local enhancement checkpoint.

No checkpoint is downloaded or loaded unless reconstruction is explicitly
enabled. Two checkpoint formats are supported (``LIGHTING_ENHANCER_MODEL_ARCH``):

* ``torchscript``: any TorchScript module whose input/output is an RGB float
  tensor in ``[0, 1]`` shaped ``NCHW``.
* ``zero_dce_pp``: a Zero-DCE++ state dict (the official ``Epoch99.pth`` or one
  produced by ``train_reconstruction.py``), loaded with ``weights_only=True``.
"""
from __future__ import annotations

from typing import Protocol
import cv2
import numpy as np

from .config import LightingEnhancerConfig

SUPPORTED_ARCHS = ("torchscript", "zero_dce_pp")


class ReconstructionModel(Protocol):
    def is_available(self) -> bool: ...
    def enhance(self, image_bgr: np.ndarray) -> np.ndarray: ...


class TorchScriptReconstructionModel:
    """Lazy model adapter; failures leave it unavailable for this process."""
    def __init__(self, config: LightingEnhancerConfig) -> None:
        self.config = config
        self._model = None
        self._load_attempted = False
        self.load_error: str | None = None

    def _load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        checkpoint = self.config.checkpoint
        if not self.config.enable_reconstruction:
            return
        if checkpoint is None or not checkpoint.is_file():
            self.load_error = f"checkpoint not found: {checkpoint}"
            return
        if self.config.model_arch not in SUPPORTED_ARCHS:
            self.load_error = f"unsupported model arch: {self.config.model_arch}"
            return
        try:
            import torch
            device_name = self.config.device
            if device_name == "auto":
                device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self._device = torch.device(device_name)
            if self.config.model_arch == "zero_dce_pp":
                from .zero_dce import ZeroDCEPPInference, load_zero_dce_pp
                net = load_zero_dce_pp(checkpoint, self.config.model_scale_factor, map_location="cpu")
                self._model = ZeroDCEPPInference(net).to(self._device).eval()
            else:
                self._model = torch.jit.load(str(checkpoint), map_location=self._device).eval()
        except Exception as exc:  # keep inference usable without the model
            self._model = None
            self.load_error = f"{type(exc).__name__}: {exc}"

    def is_available(self) -> bool:
        self._load()
        return self._model is not None

    def enhance(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("No configured reconstruction model is available")
        import torch
        height, width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div(255.0).unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        array = output.squeeze(0).detach().float().cpu().numpy().transpose(1, 2, 0)
        if array.shape[:2] != (height, width):
            array = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(np.clip(array * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
