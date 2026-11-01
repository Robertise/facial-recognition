"""
pipeline.py – the recognition step of check-in as a differentiable function:
face crop (BGR, 0..255) -> side x side -> RGB -> normalise -> embedding.

Mirrors backend.face_recognizer._preprocess_face with the recogniser's own
`preprocess` (cv2.resize INTER_LINEAR, BGR->RGB, (x/255 - mean)/std):
112 and [-1, 1] for the FaceCNN methods 1-4, 224 and ImageNet statistics for
the ViT method 5. Uses the recogniser's own weights. `parity` measures how
close it is to the backend on real crops.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class TorchPipeline:
    def __init__(self, recognizer, device: str = "cpu"):
        self.recognizer = recognizer
        self.model = recognizer._model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        pre = recognizer.preprocess
        self.size = pre.size
        self.mean = torch.tensor(pre.mean, dtype=torch.float32, device=device).view(3, 1, 1)
        self.std = torch.tensor(pre.std, dtype=torch.float32, device=device).view(3, 1, 1)

    def preprocess(self, crop_bgr: torch.Tensor) -> torch.Tensor:
        """(3, h, w) BGR in 0..255 -> (3, side, side) RGB, normalised as the model was trained."""
        x = F.interpolate(crop_bgr[None], size=(self.size, self.size), mode="bilinear",
                          align_corners=False, antialias=False)[0]
        return (x.flip(0) / 255.0 - self.mean) / self.std

    def embed(self, crops_bgr: list[torch.Tensor]) -> torch.Tensor:
        """Embeddings (n, 512), L2-normalised, for crops of any sizes."""
        x = torch.stack([self.preprocess(c.to(self.device)) for c in crops_bgr])
        return self.model.get_embedding(x)


def to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(img_bgr.transpose(2, 0, 1))).float()


@torch.no_grad()
def parity(pipeline: TorchPipeline, crops_bgr: list[np.ndarray]) -> np.ndarray:
    """Cosine between this pipeline's embedding and the backend's for each crop (target ≥ 0.999)."""
    ours = pipeline.embed([to_tensor(c) for c in crops_bgr]).cpu().numpy()
    theirs = np.stack([pipeline.recognizer.extract_embedding(c) for c in crops_bgr])
    ours /= np.linalg.norm(ours, axis=1, keepdims=True)
    theirs /= np.linalg.norm(theirs, axis=1, keepdims=True)
    return np.sum(ours * theirs, axis=1)


def main():
    """python -m innovation.attacks.adv_glasses.pipeline --frames <folder> [--checkpoint ...]: parity on real crops."""
    import argparse
    from pathlib import Path

    import cv2

    from backend.face_recognizer import FaceRecognizer
    from innovation.attacks.adv_glasses import faces

    ap = argparse.ArgumentParser(description="Check the differentiable pipeline against the backend (target: cosine >= 0.999).")
    ap.add_argument("--frames", required=True, help="a folder of captured frames")
    ap.add_argument("--checkpoint", default="checkpoints/method3_best.pth")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    rec = FaceRecognizer(ckpt_path=args.checkpoint, strict=True)
    det = faces.FaceDetector()
    crops = []
    for f in faces.list_images(Path(args.frames)):
        img = cv2.imread(str(f))
        bbox = det.detect(img) if img is not None else None
        if bbox is not None:
            face, _ = faces.enhance(faces.crop(img, bbox))
            crops.append(face)
        if len(crops) >= args.n:
            break
    cos = parity(TorchPipeline(rec), crops)
    print(f"model {rec.arch} input {rec.preprocess.size}  crops {len(cos)}  cosine min {cos.min():.5f}  mean {cos.mean():.5f}  pass {bool(cos.min() >= 0.999)}")


if __name__ == "__main__":
    main()
