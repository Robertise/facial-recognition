import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CKPT_PATH  = Path(os.environ.get("CHECKPOINT_PATH", PROJECT_ROOT / "checkpoints" / "method3_best.pth"))
EMBED_DIM  = 512
IMG_SIZE   = 112
DEVICE     = torch.device("cpu")


@dataclass(frozen=True)
class Preprocess:
    """Input contract of a recogniser: square side and per-channel RGB normalisation on [0, 1]."""
    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


# Methods 1-4 (FaceCNN): 112 x 112, (x/255 - 0.5)/0.5, i.e. [-1, 1] (training_max.ipynb val_transform).
CNN_PREPROCESS = Preprocess(IMG_SIZE, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# Method 5 (ViT-B/16): 224 x 224, ImageNet statistics (training_max.ipynb val_transform_vit).
VIT_PREPROCESS = Preprocess(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _preprocess_face(face_bgr: np.ndarray, pre: Preprocess = CNN_PREPROCESS) -> torch.Tensor:
    resized = cv2.resize(face_bgr, (pre.size, pre.size))        # INTER_LINEAR
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm = (rgb - np.array(pre.mean, np.float32)) / np.array(pre.std, np.float32)
    transposed = np.transpose(norm, (2, 0, 1))                   # (3, size, size)
    return torch.from_numpy(np.ascontiguousarray(transposed)).unsqueeze(0).to(DEVICE)


def _research_mode() -> bool:
    return os.environ.get("INNOVATION_RESEARCH_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


# ── CNN Architecture (same as Task 1) ────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        layers = [nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True)]
        if pool: layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.BatchNorm2d(ch),
        )
    def forward(self, x): return F.relu(self.net(x)+x, True)

class FaceCNN(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,32,True), ConvBlock(32,64,True), ResBlock(64),
            ConvBlock(64,128,True), ResBlock(128), ConvBlock(128,256,True),
            ResBlock(256), ConvBlock(256,512,True), nn.AdaptiveAvgPool2d(1),
        )
        self.emb_head = nn.Sequential(nn.Flatten(), nn.Linear(512,embed_dim), nn.BatchNorm1d(embed_dim))
    def forward(self, x):
        return F.normalize(self.emb_head(self.features(x)), p=2, dim=1)

class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim=512, num_classes=4000, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th    = math.cos(math.pi - m)
        self.mm    = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels=None):
        w_norm    = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(embeddings, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return self.s * cos_theta
        sin_theta   = torch.sqrt(1.0 - cos_theta ** 2)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m,
                                  cos_theta - self.mm)
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        return (one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta) * self.s

class FaceCNN_ArcFace(nn.Module):
    def __init__(self, num_classes=4000, embed_dim=512, s=64.0, m=0.5):
        super().__init__()
        self.backbone = FaceCNN(embed_dim)
        self.head     = ArcFaceHead(embed_dim, num_classes, s=s, m=m)
    def forward(self, x, labels=None):
        emb    = self.backbone(x)
        logits = self.head(emb, labels)
        return logits, emb
    def get_embedding(self, x): return self.backbone(x)

class FaceCNN_Classifier(nn.Module):
    def __init__(self, num_classes=4000, embed_dim=512):
        super().__init__()
        self.backbone   = FaceCNN(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        emb = self.backbone(x)
        return self.classifier(emb), emb
    def get_embedding(self, x): return self.backbone(x)

class FaceCNN_Triplet(nn.Module):
    """Methods 2 and 4: the backbone alone (triplet loss, no classification head)."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.backbone = FaceCNN(embed_dim)
    def forward(self, x): return self.backbone(x)
    def get_embedding(self, x): return self.backbone(x)


# ── ViT Architecture (Method 5, training_max.ipynb FaceViT_ArcFace) ────────────

class FaceViT_ArcFace(nn.Module):
    """ViT-B/16 (timm, no classifier) -> Linear(768, 512, no bias) -> BN -> L2; ArcFace weight.

    Needs `timm` (innovation/requirements.txt). Built with pretrained=False:
    every weight comes from the checkpoint.
    """
    def __init__(self, num_classes=4000, embed_dim=512):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("method5 (ViT-B/16) needs timm: pip install -r innovation/requirements.txt") from e
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.emb_head = nn.Sequential(nn.Linear(self.backbone.num_features, embed_dim, bias=False),
                                      nn.BatchNorm1d(embed_dim))
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
    def get_embedding(self, x):
        return F.normalize(self.emb_head(self.backbone(x)), p=2, dim=1)
    def forward(self, x):
        emb = self.get_embedding(x)
        return F.linear(emb, F.normalize(self.weight, p=2, dim=1)), emb


def _unwrap_state(obj) -> dict:
    """A plain state_dict from a checkpoint file (also {'state_dict': ...} and DataParallel 'module.')."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    state = dict(obj)
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    return state


def build_model_for_state(state: dict) -> tuple[nn.Module, Preprocess, str]:
    """The architecture a state_dict belongs to, its preprocessing and a short name."""
    keys = set(state)
    if {"backbone.cls_token", "backbone.pos_embed"} & keys:
        return FaceViT_ArcFace(state["weight"].shape[0], EMBED_DIM), VIT_PREPROCESS, "FaceViT_ArcFace"
    if "head.weight" in keys:
        return FaceCNN_ArcFace(state["head.weight"].shape[0], EMBED_DIM), CNN_PREPROCESS, "FaceCNN_ArcFace"
    if "classifier.weight" in keys:
        return FaceCNN_Classifier(state["classifier.weight"].shape[0], EMBED_DIM), CNN_PREPROCESS, "FaceCNN_Classifier"
    if any(k.startswith("backbone.features.") for k in keys):
        return FaceCNN_Triplet(EMBED_DIM), CNN_PREPROCESS, "FaceCNN_Triplet"
    raise ValueError(f"unknown checkpoint layout; first keys: {sorted(keys)[:8]}")


# ── Recognizer ─────────────────────────────────────────────────────────────────

class FaceRecognizer:
    def __init__(self, threshold: float = 0.5, ckpt_path: str | Path | None = None,
                 strict: bool | None = None):
        """
        ckpt_path: another method's weights (e.g. the L1 transfer study); default CHECKPOINT_PATH.
        strict: stop on a missing checkpoint or on any missing/unexpected key, instead of
            running with partly random weights. Defaults to on in research mode
            (INNOVATION_RESEARCH_MODE); research scripts always pass True.
        """
        self.threshold = threshold
        self.ckpt_path = Path(ckpt_path) if ckpt_path else CKPT_PATH
        self.strict = _research_mode() if strict is None else bool(strict)
        self._model = None
        self.preprocess = CNN_PREPROCESS
        self.arch = None
        self.load_report: dict = {}
        self._load_model()

    def _load_model(self):
        if not self.ckpt_path.exists():
            if self.strict:
                raise FileNotFoundError(f"[FaceRecognizer] checkpoint not found: {self.ckpt_path}")
            print(f"[FaceRecognizer] WARNING: No checkpoint found at {self.ckpt_path}. Using random weights.")
            self._model = FaceCNN_ArcFace(num_classes=4000, embed_dim=EMBED_DIM).to(DEVICE).eval()
            self.arch = "FaceCNN_ArcFace (random)"
            self.load_report = {"random_weights": True}
            return

        state = _unwrap_state(torch.load(str(self.ckpt_path), map_location=DEVICE, weights_only=False))
        model, self.preprocess, self.arch = build_model_for_state(state)
        res = model.load_state_dict(state, strict=False)
        self.load_report = {"arch": self.arch, "keys": len(state), "missing_keys": list(res.missing_keys),
                            "unexpected_keys": list(res.unexpected_keys), "input_size": self.preprocess.size}
        if res.missing_keys or res.unexpected_keys:
            msg = (f"[FaceRecognizer] {self.ckpt_path} -> {self.arch}: "
                   f"{len(res.missing_keys)} missing keys {res.missing_keys[:5]}, "
                   f"{len(res.unexpected_keys)} unexpected keys {res.unexpected_keys[:5]}")
            if self.strict:
                raise RuntimeError(msg + " (strict load: stopping)")
            print("WARNING: " + msg)
        self._model = model.to(DEVICE).eval()
        print(f"[FaceRecognizer] Loaded checkpoint: {self.ckpt_path} ({self.arch}, "
              f"{self.preprocess.size}x{self.preprocess.size})")

    @torch.no_grad()
    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Extract 512-dim L2-normalized embedding from face crop (BGR), with this model's preprocessing."""
        tensor = _preprocess_face(face_bgr, self.preprocess)
        emb    = self._model.get_embedding(tensor).cpu().numpy().squeeze()
        return emb  # shape (512,)


    def find_match(self, query_emb: np.ndarray, db_embeddings: list[dict]):
        """
        Compare query embedding against all DB embeddings.
        db_embeddings: list of {employee_id, name, angle, embedding}

        Returns dict with matched=True/False, best match info.
        """
        if not db_embeddings:
            return {"matched": False, "employee_id": None, "name": None, "similarity": 0.0}

        # Group by employee, take max similarity per employee
        scores: dict[str, dict] = {}
        for entry in db_embeddings:
            sim = float(np.dot(query_emb, entry["embedding"]))  # cosine (L2-normed)
            eid = entry["employee_id"]
            if eid not in scores or sim > scores[eid]["similarity"]:
                scores[eid] = {
                    "employee_id": eid,
                    "name": entry["name"],
                    "similarity": sim,
                }

        best = max(scores.values(), key=lambda x: x["similarity"])
        best["matched"] = best["similarity"] >= self.threshold
        return best
