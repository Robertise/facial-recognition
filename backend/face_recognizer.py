import os, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CKPT_PATH  = Path(os.environ.get("CHECKPOINT_PATH", PROJECT_ROOT / "checkpoints" / "method3_best.pth"))
EMBED_DIM  = 512
IMG_SIZE   = 112
DEVICE     = torch.device("cpu")

def _preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(face_bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # (x / 255.0 - 0.5) / 0.5 => (x / 127.5) - 1.0
    norm = (rgb.astype(np.float32) / 127.5) - 1.0
    transposed = np.transpose(norm, (2, 0, 1))  # (3, 112, 112)
    return torch.from_numpy(transposed).unsqueeze(0).to(DEVICE)


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


# ── Recognizer ─────────────────────────────────────────────────────────────────

class FaceRecognizer:
    def __init__(self, threshold: float = 0.5, ckpt_path: str | Path | None = None):
        self.threshold = threshold
        # ckpt_path lets research code load another method's weights side by side
        # (e.g. the L1 transfer study); the backend uses CHECKPOINT_PATH as before.
        self.ckpt_path = Path(ckpt_path) if ckpt_path else CKPT_PATH
        self._model    = None
        self._load_model()

    def _load_model(self):
        CKPT_PATH = self.ckpt_path
        if not CKPT_PATH.exists():
            print(f"[FaceRecognizer] WARNING: No checkpoint found at {CKPT_PATH}. Using random weights.")
            self._model = FaceCNN_ArcFace(num_classes=4000, embed_dim=EMBED_DIM).to(DEVICE)
            self._model.eval()
            return

        state = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
        if any(k.startswith("head.") for k in state.keys()):
            self._model = FaceCNN_ArcFace(num_classes=4000, embed_dim=EMBED_DIM).to(DEVICE)
        else:
            self._model = FaceCNN_Classifier(num_classes=4000, embed_dim=EMBED_DIM).to(DEVICE)

        self._model.load_state_dict(state, strict=False)
        self._model.eval()
        print(f"[FaceRecognizer] Loaded checkpoint: {CKPT_PATH}")

    @torch.no_grad()
    def extract_embedding(self, face_bgr: np.ndarray) -> np.ndarray:
        """Extract 512-dim L2-normalized embedding from face crop (BGR)."""
        tensor = _preprocess_face(face_bgr)
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
