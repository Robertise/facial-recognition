#!/usr/bin/env python3
"""
comparison_check.py
===================
So sánh độ giống nhau của 2 ảnh khuôn mặt bằng mô hình Face Classification
(Method 3: FaceCNN + ArcFace loss trong embeddings / checkpoints).

Mặc định:
  - Quét 2 ảnh trong thư mục `comparison_check/`
  - Tự động phát hiện & crop khuôn mặt (MediaPipe Face Detection, fallback nếu đã crop sẵn)
  - Trích xuất đặc trưng embedding 512 chiều (L2-normalized) bằng Method 3 (method3_best.pth)
  - Tính toán Cosine Similarity, Euclidean Distance & dự đoán cùng người / khác người
  - Xuất ảnh trực quan kết quả so sánh tại `comparison_check/result.png`

Cách sử dụng:
  # So sánh 2 ảnh có sẵn trong folder comparison_check:
  python comparison_check.py

  # So sánh 2 ảnh cụ thể:
  python comparison_check.py --img1 path/to/face1.jpg --img2 path/to/face2.jpg

  # Thay đổi ngưỡng so sánh (mặc định: 0.163 - ngưỡng tối ưu EER / Youden's J của Method 3):
  python comparison_check.py --threshold 0.20

  # Hiển thị cửa sổ kết quả:
  python comparison_check.py --show
"""

import os
import sys
import math
import warnings
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

# Tắt cảnh báo từ protobuf và absl của bên thứ ba
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Đường dẫn mặc định ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
CKPT_PATH    = PROJECT_ROOT / "checkpoints" / "method3_best.pth"
EMBED_CACHE  = PROJECT_ROOT / "embeddings" / "method3_val.npz"
DEFAULT_DIR  = PROJECT_ROOT / "comparison_check"

DEFAULT_IMG_SIZE  = 112
DEFAULT_EMBED_DIM = 512
DEFAULT_CLASSES   = 4000
DEFAULT_THRESHOLD = 0.163  # Ngưỡng tối ưu tính từ tập validation của Method 3 (EER ~12.78%)

VALID_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ═════════════════════════════════════════════════════════════════════════════
#  KIẾN TRÚC MÔ HÌNH METHOD 3: FaceCNN + ArcFace
# ═════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.net(x) + x, inplace=True)


class FaceCNN(nn.Module):
    """Backbone trích xuất đặc trưng khuôn mặt (L2-normalized 512D embedding)."""
    def __init__(self, embed_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ResBlock(64),
            ConvBlock(64, 128, pool=True),
            ResBlock(128),
            ConvBlock(128, 256, pool=True),
            ResBlock(256),
            ConvBlock(256, 512, pool=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.emb_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw_emb = self.emb_head(self.features(x))
        return F.normalize(raw_emb, p=2, dim=1)


class ArcFaceHead(nn.Module):
    """ArcFace Margin Head cho Face Classification."""
    def __init__(self, embed_dim: int, num_classes: int, s: float = 32.0, m: float = 0.35):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, emb: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = F.normalize(self.weight, p=2, dim=1)
        cos = F.linear(emb, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if labels is None:
            return self.s * cos
        sin = torch.sqrt(1.0 - cos**2)
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        oh = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1).long(), 1.0)
        return (oh * phi + (1.0 - oh) * cos) * self.s


class FaceCNN_ArcFace(nn.Module):
    """Mô hình Method 3 kết hợp FaceCNN backbone + ArcFace classification head."""
    def __init__(self, num_classes: int = DEFAULT_CLASSES, embed_dim: int = DEFAULT_EMBED_DIM, s: float = 32.0, m: float = 0.35):
        super().__init__()
        self.backbone = FaceCNN(embed_dim)
        self.head = ArcFaceHead(embed_dim, num_classes, s, m)

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.backbone(x)
        logits = self.head(emb, labels)
        return logits, emb

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ═════════════════════════════════════════════════════════════════════════════
#  BỘ PHÁT HIỆN & TIỀN XỬ LÝ KHUÔN MẶT
# ═════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    """Sử dụng MediaPipe Face Detection để phát hiện vị trí khuôn mặt."""
    def __init__(self, min_confidence: float = 0.5):
        self.detector = None
        try:
            import mediapipe as mp
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_confidence
            )
        except Exception as e:
            print(f"[!] Không thể khởi tạo MediaPipe ({e}), sẽ sử dụng ảnh gốc trực tiếp.")

    def detect(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Trả về bbox (x, y, w, h) của khuôn mặt rõ nhất hoặc None nếu không thấy."""
        if self.detector is None or img_bgr is None:
            return None

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = self.detector.process(img_rgb)
            if not results.detections:
                return None
            best = max(results.detections, key=lambda d: d.score[0])
            bb = best.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = min(w - x, int(bb.width * w))
            bh = min(h - y, int(bb.height * h))
            if bw <= 0 or bh <= 0:
                return None
            return (x, y, bw, bh)
        except Exception:
            return None


def crop_face(img_bgr: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], pad: float = 0.15) -> np.ndarray:
    """Crop khuôn mặt từ bbox có padding để lấy đầy đủ trán và cằm."""
    if bbox is None:
        return img_bgr

    x, y, bw, bh = bbox
    h, w = img_bgr.shape[:2]
    dx = int(bw * pad)
    dy = int(bh * pad)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(w, x + bw + dx)
    y2 = min(h, y + bh + dy)

    face = img_bgr[y1:y2, x1:x2]
    return face if face.size > 0 else img_bgr


def preprocess_face(face_bgr: np.ndarray, img_size: int = DEFAULT_IMG_SIZE, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Resize về 112x112, chuyển sang RGB và chuẩn hóa theo [-1, 1]."""
    resized = cv2.resize(face_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 127.5) - 1.0  # (x/255.0 - 0.5) / 0.5
    transposed = np.transpose(norm, (2, 0, 1))
    tensor = torch.from_numpy(transposed).unsqueeze(0).float().to(device)
    return tensor


# ═════════════════════════════════════════════════════════════════════════════
#  BỘ SO SÁNH KHUÔN MẶT BẰNG METHOD 3
# ═════════════════════════════════════════════════════════════════════════════

class Method3FaceClassifier:
    """Lớp nạp mô hình Method 3 (FaceCNN_ArcFace) và tính toán độ tương đồng."""
    def __init__(self, ckpt_path: Path = CKPT_PATH, device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = FaceCNN_ArcFace(num_classes=DEFAULT_CLASSES, embed_dim=DEFAULT_EMBED_DIM).to(self.device)
        self.detector = FaceDetector(min_confidence=0.5)

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"[LỖI] Không tìm thấy file checkpoint Method 3 tại: {ckpt_path}\n"
                f"Vui lòng kiểm tra lại thư mục checkpoints."
            )

        state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"[✓] Đã nạp thành công checkpoint Method 3: {ckpt_path.name} (Device: {self.device})")

    @torch.no_grad()
    def process_image(self, img_path: Path, do_crop: bool = True) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], np.ndarray, int, float]:
        """
        Đọc ảnh, phát hiện khuôn mặt, trích xuất embedding và dự đoán classification class.
        Returns:
            face_crop (BGR): ảnh khuôn mặt đã crop
            bbox: toạ độ (x, y, w, h) trên ảnh gốc (hoặc None)
            emb: vector embedding 512 chiều chuẩn hóa L2
            pred_class: ID phân loại dự đoán
            pred_conf: xác suất softmax của class dự đoán
        """
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"Không thể đọc file ảnh: {img_path}")

        bbox = None
        if do_crop:
            bbox = self.detector.detect(img_bgr)
            face_crop_img = crop_face(img_bgr, bbox)
        else:
            face_crop_img = img_bgr

        tensor = preprocess_face(face_crop_img, img_size=DEFAULT_IMG_SIZE, device=self.device)
        logits, emb_tensor = self.model(tensor)

        emb = emb_tensor.cpu().numpy().squeeze()  # shape (512,)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
        pred_class = int(np.argmax(probs))
        pred_conf = float(probs[pred_class])

        return face_crop_img, bbox, emb, pred_class, pred_conf

    def compare(self, img1_path: Path, img2_path: Path, threshold: float = DEFAULT_THRESHOLD, do_crop: bool = True) -> dict:
        """
        So sánh 2 ảnh và trả về kết quả chi tiết.
        """
        face1, bbox1, emb1, cls1, conf1 = self.process_image(img1_path, do_crop=do_crop)
        face2, bbox2, emb2, cls2, conf2 = self.process_image(img2_path, do_crop=do_crop)

        # Cosine Similarity giữa 2 vector chuẩn hóa L2 là tích vô hướng (dot product)
        cosine_sim = float(np.dot(emb1, emb2))
        euclidean_dist = float(np.linalg.norm(emb1 - emb2))

        # Quy đổi phần trăm độ tương đồng trực quan (khoảng giá trị thường gặp từ 0 đến 1)
        sim_percentage = max(0.0, min(100.0, ((cosine_sim + 1.0) / 2.0) * 100.0))

        is_match = cosine_sim >= threshold

        return {
            "img1_path": img1_path,
            "img2_path": img2_path,
            "face1": face1,
            "face2": face2,
            "bbox1": bbox1,
            "bbox2": bbox2,
            "cls1": cls1,
            "cls2": cls2,
            "conf1": conf1,
            "conf2": conf2,
            "cosine_similarity": cosine_sim,
            "euclidean_distance": euclidean_dist,
            "similarity_percentage": sim_percentage,
            "threshold": threshold,
            "is_match": is_match,
        }


# ═════════════════════════════════════════════════════════════════════════════
#  VẼ VÀ LƯU ẢNH KẾT QUẢ TRỰC QUAN
# ═════════════════════════════════════════════════════════════════════════════

def create_comparison_visualization(res: dict, out_path: Optional[Path] = None) -> np.ndarray:
    """Tạo ảnh ghép đối chiếu 2 khuôn mặt kèm bảng điểm số trực quan."""
    panel_size = 280
    f1 = cv2.resize(res["face1"], (panel_size, panel_size))
    f2 = cv2.resize(res["face2"], (panel_size, panel_size))

    # Ghép ngang 2 ảnh mặt
    combined_faces = np.hstack([f1, f2])

    banner_height = 110
    total_width = panel_size * 2
    total_height = panel_size + banner_height

    canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)
    canvas[banner_height:, :] = combined_faces

    # Màu sắc banner: Xanh lá nếu Cùng người, Đỏ nếu Khác người
    is_match = res["is_match"]
    banner_color = (34, 139, 34) if is_match else (30, 30, 200)   # BGR
    cv2.rectangle(canvas, (0, 0), (total_width, banner_height), banner_color, -1)

    # Viền ngăn cách giữa 2 ảnh
    cv2.line(canvas, (panel_size, banner_height), (panel_size, total_height), (255, 255, 255), 2)

    # Chữ hiển thị trên Banner
    title_text = "MATCH - SAME PERSON" if is_match else "NO MATCH - DIFFERENT PERSON"
    stat_text1 = f"Cosine Sim: {res['cosine_similarity']:+.4f}  |  Threshold: {res['threshold']:.3f}"
    stat_text2 = f"Similarity: {res['similarity_percentage']:.1f}%  |  Euclidean Dist: {res['euclidean_distance']:.4f}"

    cv2.putText(canvas, title_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, stat_text1, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, stat_text2, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)

    # Nhãn tên dưới mỗi ảnh
    name1 = res["img1_path"].name
    name2 = res["img2_path"].name
    cv2.rectangle(canvas, (0, total_height - 30), (panel_size, total_height), (0, 0, 0), -1)
    cv2.rectangle(canvas, (panel_size, total_height - 30), (total_width, total_height), (0, 0, 0), -1)
    cv2.putText(canvas, f"1: {name1[:18]}", (8, total_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"2: {name2[:18]}", (panel_size + 8, total_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)
        print(f"[✓] Đã lưu ảnh kết quả so sánh tại: {out_path}")

    return canvas


# ═════════════════════════════════════════════════════════════════════════════
#  HÀM TÌM ẢNH TRONG THƯ MỤC
# ═════════════════════════════════════════════════════════════════════════════

def find_images_in_folder(folder_path: Path) -> List[Path]:
    """Tìm tất cả các file ảnh hợp lệ trong thư mục."""
    if not folder_path.exists():
        return []
    files = []
    for f in sorted(folder_path.iterdir()):
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS:
            # Bỏ qua các file kết quả cũ
            if not f.stem.startswith("result") and not f.stem.startswith("comparison_result"):
                files.append(f)
    return files


# ═════════════════════════════════════════════════════════════════════════════
#  HÀM MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="So sánh độ giống nhau của 2 khuôn mặt bằng Method 3 (FaceCNN_ArcFace)."
    )
    parser.add_argument("--img1", type=str, default=None, help="Đường dẫn tới ảnh thứ nhất (tùy chọn)")
    parser.add_argument("--img2", type=str, default=None, help="Đường dẫn tới ảnh thứ hai (tùy chọn)")
    parser.add_argument("--folder", type=str, default=str(DEFAULT_DIR), help="Thư mục chứa 2 ảnh (mặc định: comparison_check)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Ngưỡng phân loại cùng người (mặc định: {DEFAULT_THRESHOLD})")
    parser.add_argument("--no-crop", action="store_true", help="Không tự động crop mặt, dùng toàn bộ ảnh gốc")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ popup OpenCV kết quả")
    parser.add_argument("--save", action="store_true", default=True, help="Lưu ảnh kết quả so sánh (mặc định: BẬT)")
    parser.add_argument("--no-save", dest="save", action="store_false", help="Không lưu ảnh kết quả")
    parser.add_argument("--out", type=str, default=None, help="Tùy chỉnh đường dẫn file ảnh kết quả lưu lại")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị tính toán: cpu, cuda, hoặc mps")

    args = parser.parse_args()

    print("=" * 68)
    print("   KIỂM TRA SO SÁNH ĐỘ GIỐNG NHAU KHUÔN MẶT - METHOD 3 (ARCFACE)")
    print("=" * 68)

    # 1. Xác định 2 ảnh cần so sánh
    if args.img1 and args.img2:
        path1 = Path(args.img1)
        path2 = Path(args.img2)
        if not path1.exists():
            print(f"[LỖI] Không tìm thấy file ảnh 1: {path1}")
            sys.exit(1)
        if not path2.exists():
            print(f"[LỖI] Không tìm thấy file ảnh 2: {path2}")
            sys.exit(1)
    else:
        folder = Path(args.folder)
        folder.mkdir(parents=True, exist_ok=True)
        images = find_images_in_folder(folder)

        if len(images) < 2:
            print(f"\n[!] Thư mục '{folder}' hiện đang có {len(images)} ảnh:")
            for img in images:
                print(f"    - {img.name}")
            print("\n[HƯỚNG DẪN]")
            print(f"  Vui lòng đặt ít nhất 2 ảnh cần so sánh vào thư mục: {folder.resolve()}")
            print("  Hoặc chỉ định trực tiếp đường dẫn qua tham số:")
            print("  python comparison_check.py --img1 /duong_dan/anh1.jpg --img2 /duong_dan/anh2.jpg\n")
            sys.exit(0)

        path1 = images[0]
        path2 = images[1]
        print(f"[*] Tìm thấy {len(images)} ảnh trong '{folder.name}/'. Tự động so sánh 2 ảnh đầu tiên:")
        print(f"    - Ảnh 1: {path1.name}")
        print(f"    - Ảnh 2: {path2.name}")
        if len(images) > 2:
            print(f"    (Có thêm {len(images) - 2} ảnh khác trong folder, sử dụng --img1 và --img2 để chọn ảnh khác)")

    # 2. Khởi tạo model Method 3
    classifier = Method3FaceClassifier(ckpt_path=CKPT_PATH, device=args.device)

    # 3. Tiến hành so sánh
    print("\n[*] Đang trích xuất embedding và so sánh đặc trưng...")
    res = classifier.compare(
        img1_path=path1,
        img2_path=path2,
        threshold=args.threshold,
        do_crop=not args.no_crop
    )

    # 4. In kết quả ra màn hình
    match_icon = "✅ CÙNG MỘT NGƯỜI (MATCH)" if res["is_match"] else "❌ KHÁC NGƯỜI (DIFFERENT PERSON)"
    print("\n" + "-" * 68)
    print(f"  KẾT QUẢ SO SÁNH:")
    print("-" * 68)
    print(f"  Ảnh 1             : {res['img1_path'].name}")
    print(f"    - Face BBox     : {res['bbox1'] if res['bbox1'] else 'Dùng toàn bộ ảnh (hoặc đã crop sẵn)'}")
    print(f"    - Pred Class ID : {res['cls1']} (Conf: {res['conf1']*100:.2f}%)")
    print()
    print(f"  Ảnh 2             : {res['img2_path'].name}")
    print(f"    - Face BBox     : {res['bbox2'] if res['bbox2'] else 'Dùng toàn bộ ảnh (hoặc đã crop sẵn)'}")
    print(f"    - Pred Class ID : {res['cls2']} (Conf: {res['conf2']*100:.2f}%)")
    print("-" * 68)
    print(f"  Cosine Similarity : {res['cosine_similarity']:+.4f}  (Thang đo [-1.0, +1.0])")
    print(f"  Tỷ lệ tương đồng  : {res['similarity_percentage']:.2f}%")
    print(f"  Euclidean Distance: {res['euclidean_distance']:.4f}   (Thang đo [0.0, 2.0], càng nhỏ càng giống)")
    print(f"  Ngưỡng (Threshold): {res['threshold']:.3f}   (Ngưỡng tối ưu của Method 3)")
    print(f"  KẾT LUẬN CUỐI CÙNG: {match_icon}")
    print("=" * 68)

    # 5. Lưu ảnh trực quan
    out_file = Path(args.out) if args.out else Path(args.folder) / "result.png"
    if args.save:
        canvas = create_comparison_visualization(res, out_path=out_file)

    # 6. Hiển thị GUI nếu người dùng yêu cầu
    if args.show:
        if not args.save:
            canvas = create_comparison_visualization(res, out_path=None)
        cv2.imshow("Ket Qua So Sanh Khuon Mat - Method 3", canvas)
        print("[*] Nhấn phím bất kỳ trên cửa sổ ảnh để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
