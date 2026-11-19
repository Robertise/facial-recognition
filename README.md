# Face Attendance System with Adversarial Eyeglasses Attack & Certified Defense

A real-time webcam-based face attendance system with liveness verification, emotion estimation, and facial recognition, alongside an adversarial eyeglasses attack and certified defense framework.

---

## 1. System Overview

Incoming webcam video frames are processed through a sequential pipeline:

1. **Liveness / Anti-spoofing** (`liveness/`): MediaPipe face detection combined with a deep anti-spoofing classifier to distinguish real faces from presentation attacks (photos, replay screens), smoothed across consecutive frames.
2. **Emotion Recognition & Face Recognition**: Genuine faces proceed to emotion classification (`emotion/`, 7 classes) and face recognition (`backend/face_recognizer.py`, 512-dimensional ArcFace embeddings compared against enrolled identities using cosine similarity).
3. **Attendance Logging & Registration**: Recognized identities are logged to an SQLite database (`attendance.db`). Unregistered individuals can register multiple face angles via the `/register` web interface.

---

## 2. Project Structure

```
.
├── backend/                 # FastAPI server: WebSocket (/ws/attendance, /ws/register), REST API, SQLite
├── frontend/                # Next.js web interface: Overview, Check-in, Register, People, History
├── liveness/                # Face anti-spoofing module and evaluation tools
├── emotion/                 # Facial emotion recognition (EmotionCNN)
├── extreme_cond_enhancer/   # Preprocessing enhancement for extreme lighting conditions
├── innovation/              # Adversarial eyeglasses attack and certified defense package
├── results/                 # Evaluation runs and benchmark outputs
├── tests/                   # Test suite (liveness, recognition, extreme conditions, innovation)
├── docs/                    # Project documentation and innovation experiment records
├── requirements.txt         # Runtime Python dependencies
└── requirements-dev.txt     # Development and testing dependencies
```

---

## 3. Required Checkpoints

Pretrained model weights belong in `checkpoints/` (or can be configured via environment variables):

| File | Module | Environment Variables |
| :--- | :--- | :--- |
| `checkpoints/liveness_best.pth` + `checkpoints/liveness_config.json` | `LivenessDetector` (Required) | `LIVENESS_CHECKPOINT`, `LIVENESS_CONFIG`, `LIVENESS_THRESHOLD` |
| `checkpoints/emotion/emotion_best.pt` (or `.pth`) | `EmotionDetector` | `EMOTION_CHECKPOINT` |
| `checkpoints/method3_best.pth` | `FaceRecognizer` (FaceCNN + ArcFace) | `CHECKPOINT_PATH`, `RECOGNITION_THRESHOLD` (default: 0.5) |

*(Note: `checkpoints/`, `data/`, and `attendance.db` are gitignored and managed locally.)*

---

## 4. Installation & Quickstart

### Prerequisites
- Python 3.10–3.12
- Node.js 20+

### Setup
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### Running the System
1. **Start the backend** (from repository root):
   ```bash
   python -m uvicorn backend.main:app --reload --port 8000
   ```

2. **Start the frontend** (in a separate terminal):
   ```bash
   cd frontend
   npm run dev
   ```

3. Open `http://localhost:3000` in your browser.

*(Optional: Set `TORCH_THREADS=1` if running on systems with hybrid performance/efficiency CPU architectures to optimize latency.)*

### Testing
```bash
pip install -r requirements-dev.txt
python -m pytest
```

---

## 5. Innovation: Adversarial Eyeglasses Attack & Certified Defense

This package attacks the facial attendance system with adversarial eyeglasses, explains why the attack works, and defends against it. The final defense is a multi-mask consensus anchored on eye landmarks, which **certifies** that an attacker cannot impersonate an enrolled subject using an accessory contained inside the protected facial regions.

The full experiment record is documented in [`docs/innovation/EXPERIMENTS.md`](docs/innovation/EXPERIMENTS.md): methods, decision rules fixed before results, benchmark runs, negative results, and limitations.

### Headline Results (Held-Out Test Sessions, Run `20261119_064729_certified_eval`)

| Configuration | GAR Owner | GAR Second Person | Adversarial Glasses (3 seeds) | Adaptive 17 mm Ring |
| :--- | :---: | :---: | :---: | :---: |
| **No defense** | 97.3% | 97.3% | 69–87% | 44% |
| **Purifier (U-Net Inpainter)** | 90.7% | 96.0% | 0% | 56% |
| **Certified Defense {G16, F}** | 96.0% | 97.3% | 0% | 0% |

### Innovation Layout
```
innovation/
├── attacks/adv_glasses/      # Attack: frame geometry, EOT, optimization, evaluation, ladders, printability, signatures
│   ├── glasses.py            # Frame/texture rendering onto a face (ellipse and photo-based real frame)
│   ├── faces.py              # Detector, landmark, crop, and gallery helpers (uses baseline pipeline)
│   ├── eot.py                # Expectation-Over-Transformation (EOT) augmentations
│   ├── optimize.py           # Optimize impersonation and dodging textures
│   ├── evaluate.py           # Score a run on held-out or physical sessions (+ transfer to other backbones)
│   ├── ladder.py             # Pre-registered ladder of configurations on dev set
│   ├── printability.py       # Color-chart fit and printable gamut loss for physical attempts
│   ├── export.py             # Export printable frame sheets
│   ├── signatures.py         # Representation mechanism analysis (S1/S2/S3 cosine shifts)
│   └── pipeline.py           # Attack pipeline around the baseline recognizer
├── defense/
│   ├── band.py               # Eye-landmark anchored glasses band generator
│   ├── purifier.py           # U-Net inpainter architecture
│   ├── prepare.py            # LFW face crops (identities disjoint from gallery)
│   ├── prepare_landmarks.py  # Landmark extraction (eye landmarks + 468-point mesh)
│   ├── train_purifier.py, eval_purifier.py
│   ├── adaptive.py           # Adaptive boundary ring attack against purifier (ring | optimize | eval)
│   ├── maskset.py            # Certified mask set (G0, G16, G32, G48, L, F)
│   ├── train_inpainter.py    # Multi-mask inpainter training
│   ├── prepare_faces.py      # Face crop and landmark extraction from labeled sets (VGGFace2 test)
│   ├── train_student.py      # Mask-invariant student recognizer (distillation + symmetric contrastive)
│   └── eval_certified.py     # Certified consensus: run (dev|test) | lfwval | calibrate | summarize
├── eval/                     # Wilson confidence interval metrics and run directory tracking
└── tools/
    ├── build_gallery.py      # LFW gallery generator: victim, extra victims, distractors
    └── frame_mask_from_photo.py  # Extract frame and lens masks from photos of physical glasses
```
Innovation tests: `tests/innovation/` (`pytest tests/innovation`).

### Innovation Requirements & Environment
- Python 3.10–3.12, base `requirements.txt`, plus `mediapipe==0.10.14`, `protobuf>=4.25.3,<5`, `numpy==1.26.4`, and `opencv-contrib-python==4.11.0.86`. Linux also requires `libgl1` and `libglib2.0-0`.
- GPU acceleration is recommended for training the purifier, inpainter, student models, and optimizing adaptive attacks. Evaluation scripts run on CPU.
- All commands run from the repository root using relative paths.

### Data and Checkpoints
`data/`, `checkpoints/`, and run checkpoints are gitignored. The owner and second-person webcam frames are personal data and are not published. Expected layout:

| Path | Content |
| :--- | :--- |
| `checkpoints/method3_best.pth` | Baseline face recognizer (attack target); `method1/2/4/5_best.pth` for transfer tests |
| `data/innovation/frames/owner/s1_day1 … s6_vary` | Attacker webcam sessions (75 frames each) |
| `data/innovation/frames/p01/s1, s2` | Second genuine subject evaluation sessions |
| `data/innovation/frames/owner_physical/` | Physical attempt recordings |
| `data/innovation/gallery/` | Enrolled gallery (victim + LFW distractor identities, `_build_report.json`) |
| `data/innovation/gallery_people/p01/` | Enrolment images for second subject |
| `data/innovation/frame/` | Real-frame photo, alpha mask, lens geometry |
| `data/innovation/chart/` | Color calibration chart and fitted color map |
| `data/innovation/defense/` | Band, boundary rings, LFW crops/landmarks, mask set, VGGFace2 crops |
| `data/public/lfw/lfw_home/lfw_funneled/` | LFW funneled dataset |
| `data/vggface2/identity_meta.csv`, `data/vggface2/test/<class_id>/*.jpg` | VGGFace2 test set for gallery filtering and student distillation |

### Reproducing Innovation Stages
The exact command line of every reported run is recorded in `results/innovation/<run>/meta.json`. Key commands:

#### 1. Adversarial Glasses Attack (H09 on Real Frame, Seed 0)
```bash
python -m innovation.attacks.adv_glasses.optimize \
  --attacker-frames data/innovation/frames/owner \
  --gallery data/innovation/gallery \
  --attacker-id owner \
  --victim LFW_victim \
  --mode impersonate \
  --checkpoint checkpoints/method3_best.pth \
  --frame-mask data/innovation/frame/owner_frame_mask.png \
  --frame-lens data/innovation/frame/owner_frame_lens.png \
  --grow-px 8.0 \
  --clearance-mm 9.0 \
  --ipd-mm 63.0 \
  --tex-res 15 40 \
  --steps 3000 \
  --lr 0.03 \
  --dev-frac 0.2 \
  --dev-mode block \
  --seed 0 \
  --eval-sessions s3_vary s4_vary s5_vary s6_vary

python -m innovation.attacks.adv_glasses.evaluate \
  --run results/innovation/20261107_214538_advglasses_impersonate \
  --eval-session s6_vary \
  --liveness \
  --transfer method1=checkpoints/method1_best.pth method2=checkpoints/method2_best.pth method4=checkpoints/method4_best.pth method5=checkpoints/method5_best.pth
```
`--eval-sessions` must list every held-out session so none leaks into training or dev.

#### 2. Mechanism Analysis
```bash
python -m innovation.attacks.adv_glasses.signatures \
  --extra-gallery data/innovation/gallery_people \
  --people-sessions p01/s2 p01/s1 \
  --primary-session p01/s2 \
  --extra-attack H09=results/innovation/20261107_214538_advglasses_impersonate
```

#### 3. Purifier
```bash
python -m innovation.defense.prepare
python -m innovation.defense.prepare_landmarks
python -m innovation.defense.band
python -m innovation.defense.train_purifier --data data/innovation/defense/lfw_crops --device cuda
python -m innovation.defense.eval_purifier --modes none gray purifier --purifier-ckpt results/innovation/20261112_004047_purifier_train/best.pt --extra-gallery data/innovation/gallery_people --liveness
```

#### 4. Adaptive Attack (r ∈ {8, 16, 32, 48} px)
```bash
python -m innovation.defense.adaptive ring
python -m innovation.defense.adaptive optimize --ring-px 48 --device cuda
python -m innovation.defense.adaptive eval --run results/innovation/20261113_075000_adaptive_optimize --eval-session s6_vary --liveness --device cpu
```

#### 5. Certified Defense (v3, Final)
```bash
python -m innovation.defense.maskset build
python -m innovation.defense.train_inpainter --device cuda
python -m innovation.defense.prepare_faces --images-root data/vggface2/test --meta data/vggface2/identity_meta.csv --per-id 60 --out-dir data/innovation/defense/vgg_test --workers 8
python -m innovation.defense.train_student --fill inpaint --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt --device cuda
python -m innovation.defense.train_student --fill inpaint --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt --init-ckpt results/innovation/20261117_123128_student_train/best.pt --extra-data data/innovation/defense/vgg_test --sym-weight 1.0 --pk 64 --pk-weighting count --epochs 15 --patience 4 --num-workers 8 --device cuda
```
The first `train_student` builds the v2 student that initializes v3.

Dev selection, followed by one-time held-out evaluation:
```bash
python -m innovation.defense.eval_certified lfwval --val-part B --gallery-mode symmetric --gallery-encoder student --student-ckpt results/innovation/20261118_135729_student_train/best.pt --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt
python -m innovation.defense.eval_certified run --split dev --gallery-mode symmetric --gallery-encoder student --student-ckpt results/innovation/20261118_135729_student_train/best.pt --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt
python -m innovation.defense.eval_certified run --split test --liveness --device cpu --gallery-mode symmetric --gallery-encoder student --student-ckpt results/innovation/20261118_135729_student_train/best.pt --inpainter-ckpt results/innovation/20261115_144634_inpainter_train/best.pt
```

Each run writes to `results/innovation/<timestamp>_<kind>/` containing execution parameters (`meta.json`), summary statistics (`summary.json`), and per-frame CSV files where applicable.
