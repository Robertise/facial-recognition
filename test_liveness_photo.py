"""
test_liveness_photo.py – Liveness + Emotion on static images in /photo/.
Emoji rendered via Pillow (Unicode support).
Usage:
    python3 test_liveness_photo.py [--save] [--show] [--img path]
"""
import sys, argparse, cv2, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from liveness.liveness_detector import LivenessDetector
from emotion.emotion_detector   import EmotionDetector
from utils.draw_utils           import put_liveness_emotion

PHOTO_DIR  = PROJECT_ROOT / 'photo'
OUTPUT_DIR = PHOTO_DIR / 'output'
IMG_EXTS   = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

COLOR_REAL = (0, 200, 0)
COLOR_FAKE = (0, 0, 220)
COLOR_NONE = (120, 120, 120)


def draw_bbox(img, bbox, live_label):
    if bbox is None: return img
    color = COLOR_REAL if live_label=='Real' else (COLOR_FAKE if live_label=='Fake' else COLOR_NONE)
    x, y, w, h = bbox
    cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
    return img


def crop_face(img, bbox, pad=0.1):
    if bbox is None: return img
    x,y,bw,bh = bbox; H,W = img.shape[:2]
    dx,dy = int(bw*pad), int(bh*pad)
    return img[max(0,y-dy):min(H,y+bh+dy), max(0,x-dx):min(W,x+bw+dx)]


def process(live_det, emo_det, img_path, save, show):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f'  [ERROR] Cannot read {img_path.name}'); return

    is_real, live_conf, live_label, bbox = live_det.check(img)
    emo_label = emo_emoji = emo_conf = None
    if is_real and bbox is not None:
        face = crop_face(img, bbox)
        if face.size > 0:
            emo_label, emo_emoji, emo_conf = emo_det.predict(face)

    status = '✅ REAL' if is_real else ('❌ FAKE' if live_label=='Fake' else '⚠️  NO FACE')
    emo_str = f'  |  {emo_label} {emo_emoji} {emo_conf*100:.1f}%' if emo_label else ''
    print(f'  {img_path.name:<28}  {status:<12}  {live_conf*100:.1f}%{emo_str}')

    out = draw_bbox(img.copy(), bbox, live_label)
    out = put_liveness_emotion(out, live_label, live_conf, emo_label, emo_emoji, emo_conf)

    if save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        cv2.imwrite(str(OUTPUT_DIR / img_path.name), out)
    if show:
        cv2.imshow(f'Result – {img_path.name}', out)
        cv2.waitKey(0); cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',       type=str,   default=None)
    parser.add_argument('--save',      action='store_true')
    parser.add_argument('--show',      action='store_true')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override liveness threshold (default: value in checkpoints/liveness_config.json)')
    args = parser.parse_args()

    print('='*65)
    print('  Liveness + Emotion Detection – Static Images')
    print('='*65)

    live_det = LivenessDetector(threshold=args.threshold)
    emo_det  = EmotionDetector()

    img_paths = ([Path(args.img)] if args.img else
                 [p for p in sorted(PHOTO_DIR.iterdir()) if p.suffix.lower() in IMG_EXTS])
    if not img_paths:
        print('[ERROR] No images found'); sys.exit(1)

    print(f'\nProcessing {len(img_paths)} image(s)...\n')
    counts = {'Real':0,'Fake':0,'No Face':0}
    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None: continue
        _, _, lbl, _ = live_det.check(img)
        counts[lbl if lbl in counts else 'No Face'] += 1
        process(live_det, emo_det, path, args.save, args.show)

    print(f'\n  Real:{counts["Real"]}  Fake:{counts["Fake"]}  No Face:{counts["No Face"]}')
    if args.save: print(f'  Saved to: {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
