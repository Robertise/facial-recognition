"""
test_liveness_webcam.py – Liveness + Emotion, real-time webcam.
Controls: Q/ESC=Quit  S=Save frame
"""
import sys, argparse, cv2, numpy as np, time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from liveness.liveness_detector import LivenessDetector
from emotion.emotion_detector   import EmotionDetector
from utils.draw_utils           import put_liveness_emotion

OUTPUT_DIR  = PROJECT_ROOT / 'photo' / 'output'
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)


def draw_bbox(frame, bbox, live_label):
    if bbox is None:
        return frame
    color = (0,220,0) if live_label=='Real' else ((0,0,220) if live_label=='Fake' else (120,120,120))
    x, y, bw, bh = bbox
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
    sz, th = 16, 3
    for cx, cy, dx, dy in [(x,y,1,1),(x+bw,y,-1,1),(x,y+bh,1,-1),(x+bw,y+bh,-1,-1)]:
        cv2.line(frame, (cx,cy), (cx+dx*sz,cy), color, th)
        cv2.line(frame, (cx,cy), (cx,cy+dy*sz), color, th)
    return frame


def crop_face(img, bbox, pad=0.1):
    if bbox is None:
        return img
    x, y, bw, bh = bbox
    H, W = img.shape[:2]
    dx, dy = int(bw*pad), int(bh*pad)
    return img[max(0,y-dy):min(H,y+bh+dy), max(0,x-dx):min(W,x+bw+dx)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam',       type=int,   default=0)
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override liveness threshold (default: value in checkpoints/liveness_config.json)')
    parser.add_argument('--width',     type=int,   default=640)
    parser.add_argument('--height',    type=int,   default=480)
    parser.add_argument('--skip',      type=int,   default=3)
    args = parser.parse_args()

    print('='*65)
    print('  Liveness + Emotion Detection – Webcam Real-Time')
    print(f'  Cam: {args.cam}  Threshold: {args.threshold if args.threshold is not None else "from config"}  Skip: {args.skip} frames')
    print('  Controls: Q/ESC=Quit   S=Save frame')
    print('='*65+'\n')

    live_det = LivenessDetector(threshold=args.threshold)
    emo_det  = EmotionDetector()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera {args.cam}')
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # macOS: give camera time to warm up, flush stale frames
    time.sleep(0.8)
    for _ in range(5):
        cap.read()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state = dict(live_label='Detecting...', live_conf=0.0, bbox=None,
                 emo_label=None, emo_emoji=None, emo_conf=None)
    frame_idx  = 0
    t_prev     = time.time()
    fail_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count > 10:
                    print('[ERROR] Camera lost.')
                    break
                time.sleep(0.05)
                continue
            fail_count = 0
            frame_idx += 1

            # Run inference every N frames
            if frame_idx % args.skip == 0:
                is_real, live_conf, live_label, bbox = live_det.check(frame)
                state.update(live_label=live_label, live_conf=live_conf, bbox=bbox)

                if is_real and bbox is not None:
                    face = crop_face(frame, bbox)
                    if face.size > 0:
                        lbl, emj, conf = emo_det.predict(face)
                        state.update(emo_label=lbl, emo_emoji=emj, emo_conf=conf)
                else:
                    state.update(emo_label=None, emo_emoji=None, emo_conf=None)

            # FPS
            t_now  = time.time()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            # Draw
            display = frame.copy()
            display = draw_bbox(display, state['bbox'], state['live_label'])
            display = put_liveness_emotion(
                display,
                state['live_label'], state['live_conf'],
                state['emo_label'],  state['emo_emoji'], state['emo_conf'],
            )
            cv2.putText(display, f'FPS: {fps:.1f}', (display.shape[1]-120, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1, cv2.LINE_AA)
            h = display.shape[0]
            cv2.rectangle(display, (0, h-30), (display.shape[1], h), COLOR_BLACK, -1)
            cv2.putText(display, 'Q/ESC: Quit   S: Save', (10, h-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,180,180), 1, cv2.LINE_AA)

            cv2.imshow('Liveness + Emotion (Q=Quit, S=Save)', display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key in (ord('s'), ord('S')):
                ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
                emo  = state['emo_label'] or 'NoEmo'
                path = OUTPUT_DIR / f'webcam_{state["live_label"]}_{emo}_{ts}.jpg'
                cv2.imwrite(str(path), display)
                print(f'  [Saved] {path}')

    finally:
        cap.release()
        cv2.destroyAllWindows()
        time.sleep(0.3)  # macOS: let OS reclaim camera before exit
        print('Done.')


if __name__ == '__main__':
    main()
