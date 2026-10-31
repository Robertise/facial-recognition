"""
enroll_public.py – add people from public datasets to the attendance database:
the simulated class roster (distractors for 1:N recognition) and the virtual
victim identities of the L1 attack (face and voice of the same person).

Layout, one folder per person:

    <root>/<person_id>/face/*.jpg|png     face photos or video frames (any size; the face is detected)
    <root>/<person_id>/voice/*.wav|flac   speech clips (optional)
    <root>/<person_id>/name.txt           display name (optional; defaults to the folder name)

    python -m innovation.tools.enroll_public data/public/roster --prefix PUB --max-faces 5 --max-voice 3

IDs become <prefix>_<person_id>. Existing IDs are skipped unless --replace.
Uses the same face crop and recognizer as the backend, so the stored
embeddings match what check-in compares against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUDIO_EXT = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}


def detect_largest_face(detector, img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    res = detector.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if not res.detections:
        return None
    best = max(res.detections, key=lambda d: d.location_data.relative_bounding_box.width)
    bb = best.location_data.relative_bounding_box
    return [max(0, int(bb.xmin * w)), max(0, int(bb.ymin * h)), int(bb.width * w), int(bb.height * h)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder with one sub-folder per person")
    ap.add_argument("--prefix", default="PUB")
    ap.add_argument("--max-faces", type=int, default=5)
    ap.add_argument("--max-voice", type=int, default=3)
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--replace", action="store_true", help="re-create people that already exist")
    args = ap.parse_args()

    import mediapipe as mp
    from backend import database
    from backend.processing import _get_recognizer, crop_face_from_bbox
    from innovation import db as idb
    from innovation.media import decode_media

    database.init_db()
    idb.init_innovation_db()
    recognizer = _get_recognizer()
    detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    encoder = None
    if not args.no_voice:
        from innovation.voice import get_encoder, quality
        encoder = get_encoder()

    people = sorted(p for p in Path(args.root).iterdir() if p.is_dir())
    added = skipped = 0
    for person in people:
        emp_id = f"{args.prefix}_{person.name}"
        name_file = person / "name.txt"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else person.name
        if idb.employee_exists(emp_id):
            if not args.replace:
                skipped += 1
                continue
            database.delete_employee(emp_id)

        faces = []
        for f in sorted((person / "face").glob("*")):
            if f.suffix.lower() not in IMG_EXT or len(faces) >= args.max_faces:
                continue
            img = cv2.imread(str(f))
            bbox = detect_largest_face(detector, img) if img is not None else None
            if bbox is None:
                continue
            crop = crop_face_from_bbox(img, bbox)
            if crop.size:
                faces.append(recognizer.extract_embedding(crop))
        if not faces:
            print(f"  skip {emp_id}: no face found")
            skipped += 1
            continue

        database.create_employee(emp_id, name)
        for k, emb in enumerate(faces):
            database.save_embedding(emp_id, f"public-{k}", emb)

        n_voice = 0
        if encoder is not None and (person / "voice").is_dir():
            for f in sorted((person / "voice").glob("*")):
                if f.suffix.lower() not in AUDIO_EXT or n_voice >= args.max_voice:
                    continue
                audio = decode_media(f, with_video=False).audio
                if not quality(audio).ok:
                    continue
                idb.save_voice_embedding(emp_id, encoder.embed(audio), source=f"public:{f.name}")
                n_voice += 1
        added += 1
        print(f"  added {emp_id} ({name}): {len(faces)} faces, {n_voice} voice clips")

    print(f"done: {added} added, {skipped} skipped")


if __name__ == "__main__":
    main()
