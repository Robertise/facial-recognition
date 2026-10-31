"use client";

interface Props {
  bbox: [number, number, number, number] | null | undefined;
  /** The frame the bbox was measured in (the camera's real resolution). */
  frame: { width: number; height: number };
  state: "real" | "fake" | "pending";
  /** Text on the tag above the box: a name, or what the person should do. */
  tag?: string | null;
}

/** The detected face, drawn over the video in the same coordinates the backend used. */
export default function FaceOverlay({ bbox, frame, state, tag }: Props) {
  if (!bbox) return null;
  const [x, y, w, h] = bbox;
  const pct = (v: number, of: number) => `${Math.max(0, Math.min(100, (v / of) * 100))}%`;
  return (
    <div className="stage-layer" aria-hidden="true">
      <div className="face-box" data-state={state}
        style={{ left: pct(x, frame.width), top: pct(y, frame.height), width: pct(w, frame.width), height: pct(h, frame.height) }}>
        {tag && <span className="face-tag">{tag}</span>}
      </div>
    </div>
  );
}
