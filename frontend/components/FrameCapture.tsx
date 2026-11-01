"use client";
import { useEffect, useRef, useState } from "react";
import { Camera, CircleAlert, LoaderCircle, Square } from "lucide-react";
import { getInnovationConfig, saveFrame } from "@/lib/innovation";

const NAME = /^[A-Za-z0-9_-]{1,64}$/;

/**
 * Save a run of frames exactly as check-in sends them (the same canvas, JPEG
 * quality 0.95), for the L1 attack experiments. Needs INNOVATION_RESEARCH_MODE=1.
 */
export default function FrameCapture({ getCanvas }: { getCanvas: () => HTMLCanvasElement | null }) {
  const [research, setResearch] = useState<boolean | null>(null);
  const [person, setPerson] = useState("");
  const [session, setSession] = useState("");
  const [count, setCount] = useState(60);
  const [intervalMs, setIntervalMs] = useState(200);
  const [saved, setSaved] = useState(0);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stop = useRef(false);

  useEffect(() => {
    getInnovationConfig().then((c) => setResearch(c.research_mode)).catch(() => setResearch(false));
  }, []);

  const valid = NAME.test(person) && NAME.test(session) && count > 0 && count <= 600;

  const run = async () => {
    setError(null);
    setSaved(0);
    setRunning(true);
    stop.current = false;
    const t0 = performance.now();
    try {
      for (let i = 0; i < count && !stop.current; i++) {
        const due = t0 + i * intervalMs;
        await new Promise((r) => setTimeout(r, Math.max(0, due - performance.now())));
        const canvas = getCanvas();
        if (!canvas) throw new Error("the camera is not ready");
        const blob = await new Promise<Blob | null>((r) => canvas.toBlob(r, "image/jpeg", 0.95));
        if (!blob) throw new Error("the frame could not be encoded");
        await saveFrame(person, session, i, blob, performance.now() - t0);
        setSaved(i + 1);
      }
    } catch (e) {
      setError(`Capture stopped: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="panel" aria-labelledby="capture-title">
      <div className="panel-section">
        <h2 id="capture-title">Frame capture</h2>
        <p className="lead">Saves frames as check-in sends them, to data/innovation/frames/person/session.</p>
      </div>
      {research === false ? (
        <div className="panel-section">
          <p className="notice"><CircleAlert size={16} aria-hidden="true" />Start the backend with INNOVATION_RESEARCH_MODE=1 to capture frames.</p>
        </div>
      ) : (
        <>
          <div className="panel-section form-grid">
            <div className="field">
              <label htmlFor="cap-person">Person</label>
              <input id="cap-person" name="person" className="input" value={person} onChange={(e) => setPerson(e.target.value)}
                placeholder="owner" autoComplete="off" disabled={running} aria-invalid={(person !== "" && !NAME.test(person)) || undefined} />
              <span className="helper">Letters, digits, - and _.</span>
            </div>
            <div className="field">
              <label htmlFor="cap-session">Session</label>
              <input id="cap-session" name="session" className="input" value={session} onChange={(e) => setSession(e.target.value)}
                placeholder="s1_bright" autoComplete="off" disabled={running} aria-invalid={(session !== "" && !NAME.test(session)) || undefined} />
              <span className="helper">Say the light and day, e.g. s2_dim.</span>
            </div>
            <div className="field">
              <label htmlFor="cap-count">Frames</label>
              <input id="cap-count" name="count" type="number" min={1} max={600} className="input" value={count}
                onChange={(e) => setCount(Number(e.target.value))} disabled={running} />
              <span className="helper">Move your head slowly while it runs.</span>
            </div>
            <div className="field">
              <label htmlFor="cap-interval">Every (ms)</label>
              <input id="cap-interval" name="interval" type="number" min={50} max={2000} step={50} className="input" value={intervalMs}
                onChange={(e) => setIntervalMs(Number(e.target.value))} disabled={running} />
              <span className="helper">200 ms matches check-in.</span>
            </div>
          </div>
          <div className="panel-section">
            <div className="actions">
              {running ? (
                <button type="button" className="btn btn--primary" onClick={() => { stop.current = true; }}>
                  <Square size={14} aria-hidden="true" />Stop
                </button>
              ) : (
                <button type="button" className="btn btn--primary" onClick={run} disabled={!valid || research === null}>
                  {research === null ? <LoaderCircle size={16} className="spin" aria-hidden="true" /> : <Camera size={16} aria-hidden="true" />}
                  Capture {count} frames
                </button>
              )}
            </div>
            <p className="helper gap-above" role="status"><span className="measured">{saved}</span> of {count} saved</p>
            {error && <p className="notice" data-tone="bad" role="alert"><CircleAlert size={16} aria-hidden="true" />{error}</p>}
          </div>
        </>
      )}
    </section>
  );
}
