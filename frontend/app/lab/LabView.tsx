"use client";
import { useEffect, useRef, useState } from "react";
import { CircleAlert, LoaderCircle, Mic } from "lucide-react";
import WebcamCapture from "@/components/WebcamCapture";
import { DEFAULT_PROCESSING, useAVRecorder, type AudioProcessing } from "@/lib/media";
import { probeMedia, type MediaProbe } from "@/lib/innovation";

const RECORD_MS = 4000;

type Phase = "idle" | "recording" | "uploading" | "done" | "error";

const PROCESSING_LABELS: Record<keyof AudioProcessing, string> = {
  echoCancellation: "Echo cancellation",
  noiseSuppression: "Noise suppression",
  autoGainControl: "Automatic gain",
};

/** Records a 4 second clip from the shared camera stream plus the microphone and asks the backend to decode it. */
export default function LabView() {
  const noSocket = useRef<WebSocket | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [processing, setProcessing] = useState<AudioProcessing>(DEFAULT_PROCESSING);
  const recorder = useAVRecorder(stream, processing);
  const [phase, setPhase] = useState<Phase>("idle");
  const [probe, setProbe] = useState<MediaProbe | null>(null);
  const [clipUrl, setClipUrl] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [secondsLeft, setSecondsLeft] = useState(0);

  useEffect(() => () => { if (clipUrl) URL.revokeObjectURL(clipUrl); }, [clipUrl]);

  const run = async () => {
    setFailure(null);
    setProbe(null);
    if (!(await recorder.start())) { setPhase("error"); return; }
    setPhase("recording");
    const startedAt = performance.now();
    const timer = setInterval(() => {
      setSecondsLeft(Math.max(0, Math.ceil((RECORD_MS - (performance.now() - startedAt)) / 1000)));
    }, 200);
    setSecondsLeft(RECORD_MS / 1000);
    await new Promise((r) => setTimeout(r, RECORD_MS));
    clearInterval(timer);
    const blob = await recorder.stop();
    if (!blob) { setPhase("error"); setFailure("Recording did not start."); return; }
    setClipUrl(URL.createObjectURL(blob));
    setPhase("uploading");
    try {
      setProbe(await probeMedia(blob));
      setPhase("done");
    } catch (e) {
      setPhase("error");
      setFailure(`The backend could not decode the clip: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const busy = phase === "recording" || phase === "uploading" || recorder.state === "requesting";
  const error = recorder.error ?? failure;

  return (
    <div className="workbench">
      <WebcamCapture wsRef={noSocket} onStream={setStream} />

      <section className="panel" aria-labelledby="probe-title">
        <div className="panel-section">
          <h2 id="probe-title">Recorder check</h2>
          <p className="lead">Records 4 seconds of camera and microphone from the stream above, then the backend decodes it.</p>
        </div>

        <fieldset className="panel-section fieldset-plain" disabled={busy}>
          <legend className="visually-hidden">Browser audio processing</legend>
          {(Object.keys(PROCESSING_LABELS) as (keyof AudioProcessing)[]).map((k) => (
            <label key={k} className="check">
              <input type="checkbox" name={k} checked={processing[k]}
                onChange={(e) => setProcessing((p) => ({ ...p, [k]: e.target.checked }))} />
              <span>{PROCESSING_LABELS[k]}</span>
            </label>
          ))}
        </fieldset>

        <div className="panel-section">
          <div className="level">
            <Mic size={16} aria-hidden="true" />
            <div className="meter" role="meter" aria-label="Microphone level" aria-valuemin={0} aria-valuemax={1} aria-valuenow={Number(recorder.level.toFixed(2))}>
              <span style={{ transform: `scaleX(${recorder.level})` }} />
            </div>
          </div>
          <div className="actions gap-above">
            <button type="button" className="btn btn--primary" onClick={run} disabled={busy || !stream} aria-busy={busy || undefined}>
              {busy && <LoaderCircle size={16} className="spin" aria-hidden="true" />}
              {phase === "recording" ? `Recording, ${secondsLeft} s left` : phase === "uploading" ? "Decoding" : "Record 4 seconds"}
            </button>
          </div>
          {!stream && <p className="helper gap-above">Waiting for the camera.</p>}
        </div>

        {error && (
          <div className="panel-section" role="alert">
            <p className="notice" data-tone="bad"><CircleAlert size={16} aria-hidden="true" />{error}</p>
          </div>
        )}

        {probe && (
          <div className="panel-section" role="status">
            <dl className="facts">
              <dt>Video frames</dt>
              <dd><span className="measured">{probe.video.frames}</span></dd>
              <dt>Measured fps</dt>
              <dd>
                <span className="measured measured--cool">{probe.video.fps.toFixed(1)}</span>
                <span className="chip" data-tone={probe.checks.fps_ok ? "ok" : "bad"}>{probe.checks.fps_ok ? "20 or more" : "Under 20"}</span>
              </dd>
              <dt>Frame size</dt>
              <dd><span className="measured">{probe.video.width} × {probe.video.height}</span></dd>
              <dt>Audio</dt>
              <dd>
                <span className="measured measured--cool">{probe.audio.sample_rate} Hz, {probe.audio.duration_s.toFixed(2)} s</span>
                <span className="chip" data-tone={probe.checks.audio_ok ? "ok" : "bad"}>{probe.checks.audio_ok ? "16 kHz mono" : "Missing"}</span>
              </dd>
              <dt>Audio level</dt>
              <dd><span className="measured">{probe.audio.rms_dbfs === null ? "silent" : `${probe.audio.rms_dbfs} dBFS`}</span></dd>
              <dt>Decode time</dt>
              <dd><span className="measured">{Math.round(probe.decode_ms)} ms</span></dd>
              <dt>Browser reported</dt>
              <dd>
                <span className="measured">
                  {recorder.actualFps ? `${recorder.actualFps} fps` : "fps unknown"}
                  {recorder.audioSettings?.sampleRate ? `, mic ${recorder.audioSettings.sampleRate} Hz` : ""}
                </span>
              </dd>
              <dt>Container</dt>
              <dd><span className="measured">{probe.content_type ?? "unknown"}, {(probe.bytes / 1024).toFixed(0)} KB</span></dd>
            </dl>
            {clipUrl && <video className="playback gap-above" src={clipUrl} controls playsInline aria-label="Recorded clip" />}
          </div>
        )}
      </section>
    </div>
  );
}
