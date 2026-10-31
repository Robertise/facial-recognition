// lib/media.ts – record video + audio from the camera stream the page already has open.
//
// The camera is opened once (by WebcamCapture, which also feeds the 5 fps frame
// socket). Recording adds a microphone track to that same video track, so the
// camera is never opened twice (Windows drivers often refuse a second open).
import { useCallback, useEffect, useRef, useState } from "react";

export type AudioProcessing = {
  echoCancellation: boolean;
  noiseSuppression: boolean;
  autoGainControl: boolean;
};

export const DEFAULT_PROCESSING: AudioProcessing = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
};

export type RecorderState = "idle" | "requesting" | "recording" | "error";

const MIME_CANDIDATES = [
  "video/webm;codecs=vp8,opus",
  "video/webm;codecs=vp9,opus",
  "video/webm",
  "video/mp4",
];

function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "";
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
}

function micErrorMessage(e: unknown): string {
  const name = e instanceof DOMException ? e.name : "";
  if (name === "NotAllowedError") return "Microphone access was blocked. Allow the microphone for this site, then try again.";
  if (name === "NotFoundError") return "No microphone was found. Connect one, then try again.";
  return "The microphone could not start. Close other apps using it, then try again.";
}

/**
 * Record the given camera stream together with a microphone track.
 * `level` is the live input level (0–1) while recording; `actualFps` and
 * `audioSettings` are what the browser actually delivered.
 */
export function useAVRecorder(videoStream: MediaStream | null, processing: AudioProcessing = DEFAULT_PROCESSING) {
  const [state, setState] = useState<RecorderState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [level, setLevel] = useState(0);
  const [actualFps, setActualFps] = useState<number | null>(null);
  const [audioSettings, setAudioSettings] = useState<MediaTrackSettings | null>(null);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const micRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const stopResolveRef = useRef<((b: Blob) => void) | null>(null);

  const release = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    audioCtxRef.current?.close().catch(() => {});
    audioCtxRef.current = null;
    micRef.current?.getTracks().forEach((t) => t.stop());
    micRef.current = null;
    setLevel(0);
  }, []);

  useEffect(() => release, [release]);

  /** Start recording; resolves false (with `error` set) when the microphone or recorder fails. */
  const start = useCallback(async (): Promise<boolean> => {
    if (!videoStream) { setError("The camera is not ready yet."); setState("error"); return false; }
    if (recorderRef.current?.state === "recording") return true;
    setError(null);
    setState("requesting");
    try {
      const mic = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: 48000, ...processing },
        video: false,
      });
      micRef.current = mic;
      const audioTrack = mic.getAudioTracks()[0];
      const videoTrack = videoStream.getVideoTracks()[0];
      setAudioSettings(audioTrack.getSettings());
      setActualFps(videoTrack?.getSettings().frameRate ?? null);

      // Live input level from an analyser on the microphone track.
      const ctx = new AudioContext();
      audioCtxRef.current = ctx;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      ctx.createMediaStreamSource(mic).connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      const tick = () => {
        analyser.getFloatTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
        setLevel(Math.min(1, Math.sqrt(sum / buf.length) * 4));
        rafRef.current = requestAnimationFrame(tick);
      };
      tick();

      const mimeType = pickMimeType();
      const combined = new MediaStream([...(videoTrack ? [videoTrack] : []), audioTrack]);
      const rec = new MediaRecorder(combined, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      rec.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: rec.mimeType || mimeType || "video/webm" });
        release();
        setState("idle");
        stopResolveRef.current?.(blob);
        stopResolveRef.current = null;
      };
      recorderRef.current = rec;
      rec.start(250);
      setState("recording");
      return true;
    } catch (e) {
      release();
      setError(micErrorMessage(e));
      setState("error");
      return false;
    }
  }, [videoStream, processing, release]);

  /** Stop and resolve with the recorded clip. */
  const stop = useCallback((): Promise<Blob | null> => {
    const rec = recorderRef.current;
    if (!rec || rec.state !== "recording") return Promise.resolve(null);
    return new Promise((resolve) => {
      stopResolveRef.current = resolve;
      rec.stop();
    });
  }, []);

  return { state, error, level, actualFps, audioSettings, start, stop };
}
