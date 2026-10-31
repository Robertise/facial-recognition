"use client";
import { useRef, useEffect, useState } from "react";
import { CameraOff } from "lucide-react";
import { canSendFrame, sendFrame } from "@/lib/websocket";

interface Props {
  wsRef: React.MutableRefObject<WebSocket | null>;
  intervalMs?: number;
  onFrame?: (canvas: HTMLCanvasElement) => void;
  /** Called with the camera's real frame size once the stream starts. */
  onFrameSize?: (size: { width: number; height: number }) => void;
  children?: React.ReactNode;
}

export default function WebcamCapture({ wsRef, intervalMs = 200, onFrame, onFrameSize, children }: Props) {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [aspect, setAspect] = useState("4 / 3");

  useEffect(() => {
    let stream: MediaStream | null = null;
    let stopped = false;

    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
      .then((s) => {
        if (stopped) { s.getTracks().forEach((t) => t.stop()); return; }
        stream = s;
        setCameraError(null);
        if (videoRef.current) videoRef.current.srcObject = s;
      })
      .catch((e: unknown) => {
        const name = e instanceof DOMException ? e.name : "";
        setCameraError(name === "NotAllowedError"
          ? "Camera access was blocked. Allow the camera for this site in the browser, then reload."
          : name === "NotFoundError"
            ? "No camera was found. Connect a webcam, then reload."
            : "The camera could not start. Close other apps using it, then reload.");
      });

    const timer = setInterval(() => {
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas || video.readyState < 2) return;
      // Skip this tick while the backend is still busy with the previous frame.
      const ws = wsRef.current;
      if (!canSendFrame(ws) && !onFrame) return;
      const ctx = canvas.getContext("2d")!;
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0);
      if (canSendFrame(ws)) sendFrame(ws, canvas);
      onFrame?.(canvas);
    }, intervalMs);

    return () => {
      stopped = true;
      clearInterval(timer);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [wsRef, intervalMs, onFrame]);

  return (
    <div className="stage" style={{ aspectRatio: aspect }}>
      <video ref={videoRef} autoPlay playsInline muted aria-label="Live camera"
        onLoadedMetadata={(e) => {
          const size = { width: e.currentTarget.videoWidth || 640, height: e.currentTarget.videoHeight || 480 };
          setAspect(`${size.width} / ${size.height}`);
          onFrameSize?.(size);
        }} />
      <canvas ref={canvasRef} hidden aria-hidden="true" />
      {cameraError ? (
        <div className="stage-blank" role="alert">
          <CameraOff size={28} aria-hidden="true" />
          <p>{cameraError}</p>
        </div>
      ) : children}
    </div>
  );
}
