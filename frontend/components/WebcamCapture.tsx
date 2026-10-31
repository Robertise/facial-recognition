"use client";
import { useRef, useEffect, useCallback } from "react";
import { canSendFrame, sendFrame } from "@/lib/websocket";

interface Props {
  wsRef: React.MutableRefObject<WebSocket | null>;
  intervalMs?: number;
  onFrame?: (canvas: HTMLCanvasElement) => void;
}

export default function WebcamCapture({ wsRef, intervalMs = 200, onFrame }: Props) {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let timer: ReturnType<typeof setInterval>;

    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } })
      .then((s) => {
        stream = s;
        if (videoRef.current) videoRef.current.srcObject = s;
      });

    timer = setInterval(() => {
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
      clearInterval(timer);
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [wsRef, intervalMs, onFrame]);

  return (
    <div className="relative w-full max-w-2xl mx-auto">
      <video ref={videoRef} autoPlay playsInline muted
        className="w-full rounded-xl shadow-lg" />
      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}
