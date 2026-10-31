"use client";
import { useEffect, useRef } from "react";
import { AttendanceResult } from "@/lib/websocket";

interface Props {
  result: AttendanceResult | null;
  videoWidth: number;
  videoHeight: number;
}

export default function FaceOverlay({ result, videoWidth, videoHeight }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!result?.face_detected || !result.bbox) return;

    const [x, y, w, h] = result.bbox;
    const live  = result.liveness;
    const rec   = result.recognition;
    const emo   = result.emotion;

    // Bounding box color
    const color = live?.label === "Real" ? "#22c55e" : live?.label === "Fake" ? "#ef4444" : "#94a3b8";
    ctx.strokeStyle = color;
    ctx.lineWidth   = 3;
    ctx.strokeRect(x, y, w, h);

    // Corner accents
    const cs = 18;
    ctx.lineWidth = 4;
    [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]].forEach(([cx,cy,dx,dy]) => {
      ctx.beginPath(); ctx.moveTo(cx as number,cy as number); ctx.lineTo((cx as number)+(dx as number)*cs,cy as number); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(cx as number,cy as number); ctx.lineTo(cx as number,(cy as number)+(dy as number)*cs); ctx.stroke();
    });

    // Name label
    ctx.font = "bold 16px sans-serif";
    ctx.fillStyle = color;
    const name = rec?.matched ? `${rec.name} (${((rec.similarity||0)*100).toFixed(0)}%)` : "Unknown";
    ctx.fillText(name, x, y - 8);

    // Liveness badge
    if (live) {
      ctx.font = "13px sans-serif";
      ctx.fillStyle = live.label === "Real" ? "#86efac" : "#fca5a5";
      ctx.fillText(`${live.label} ${(live.score*100).toFixed(0)}%`, x, y + h + 18);
    }

    // Emotion
    if (emo) {
      ctx.font = "14px sans-serif";
      ctx.fillStyle = "#fde68a";
      ctx.fillText(`${emo.emoji} ${emo.label}`, x, y + h + 36);
    }
  }, [result]);

  return (
    <canvas
      ref={canvasRef}
      width={videoWidth}
      height={videoHeight}
      className="absolute inset-0 w-full h-full pointer-events-none rounded-xl"
    />
  );
}
