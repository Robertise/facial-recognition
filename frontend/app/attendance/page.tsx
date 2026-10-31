"use client";
import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, UserPlus } from "lucide-react";
import WebcamCapture from "@/components/WebcamCapture";
import FaceOverlay   from "@/components/FaceOverlay";
import AttendanceLog from "@/components/AttendanceLog";
import { useFrameSocket, AttendanceResult } from "@/lib/websocket";

export default function AttendancePage() {
  const [result, setResult] = useState<AttendanceResult | null>(null);
  const [loggedMsg, setLoggedMsg] = useState<string | null>(null);

  const { wsRef, connected } = useFrameSocket<AttendanceResult>("/ws/attendance", (data) => {
    setResult(data);
    if (data.attendance_logged && data.recognition?.name) {
      setLoggedMsg(`✅ Đã điểm danh: ${data.recognition.name}`);
      setTimeout(() => setLoggedMsg(null), 3000);
    }
  });

  const rec   = result?.recognition;
  const live  = result?.liveness;
  const emo   = result?.emotion;

  return (
    <main className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <Link href="/" className="text-gray-400 hover:text-white"><ArrowLeft /></Link>
          <h1 className="text-2xl font-bold text-white">📷 Chế độ Điểm Danh</h1>
          <span className={`ml-auto px-3 py-1 rounded-full text-xs font-medium
            ${connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}>
            {connected ? "● Đã kết nối" : "○ Mất kết nối"}
          </span>
        </div>

        {/* Logged notification */}
        {loggedMsg && (
          <div className="mb-4 px-4 py-3 bg-green-900/60 border border-green-600 rounded-xl text-green-300 font-medium">
            {loggedMsg}
          </div>
        )}

        {/* Webcam + Overlay */}
        <div className="relative rounded-xl overflow-hidden">
          <WebcamCapture wsRef={wsRef} intervalMs={200} />
          <FaceOverlay result={result} videoWidth={640} videoHeight={480} />
        </div>

        {/* Status panel */}
        <div className="mt-4 grid grid-cols-3 gap-4">
          {/* Liveness */}
          <div className={`rounded-xl p-4 border ${
            live?.label === "Real" ? "border-green-600 bg-green-950/40" :
            live?.label === "Fake" ? "border-red-600 bg-red-950/40" :
                                     "border-gray-700 bg-gray-800/40"}`}>
            <p className="text-xs text-gray-400 mb-1">Liveness</p>
            <p className={`text-lg font-bold ${
              live?.label === "Real" ? "text-green-400" :
              live?.label === "Fake" ? "text-red-400" : "text-gray-400"}`}>
              {live ? `${live.label} ${(live.score*100).toFixed(0)}%` : "—"}
            </p>
          </div>

          {/* Emotion */}
          <div className="rounded-xl p-4 border border-yellow-700 bg-yellow-950/30">
            <p className="text-xs text-gray-400 mb-1">Cảm xúc</p>
            <p className="text-lg font-bold text-yellow-300">
              {emo ? `${emo.emoji} ${emo.label}` : "—"}
            </p>
          </div>

          {/* Recognition */}
          <div className={`rounded-xl p-4 border ${
            rec?.matched ? "border-blue-600 bg-blue-950/40" : "border-gray-700 bg-gray-800/40"}`}>
            <p className="text-xs text-gray-400 mb-1">Nhận diện</p>
            {rec?.matched ? (
              <p className="text-base font-bold text-blue-300">{rec.name}<br/>
                <span className="text-xs font-normal text-blue-400">{(rec.similarity*100).toFixed(1)}% tương đồng</span>
              </p>
            ) : result?.face_detected && live?.label === "Real" ? (
              <div>
                <p className="text-sm text-gray-300 font-medium">Chưa đăng ký</p>
                <Link href="/register"
                  className="mt-2 flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300">
                  <UserPlus className="w-3 h-3" /> Đăng ký ngay
                </Link>
              </div>
            ) : (
              <p className="text-gray-400">—</p>
            )}
          </div>
        </div>

        <AttendanceLog />
      </div>
    </main>
  );
}
