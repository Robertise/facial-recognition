"use client";
import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { ArrowLeft, CheckCircle } from "lucide-react";
import WebcamCapture from "@/components/WebcamCapture";
import AngleGuide, { ANGLES, Angle } from "@/components/AngleGuide";
import { useFrameSocket, API_BASE, RegisterResult } from "@/lib/websocket";

type Capture = { angle: Angle; face_b64: string; preview: string };

export default function RegisterPage() {
  const [result, setResult]     = useState<RegisterResult | null>(null);
  const { wsRef } = useFrameSocket<RegisterResult>("/ws/register", setResult);
  const [empId, setEmpId]       = useState("");
  const [empName, setEmpName]   = useState("");
  const [captures, setCaptures] = useState<Capture[]>([]);
  const [status, setStatus]     = useState<"idle"|"capturing"|"done"|"saving">("idle");
  const [message, setMessage]   = useState("");
  const capturedAngles = new Set(captures.map((c) => c.angle));

  // Next angle to capture
  const nextAngle = ANGLES.find((a) => !capturedAngles.has(a)) ?? null;

  // Auto-capture when angle matches and liveness=Real and NOT already registered
  useEffect(() => {
    if (status !== "capturing") return;
    if (result?.already_registered) return;
    if (!result?.ready_to_capture || !result.head_pose || !result.face_preview) return;
    const detectedAngle = result.head_pose.angle as Angle | null;
    if (!detectedAngle || detectedAngle !== nextAngle || capturedAngles.has(detectedAngle)) return;

    // Capture!
    setCaptures((prev) => [...prev, {
      angle: detectedAngle,
      face_b64: result.face_preview!,
      preview: `data:image/jpeg;base64,${result.face_preview}`,
    }]);
  }, [result, nextAngle, status, capturedAngles]);

  const startCapture = async () => {
    if (!empId.trim() || !empName.trim()) {
      setMessage("Vui lòng nhập đầy đủ Mã NV và Tên."); return;
    }
    // Check if face already belongs to an existing person
    if (result?.already_registered && result?.recognition?.name) {
      setMessage(`❌ Khuôn mặt này đã được đăng ký cho nhân viên "${result.recognition.name}" (Mã: ${result.recognition.employee_id}). Không thể đăng ký lại!`);
      return;
    }
    // Check if employee ID already exists
    try {
      const emps = await fetch(`${API_BASE}/api/employees`).then(r => r.json());
      if (emps.some((e: { id: string }) => e.id.toLowerCase() === empId.trim().toLowerCase())) {
        setMessage(`❌ Mã nhân viên "${empId}" đã tồn tại trong hệ thống.`); return;
      }
    } catch { /* ignore network error, backend will catch */ }
    setCaptures([]);
    setStatus("capturing");
    setMessage("");
  };


  const saveRegistration = async () => {
    if (captures.length < 5) { setMessage("Cần đủ 5 góc!"); return; }
    setStatus("saving");
    try {
      // Create employee first
      const empRes = await fetch(`${API_BASE}/api/employees`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: empId, name: empName }),
      });
      if (!empRes.ok) {
        const err = await empRes.json();
        setStatus("idle");
        setMessage(`❌ ${err.detail || "Mã nhân viên đã tồn tại."}`);
        return;
      }
      // Save embeddings
      const res = await fetch(`${API_BASE}/api/register/save`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          employee_id: empId,
          captures: captures.map((c) => ({ angle: c.angle, face_b64: c.face_b64 })),
        }),
      });
      const data = await res.json();
      setStatus("done");
      setMessage(`✅ Đăng ký thành công! Đã lưu ${data.saved_angles.length} góc mặt.`);
    } catch {
      setStatus("capturing");
      setMessage("❌ Lỗi khi lưu. Vui lòng thử lại.");
    }
  };

  const reset = () => {
    setEmpId(""); setEmpName(""); setCaptures([]);
    setStatus("idle"); setMessage("");
  };

  const live = result?.liveness;
  const pose = result?.head_pose;

  return (
    <main className="min-h-screen bg-gray-950 p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <Link href="/" className="text-gray-400 hover:text-white"><ArrowLeft /></Link>
          <h1 className="text-2xl font-bold text-white">👤 Đăng Ký Nhân Viên</h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: Form + Webcam */}
          <div>
            {/* Form */}
            <div className="bg-gray-800/60 rounded-xl p-4 mb-4 space-y-3">
              <div>
                <label className="text-sm text-gray-300 mb-1 block">Mã nhân viên</label>
                <input value={empId} onChange={(e) => setEmpId(e.target.value)}
                  disabled={status === "capturing" || status === "done"}
                  placeholder="VD: NV001"
                  className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 text-sm border border-gray-600 focus:border-blue-500 outline-none" />
              </div>
              <div>
                <label className="text-sm text-gray-300 mb-1 block">Họ và tên</label>
                <input value={empName} onChange={(e) => setEmpName(e.target.value)}
                  disabled={status === "capturing" || status === "done"}
                  placeholder="VD: Nguyễn Văn A"
                  className="w-full bg-gray-700 text-white rounded-lg px-3 py-2 text-sm border border-gray-600 focus:border-blue-500 outline-none" />
              </div>
            </div>

            {/* Webcam */}
            <div className="relative rounded-xl overflow-hidden bg-black">
              <WebcamCapture wsRef={wsRef} intervalMs={150} />

              {/* Head pose indicator */}
              {pose && (
                <div className="absolute top-2 left-2 bg-black/70 rounded-lg px-3 py-1 text-xs text-white">
                  Yaw: {pose.yaw}°  Pitch: {pose.pitch}°
                  {pose.angle && <span className="ml-2 text-yellow-300 font-medium">→ {pose.angle}</span>}
                </div>
              )}

              {/* Liveness */}
              {live && (
                <div className={`absolute top-2 right-2 px-3 py-1 rounded-lg text-xs font-medium
                  ${live.label==="Real" ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}>
                  {live.label} {(live.score*100).toFixed(0)}%
                </div>
              )}

              {/* Already registered warning on webcam */}
              {result?.already_registered && result?.recognition?.name && (
                <div className="absolute inset-x-3 top-10 bg-red-950/90 border border-red-500 text-red-200 text-xs px-3 py-2 rounded-lg text-center shadow-lg backdrop-blur">
                  ⚠️ <b>Khuôn mặt đã có trong hệ thống:</b> {result.recognition.name} ({result.recognition.employee_id}) - {((result.recognition.similarity || 0) * 100).toFixed(0)}%
                </div>
              )}

              {/* Current angle prompt / blocker */}
              {result?.already_registered ? (
                <div className="absolute bottom-2 left-0 right-0 mx-auto w-fit
                  bg-red-900/90 border border-red-500 text-red-100 text-xs sm:text-sm font-semibold
                  px-4 py-2 rounded-xl text-center shadow-lg">
                  ❌ Khuôn mặt đã đăng ký – Không thể chụp lại
                </div>
              ) : status === "capturing" && nextAngle ? (
                <div className="absolute bottom-2 left-0 right-0 mx-auto w-fit
                  bg-yellow-900/80 border border-yellow-500 text-yellow-200 text-sm font-medium
                  px-4 py-2 rounded-xl text-center">
                  {pose?.angle === nextAngle ? "📸 Đang chụp..." : `➡ Hãy: ${AngleLabels[nextAngle]}`}
                </div>
              ) : null}
            </div>
          </div>

          {/* Right: Angle guide + captures + actions */}
          <div>
            {/* Already registered notice card */}
            {result?.already_registered && result?.recognition?.name && (
              <div className="mb-4 p-4 rounded-xl bg-red-950/60 border border-red-500 text-red-200 text-sm">
                <div className="flex items-center gap-2 font-bold text-red-400">
                  <span>⚠️</span> Đã tìm thấy nhân viên trong cơ sở dữ liệu
                </div>
                <p className="mt-1 text-xs text-red-300">
                  Khuôn mặt này trùng khớp với <b>{result.recognition.name}</b> (Mã: <code>{result.recognition.employee_id}</code>).
                </p>
                <div className="mt-3 flex gap-2">
                  <Link href="/attendance" className="px-3 py-1.5 bg-blue-700 hover:bg-blue-600 text-white rounded-lg text-xs font-medium transition">
                    👉 Tới Điểm Danh
                  </Link>
                </div>
              </div>
            )}

            <AngleGuide captured={capturedAngles as Set<Angle>} current={nextAngle} />

            {/* Preview grid */}
            <div className="grid grid-cols-5 gap-2 mb-4">
              {ANGLES.map((angle) => {
                const cap = captures.find((c) => c.angle === angle);
                return (
                  <div key={angle} className="aspect-square rounded-lg overflow-hidden bg-gray-800 border border-gray-700">
                    {cap ? (
                      <img src={cap.preview} alt={angle} className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs">{angle}</div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Progress */}
            <div className="mb-4">
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>Tiến trình chụp</span>
                <span>{captures.length}/5</span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full bg-green-500 transition-all rounded-full"
                  style={{ width: `${(captures.length/5)*100}%` }} />
              </div>
            </div>

            {/* Message */}
            {message && (
              <div className={`mb-4 px-4 py-3 rounded-xl text-sm
                ${message.startsWith("✅") ? "bg-green-900/50 text-green-300 border border-green-700"
                : message.startsWith("❌") ? "bg-red-900/50 text-red-300 border border-red-700"
                : "bg-yellow-900/50 text-yellow-300 border border-yellow-700"}`}>
                {message}
              </div>
            )}

            {/* Action buttons */}
            <div className="space-y-3">
              {status === "idle" && (
                <button onClick={startCapture}
                  disabled={result?.already_registered}
                  className={`w-full font-semibold py-3 rounded-xl transition ${
                    result?.already_registered
                      ? "bg-gray-800 text-gray-500 border border-red-800/50 cursor-not-allowed"
                      : "bg-blue-600 hover:bg-blue-500 text-white"
                  }`}>
                  {result?.already_registered ? "⛔ Khuôn mặt đã tồn tại trong hệ thống" : "🎬 Bắt đầu chụp"}
                </button>
              )}
              {status === "capturing" && captures.length === 5 && (
                <button onClick={saveRegistration}
                  className="w-full bg-green-600 hover:bg-green-500 text-white font-semibold py-3 rounded-xl transition">
                  💾 Xác nhận & Lưu
                </button>
              )}

              {status === "done" && (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-green-400 font-medium">
                    <CheckCircle className="w-5 h-5" /> Đăng ký hoàn tất!
                  </div>
                  <button onClick={reset}
                    className="w-full bg-purple-700 hover:bg-purple-600 text-white font-semibold py-3 rounded-xl transition">
                    ➕ Đăng ký nhân viên mới
                  </button>
                  <Link href="/attendance"
                    className="w-full block text-center bg-blue-700 hover:bg-blue-600 text-white font-semibold py-3 rounded-xl transition">
                    📷 Tới Điểm Danh
                  </Link>
                </div>
              )}
              {(status === "capturing" && captures.length < 5) && (
                <button onClick={reset} className="w-full bg-gray-700 hover:bg-gray-600 text-gray-300 py-2 rounded-xl text-sm">
                  ↺ Làm lại
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

const AngleLabels: Record<Angle, string> = {
  front: "Nhìn thẳng vào camera",
  left:  "Quay mặt sang TRÁI",
  right: "Quay mặt sang PHẢI",
  up:    "Ngẩng đầu lên",
  down:  "Cúi đầu xuống",
};
