"use client";
import { useState } from "react";
import Link from "next/link";
import { CircleCheck, UserPlus } from "lucide-react";
import WebcamCapture from "@/components/WebcamCapture";
import FaceOverlay   from "@/components/FaceOverlay";
import AttendanceLog from "@/components/AttendanceLog";
import CheckLadder   from "@/components/CheckLadder";
import { useFrameSocket, AttendanceResult } from "@/lib/websocket";
import { buildChecks, livenessCheck } from "@/lib/checks";
import { formatTime } from "@/lib/format";

export default function AttendancePage() {
  const [result, setResult] = useState<AttendanceResult | null>(null);
  const [logged, setLogged] = useState<{ name: string; at: Date } | null>(null);
  const [frame, setFrame] = useState({ width: 640, height: 480 });

  const { wsRef, connected } = useFrameSocket<AttendanceResult>("/ws/attendance", (data) => {
    setResult(data);
    if (data.attendance_logged && data.recognition?.name) {
      setLogged({ name: data.recognition.name, at: new Date() });
      setTimeout(() => setLogged(null), 3000);
    }
  });

  const rec  = result?.recognition;
  const live = result?.liveness;
  const face = !!result?.face_detected;
  const liveness = livenessCheck(live?.label);

  const who = !face ? "idle" : rec?.matched ? "known" : rec ? "unknown" : "checking";
  const tag = rec?.matched ? rec.name : face ? (live?.label === "Real" ? "Unknown" : liveness.detail) : null;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Check-in</h1>
          <p className="lead">Face the camera. Liveness runs first, then identity and emotion.</p>
        </div>
        <span className="chip" data-tone={connected ? "ok" : "bad"} role="status">
          {connected ? "Backend connected" : "Backend unreachable, retrying"}
        </span>
      </div>

      <div className="workbench">
        <WebcamCapture wsRef={wsRef} intervalMs={200} onFrameSize={setFrame}>
          <FaceOverlay bbox={result?.bbox} frame={frame} state={face ? liveness.face : "pending"} tag={tag} />
        </WebcamCapture>

        <section className="panel" aria-labelledby="identity-name">
          <div className="identity" data-state={who}>
            <h2 id="identity-name" className="identity-name">
              {who === "known" ? rec!.name : who === "unknown" ? "Not registered" : who === "checking" ? "Checking this face" : "No one in view"}
            </h2>
            <div className="identity-meta">
              {who === "known" && <span className="measured measured--cool">{rec!.employee_id}</span>}
              {who === "unknown" && (
                <Link href="/register" className="link"><UserPlus size={14} aria-hidden="true" />Register this person</Link>
              )}
              {who === "checking" && <span>{liveness.detail}</span>}
              {who === "idle" && <span>Registered faces are checked in automatically.</span>}
            </div>
          </div>

          <CheckLadder checks={buildChecks(result)} />

          <div className="panel-section" role="status" aria-live="polite">
            {logged ? (
              <p className="notice" data-tone="ok">
                <CircleCheck size={16} aria-hidden="true" />
                <span>Checked in {logged.name} at <span className="measured">{formatTime(logged.at)}</span></span>
              </p>
            ) : (
              <p className="lead">A matched face is checked in once every 30 seconds.</p>
            )}
          </div>
        </section>
      </div>

      <AttendanceLog />
    </>
  );
}
