"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { CircleAlert, CircleCheck, LoaderCircle, RotateCcw } from "lucide-react";
import WebcamCapture from "@/components/WebcamCapture";
import FaceOverlay from "@/components/FaceOverlay";
import AngleGuide, { ANGLES, ANGLE_META, Angle } from "@/components/AngleGuide";
import { useFrameSocket, API_BASE, RegisterResult } from "@/lib/websocket";
import { livenessCheck } from "@/lib/checks";
import { INNOVATION } from "@/lib/flags";
import VoiceEnrollment from "@/components/VoiceEnrollment";

type Capture = { angle: Angle; face_b64: string; preview: string };
type Message = { tone: "ok" | "bad"; text: string };

export default function RegisterPage() {
  const [result, setResult]     = useState<RegisterResult | null>(null);
  const { wsRef } = useFrameSocket<RegisterResult>("/ws/register", setResult);
  const [empId, setEmpId]       = useState("");
  const [empName, setEmpName]   = useState("");
  const [captures, setCaptures] = useState<Capture[]>([]);
  const [status, setStatus]     = useState<"idle"|"capturing"|"done"|"saving">("idle");
  const [message, setMessage]   = useState<Message | null>(null);
  const [touched, setTouched]   = useState(false);
  const [frame, setFrame]       = useState({ width: 640, height: 480 });
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
    setTouched(true);
    if (!empId.trim() || !empName.trim()) {
      setMessage({ tone: "bad", text: "Enter both an ID and a full name before capturing." }); return;
    }
    // Check if face already belongs to an existing person
    if (result?.already_registered && result?.recognition?.name) {
      setMessage({ tone: "bad", text: `This face is already registered to ${result.recognition.name} (ID ${result.recognition.employee_id}). It cannot be registered again.` });
      return;
    }
    // Check if employee ID already exists
    try {
      const emps = await fetch(`${API_BASE}/api/employees`).then(r => r.json());
      if (emps.some((e: { id: string }) => e.id.toLowerCase() === empId.trim().toLowerCase())) {
        setMessage({ tone: "bad", text: `The ID ${empId} is already in use. Choose another ID.` }); return;
      }
    } catch { /* ignore network error, backend will catch */ }
    setCaptures([]);
    setStatus("capturing");
    setMessage(null);
  };

  const saveRegistration = async () => {
    if (captures.length < 5) { setMessage({ tone: "bad", text: "All five angles are needed before saving." }); return; }
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
        setMessage({ tone: "bad", text: err.detail || "This ID is already in use." });
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
      setMessage({ tone: "ok", text: `Registered ${empName} with ${data.saved_angles.length} face angles.` });
    } catch {
      setStatus("capturing");
      setMessage({ tone: "bad", text: "Saving failed because the backend did not answer. Try saving again." });
    }
  };

  const reset = () => {
    setEmpId(""); setEmpName(""); setCaptures([]);
    setStatus("idle"); setMessage(null); setTouched(false);
  };

  const live = result?.liveness;
  const pose = result?.head_pose;
  const liveness = livenessCheck(live?.label);
  const already = !!(result?.already_registered && result?.recognition?.name);
  const locked = status === "capturing" || status === "saving" || status === "done";
  const previews = Object.fromEntries(captures.map((c) => [c.angle, c.preview])) as Partial<Record<Angle, string>>;
  const idMissing = touched && !empId.trim();
  const nameMissing = touched && !empName.trim();

  const prompt = already
    ? { tone: "plain", text: "Already registered" }
    : status === "capturing" && nextAngle
      ? (pose?.angle === nextAngle ? { tone: "accent", text: "Hold still, capturing" } : { tone: "plain", text: ANGLE_META[nextAngle].instruction })
      : null;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Register a person</h1>
          <p className="lead">Enter their details, then capture five angles of their face.</p>
        </div>
      </div>

      <div className="workbench register-workbench">
        <WebcamCapture wsRef={wsRef} intervalMs={150} onFrameSize={setFrame}>
          <FaceOverlay bbox={result?.bbox} frame={frame} state={result?.face_detected ? liveness.face : "pending"} />
          <div className="stage-top">
            {pose ? (
              <span className="chip measured" aria-label={`Head yaw ${pose.yaw} degrees, pitch ${pose.pitch} degrees`}>
                yaw {pose.yaw}° pitch {pose.pitch}°
              </span>
            ) : <span />}
            {live && <span className="chip" data-tone={liveness.state === "pass" ? "ok" : liveness.state === "fail" ? "bad" : undefined}>{liveness.detail}</span>}
          </div>
          {prompt && <p className="stage-prompt" data-tone={prompt.tone} role="status">{prompt.text}</p>}
        </WebcamCapture>

        <section className="panel" aria-label="Registration">
          <div className="panel-section">
            <h2>Details</h2>
            <div className="form-grid gap-above">
              <div className="field">
                <label htmlFor="emp-id">ID</label>
                <input id="emp-id" name="employee_id" className="input" value={empId} onChange={(e) => setEmpId(e.target.value)}
                  disabled={locked} placeholder="NV001" autoComplete="off" spellCheck={false}
                  aria-invalid={idMissing || undefined} aria-describedby="emp-id-help" />
                <span id="emp-id-help" className="helper" data-tone={idMissing ? "error" : undefined}>{idMissing ? "Enter an ID." : "Unique, used in every record."}</span>
              </div>
              <div className="field">
                <label htmlFor="emp-name">Full name</label>
                <input id="emp-name" name="name" className="input" value={empName} onChange={(e) => setEmpName(e.target.value)}
                  disabled={locked} placeholder="Nguyen Van A" autoComplete="off"
                  aria-invalid={nameMissing || undefined} aria-describedby="emp-name-help" />
                <span id="emp-name-help" className="helper" data-tone={nameMissing ? "error" : undefined}>{nameMissing ? "Enter a full name." : "Shown at check-in."}</span>
              </div>
            </div>
          </div>

          <div className="panel-section">
            <h2>Face angles</h2>
            <p className="lead">Captured automatically when the head turns to the marked angle and liveness passes.</p>
            <div className="gap-above">
              <AngleGuide previews={previews} current={status === "capturing" ? nextAngle : null} />
            </div>
            <div className="steps-progress">
              <span><span className="measured">{captures.length}</span> of 5 captured</span>
              {status === "capturing" && nextAngle && <span>Next: {ANGLE_META[nextAngle].label.toLowerCase()}</span>}
            </div>
          </div>

          {(already || message) && (
            <div className="panel-section" role={message?.tone === "bad" || already ? "alert" : "status"}>
              {already ? (
                <div className="notice" data-tone="bad">
                  <CircleAlert size={16} aria-hidden="true" />
                  <div>
                    <p>This face is already registered to <strong>{result!.recognition!.name}</strong> (ID <span className="measured">{result!.recognition!.employee_id}</span>).</p>
                    <p><Link href="/attendance" className="link">Go to check-in</Link></p>
                  </div>
                </div>
              ) : message && (
                <div className="notice" data-tone={message.tone}>
                  {message.tone === "ok" ? <CircleCheck size={16} aria-hidden="true" /> : <CircleAlert size={16} aria-hidden="true" />}
                  <p>{message.text}</p>
                </div>
              )}
            </div>
          )}

          <div className="panel-section actions">
            {status === "idle" && (
              <button type="button" onClick={startCapture} disabled={already} className="btn btn--primary">
                Start capture
              </button>
            )}
            {status === "capturing" && captures.length === 5 && (
              <button type="button" onClick={saveRegistration} className="btn btn--primary">Save registration</button>
            )}
            {status === "saving" && (
              <button type="button" className="btn btn--primary" aria-busy="true" disabled>
                <LoaderCircle size={16} className="spin" aria-hidden="true" /> Saving
              </button>
            )}
            {status === "capturing" && (
              <button type="button" onClick={reset} className="btn btn--quiet">
                <RotateCcw size={16} aria-hidden="true" /> Start over
              </button>
            )}
            {status === "done" && (
              <>
                <button type="button" onClick={reset} className={INNOVATION ? "btn" : "btn btn--primary"}>Register another person</button>
                <Link href="/attendance" className="btn">Go to check-in</Link>
              </>
            )}
          </div>
        </section>
      </div>

      {INNOVATION && status === "done" && (
        <div className="section-gap">
          <VoiceEnrollment employeeId={empId} name={empName} />
        </div>
      )}
    </>
  );
}
