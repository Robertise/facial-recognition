"use client";
import { useEffect, useRef, useState } from "react";
import { Check, CircleAlert, LoaderCircle, Mic, RotateCcw, Square } from "lucide-react";
import { DEFAULT_PROCESSING, useAVRecorder } from "@/lib/media";
import {
  getInnovationConfig, startVoiceEnrollment, submitVoiceSample,
  VOICE_ISSUE_TEXT, type EnrollStart, type VoiceIssue,
} from "@/lib/innovation";

const MAX_RECORD_MS = 6000;
const COUNTDOWN_S = 3;

type Sample = { status: "todo" | "ok" | "retry"; issues: VoiceIssue[]; url: string | null };
type Phase = "consent" | "starting" | "ready" | "countdown" | "recording" | "checking" | "done" | "error";

/** Read three digit strings aloud; each good recording stores one voice embedding. */
export default function VoiceEnrollment({ employeeId, name, onComplete }: {
  employeeId: string;
  name: string;
  onComplete?: () => void;
}) {
  const recorder = useAVRecorder(null, DEFAULT_PROCESSING, { audioOnly: true });
  const [consent, setConsent] = useState(false);
  const [keepAudio, setKeepAudio] = useState(false);
  const [researchMode, setResearchMode] = useState(false);
  const [phase, setPhase] = useState<Phase>("consent");
  const [session, setSession] = useState<EnrollStart | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [count, setCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const stopTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const urls = useRef<string[]>([]);

  useEffect(() => {
    getInnovationConfig().then((c) => setResearchMode(c.research_mode)).catch(() => {});
    const made = urls.current;
    return () => made.forEach((u) => URL.revokeObjectURL(u));
  }, []);

  const current = samples.findIndex((s) => s.status !== "ok");
  const prompt = session && current >= 0 ? session.prompts[current] : null;

  const begin = async () => {
    setPhase("starting");
    setError(null);
    try {
      const s = await startVoiceEnrollment(employeeId, true);
      setSession(s);
      setSamples(s.prompts.map(() => ({ status: "todo", issues: [], url: null })));
      setPhase("ready");
    } catch (e) {
      setError(`Could not start voice enrolment: ${e instanceof Error ? e.message : String(e)}`);
      setPhase("error");
    }
  };

  const finish = async () => {
    if (stopTimer.current) clearTimeout(stopTimer.current);
    stopTimer.current = null;
    const blob = await recorder.stop();
    if (!blob || !prompt) return;
    const url = URL.createObjectURL(blob);
    urls.current.push(url);
    setPhase("checking");
    try {
      const r = await submitVoiceSample(employeeId, prompt.challenge_id, blob, keepAudio);
      setSamples((prev) => prev.map((s, i) => i === current
        ? { status: r.ok ? "ok" : "retry", issues: r.quality.issues, url } : s));
      if (r.enrolled) { setPhase("done"); onComplete?.(); } else setPhase("ready");
    } catch (e) {
      setError(`The recording could not be checked: ${e instanceof Error ? e.message : String(e)}`);
      setSamples((prev) => prev.map((s, i) => i === current ? { ...s, url } : s));
      setPhase("ready");
    }
  };

  const record = async () => {
    setError(null);
    setPhase("countdown");
    for (let n = COUNTDOWN_S; n > 0; n--) {
      setCount(n);
      await new Promise((r) => setTimeout(r, 1000));
    }
    if (!(await recorder.start())) { setPhase("ready"); return; }
    setPhase("recording");
    stopTimer.current = setTimeout(finish, MAX_RECORD_MS);
  };

  const okCount = samples.filter((s) => s.status === "ok").length;
  const need = session?.samples_needed ?? 3;

  return (
    <section className="panel" aria-labelledby="voice-title">
      <div className="panel-section">
        <h2 id="voice-title">Voice</h2>
        <p className="lead">{name} reads three short digit strings so check-in can confirm it is their voice.</p>
      </div>

      {phase === "consent" || phase === "starting" || (phase === "error" && !session) ? (
        <div className="panel-section">
          <p className="consent">
            Each recording becomes a voice embedding of 192 numbers, stored in this system&apos;s database on this machine.
            The audio itself is not kept unless you choose to keep it below. Removing the person from People deletes it.
          </p>
          <label className="check gap-above">
            <input type="checkbox" name="voice_consent" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
            <span>{name} agrees to their voice being stored for verification</span>
          </label>
          {researchMode && (
            <label className="check">
              <input type="checkbox" name="keep_audio" checked={keepAudio} onChange={(e) => setKeepAudio(e.target.checked)} />
              <span>Also keep the recordings for research (saved under data/innovation)</span>
            </label>
          )}
          <div className="actions gap-above">
            <button type="button" className="btn btn--primary" onClick={begin} disabled={!consent || phase === "starting"}
              aria-busy={phase === "starting" || undefined}>
              {phase === "starting" ? <LoaderCircle size={16} className="spin" aria-hidden="true" /> : <Mic size={16} aria-hidden="true" />}
              Start voice enrolment
            </button>
          </div>
        </div>
      ) : (
        <>
          <ol className="prompts">
            {session?.prompts.map((p, i) => {
              const s = samples[i];
              const active = i === current && phase !== "done";
              return (
                <li key={p.challenge_id} className="prompt" data-state={s.status === "ok" ? "done" : active ? "current" : "todo"}>
                  <div className="prompt-head">
                    <span className="prompt-label">Recording {i + 1}</span>
                    {s.status === "ok" && <span className="chip" data-tone="ok"><Check size={14} aria-hidden="true" />Saved</span>}
                  </div>
                  <p className="digits" aria-label={`Read: ${p.digits.split("").join(" ")}`}>{p.digits}</p>
                  {s.status === "retry" && (
                    <ul className="issues" role="alert">
                      {s.issues.map((code) => <li key={code}><CircleAlert size={14} aria-hidden="true" />{VOICE_ISSUE_TEXT[code]}</li>)}
                    </ul>
                  )}
                  {s.url && <audio className="playback" src={s.url} controls aria-label={`Recording ${i + 1}`} />}
                </li>
              );
            })}
          </ol>

          {phase !== "done" && (
            <div className="panel-section">
              <div className="level">
                <Mic size={16} aria-hidden="true" />
                <div className="meter" role="meter" aria-label="Microphone level" aria-valuemin={0} aria-valuemax={1}
                  aria-valuenow={Number(recorder.level.toFixed(2))}>
                  <span style={{ transform: `scaleX(${recorder.level})` }} />
                </div>
              </div>
              <div className="actions gap-above">
                {phase === "recording" ? (
                  <button type="button" className="btn btn--primary" onClick={finish}>
                    <Square size={14} aria-hidden="true" />Stop
                  </button>
                ) : (
                  <button type="button" className="btn btn--primary" onClick={record}
                    disabled={phase !== "ready" || !prompt} aria-busy={phase === "checking" || undefined}>
                    {phase === "checking" ? <LoaderCircle size={16} className="spin" aria-hidden="true" />
                      : samples[current]?.status === "retry" ? <RotateCcw size={16} aria-hidden="true" /> : <Mic size={16} aria-hidden="true" />}
                    {phase === "countdown" ? `Starting in ${count}` : phase === "checking" ? "Checking"
                      : samples[current]?.status === "retry" ? `Record ${current + 1} again` : `Record ${current + 1}`}
                  </button>
                )}
              </div>
              <p className="helper gap-above" role="status">
                {phase === "recording" ? "Recording. Read the digits, then press Stop." : `${okCount} of ${need} saved. Recording stops by itself after 6 seconds.`}
              </p>
            </div>
          )}

          {phase === "done" && (
            <div className="panel-section" role="status">
              <p className="notice" data-tone="ok"><Check size={16} aria-hidden="true" />Voice enrolled for {name} ({okCount} of {need}).</p>
            </div>
          )}
        </>
      )}

      {(error || recorder.error) && (
        <div className="panel-section" role="alert">
          <p className="notice" data-tone="bad"><CircleAlert size={16} aria-hidden="true" />{error ?? recorder.error}</p>
        </div>
      )}
    </section>
  );
}
