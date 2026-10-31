"""
main.py – FastAPI application
Run: uvicorn backend.main:app --reload --port 8000
"""
import json, asyncio, traceback, time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from backend.database import (
    init_db, create_employee, get_all_employees, delete_employee,
    save_embedding, get_all_embeddings, log_attendance,
    get_today_attendance, get_attendance_history, already_checked_in_today
)
from backend.processing import (
    decode_frame, process_attendance_frame,
    process_register_frame, extract_embedding_from_b64,
    _get_recognizer, _get_liveness, _get_head_pose, _get_emotion
)
from liveness.temporal import make_smoother
from innovation import config as innovation_config


app = FastAPI(title="Face Attendance API")

# The innovation extension (voice, spoken challenge, AV sync) is off unless
# INNOVATION_ENABLED=1; without it the API is exactly the attendance system.
if innovation_config.enabled():
    from innovation.api import router as innovation_router
    app.include_router(innovation_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    init_db()
    print("[API] Database initialized.")
    if innovation_config.enabled():
        from innovation.db import init_innovation_db
        init_innovation_db()
        print("[API] Innovation extension enabled.")
    # Load every model now so a broken dependency shows up in the startup log
    # instead of silently killing the first WebSocket connection.
    for name, loader in [("liveness", _get_liveness), ("head_pose", _get_head_pose),
                         ("emotion", _get_emotion), ("recognizer", _get_recognizer)]:
        try:
            loader()
        except Exception as e:
            print(f"[API] ERROR loading {name} model: {type(e).__name__}: {e}")
            traceback.print_exc()

# ── WebSocket: Attendance ─────────────────────────────────────────────────────

@app.websocket("/ws/attendance")
async def ws_attendance(ws: WebSocket):
    await ws.accept()
    smoother = make_smoother(_get_liveness().config)
    # Track recently logged employees to avoid duplicate logs (30s debounce)
    recent_logs: dict[str, float] = {}
    try:
        while True:
            data = await ws.receive_text()
            t    = time.monotonic()
            msg  = json.loads(data)
            frame = decode_frame(msg.get("frame", ""))
            if frame is None:
                await ws.send_json({"error": "invalid frame"})
                continue

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, process_attendance_frame, frame, smoother, t
                )
            except Exception as e:
                traceback.print_exc()
                await ws.send_json({"error": f"{type(e).__name__}: {e}"})
                continue

            # Auto-log attendance (debounce 30s per employee)
            rec = result.get("recognition") or {}
            if rec.get("matched") and rec.get("employee_id"):
                eid  = rec["employee_id"]
                now  = time.time()
                last = recent_logs.get(eid, 0)
                if now - last > 30:
                    recent_logs[eid] = now
                    emo   = (result.get("emotion") or {}).get("label", "Unknown")
                    live  = (result.get("liveness") or {}).get("score", 0.0)
                    sim   = rec.get("similarity", 0.0)
                    log_attendance(eid, emo, live, sim)
                    result["attendance_logged"] = True
                else:
                    result["attendance_logged"] = False

            await ws.send_json(result)

    except WebSocketDisconnect:
        pass


# ── WebSocket: Registration ───────────────────────────────────────────────────

@app.websocket("/ws/register")
async def ws_register(ws: WebSocket):
    await ws.accept()
    smoother = make_smoother(_get_liveness().config)
    try:
        while True:
            data  = await ws.receive_text()
            t     = time.monotonic()
            msg   = json.loads(data)
            frame = decode_frame(msg.get("frame", ""))
            if frame is None:
                await ws.send_json({"error": "invalid frame"})
                continue

            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, process_register_frame, frame, smoother, t
                )
            except Exception as e:
                traceback.print_exc()
                await ws.send_json({"error": f"{type(e).__name__}: {e}"})
                continue
            await ws.send_json(result)

    except WebSocketDisconnect:
        pass


# ── REST: Employees ───────────────────────────────────────────────────────────

class EmployeeCreate(BaseModel):
    id: str
    name: str

@app.get("/api/employees")
def list_employees():
    return get_all_employees()

@app.post("/api/employees", status_code=201)
def add_employee(body: EmployeeCreate):
    ok = create_employee(body.id, body.name)
    if not ok:
        raise HTTPException(400, "Employee ID already exists")
    return {"id": body.id, "name": body.name}

@app.delete("/api/employees/{emp_id}")
def remove_employee(emp_id: str):
    delete_employee(emp_id)
    return {"deleted": emp_id}


# ── REST: Registration Save ───────────────────────────────────────────────────

class AngleCapture(BaseModel):
    angle: str
    face_b64: str  # base64 JPEG face crop

class RegisterSave(BaseModel):
    employee_id: str
    captures: list[AngleCapture]

@app.post("/api/register/save")
def register_save(body: RegisterSave):
    # Verify that face doesn't already belong to an existing person
    db_embs = get_all_embeddings()
    if db_embs:
        rec = _get_recognizer()
        for cap in body.captures:
            emb = extract_embedding_from_b64(cap.face_b64)
            if emb is not None:
                match = rec.find_match(emb, db_embs)
                if match["matched"] and match["employee_id"] != body.employee_id:
                    raise HTTPException(
                        status_code=400,
                        detail=f"This face is already registered to {match['name']} (ID {match['employee_id']})."
                    )

    saved = []
    for cap in body.captures:
        emb = extract_embedding_from_b64(cap.face_b64)
        if emb is not None:
            save_embedding(body.employee_id, cap.angle, emb)
            saved.append(cap.angle)
    return {"employee_id": body.employee_id, "saved_angles": saved}



# ── REST: Attendance ──────────────────────────────────────────────────────────

@app.get("/api/attendance/today")
def today():
    return get_today_attendance()

@app.get("/api/attendance/history")
def history(limit: int = 100):
    return get_attendance_history(limit)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}
