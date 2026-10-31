#!/bin/bash
# Start backend + frontend
echo "Starting Face Attendance System..."

# Backend
cd "$(dirname "$0")"
uvicorn backend.main:app --reload --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID (http://localhost:8000)"

# Frontend
cd frontend && npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID (http://localhost:3000)"

echo ""
echo "Open: http://localhost:3000"
echo "Press Ctrl+C to stop both servers"

wait
