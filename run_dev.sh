#!/usr/bin/env bash
# Starts the FastAPI backend and the Streamlit frontend together, in one
# terminal, with one command. They're still two separate processes under the
# hood — that's inherent, Streamlit has no backend of its own — this script
# just launches the API in the background and the frontend in the
# foreground, and stops both together on Ctrl+C.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

if [ ! -f data/pke.sqlite3 ]; then
  echo "No database found — initializing..."
  python -m app.db.init_db
fi

echo "Starting API server on http://127.0.0.1:8000 ..."
uvicorn app.main:app --reload &
API_PID=$!

cleanup() {
  echo ""
  echo "Stopping API server (pid $API_PID)..."
  kill "$API_PID" 2>/dev/null || true
  wait "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 2  # give the API a moment to finish booting before Streamlit starts hitting it

echo "Starting Streamlit frontend on http://localhost:8501 ..."
streamlit run streamlit_app.py
