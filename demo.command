#!/usr/bin/env bash
#
# demo.sh — launch Movix for the live demo (backend + site web).
#   Double-click, or from the terminal : ./demo.sh
#   Stop the server : Ctrl + C dans cette fenêtre.
#
set -e

# Locate itself at the root of the project (where this script lives), whatever the folder from which it was launched.
cd "$(dirname "$0")"

echo "=============================================="
echo "  Movix — launching the demo"
echo "=============================================="

# 1) Active the Python environment (venv) — if not found, exit with an error message.
if [ ! -d "venv" ]; then
  echo "ERROR : venv file not found. Launch the installation first (see README)."
  exit 1
fi
source venv/bin/activate
echo "[ok] Python environment activated (venv)"

# 2) Display the current Git branch (if any) — useful to check that we're on the right version for the demo.
BRANCH=$(git branch --show-current 2>/dev/null || echo "?")
echo "[info] git branch : $BRANCH"

# 3) Open the web browser on the local URL of the site (after a short delay to let the server start).
URL="http://127.0.0.1:8000"
( sleep 3; open "$URL" >/dev/null 2>&1 || true ) &

echo "[ok] the site will open at $URL"
echo "----------------------------------------------"
echo "  Choose the profile \"Lenny\" for an immediate demo."
echo "  To stop : Ctrl + C here."
echo "----------------------------------------------"

# 4) Launch the backend server with Uvicorn (FastAPI) — it will serve the API for the frontend.
exec uvicorn backend.main:app
