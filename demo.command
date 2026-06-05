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

# 2) Ensure the model artifacts are present. download_artifacts.py is idempotent:
#    if every required file is already in backend/artifacts/ it just prints
#    "already present" and exits 0; otherwise it downloads + extracts artifacts.zip
#    from the GitHub Release. This lets anyone run the demo without manually
#    fetching the artifacts first.
echo "[info] checking model artifacts ..."
if python download_artifacts.py; then
  echo "[ok] model artifacts ready"
else
  echo "ERROR : could not obtain the model artifacts (see message above)."
  echo "        Check your internet connection, then relaunch the demo."
  exit 1
fi

# 3) Display the current Git branch (if any) — useful to check that we're on the right version for the demo.
BRANCH=$(git branch --show-current 2>/dev/null || echo "?")
echo "[info] git branch : $BRANCH"

# 4) Open the web browser on the local URL of the site (after a short delay to let the server start).
URL="http://127.0.0.1:8000"
( sleep 3; open "$URL" >/dev/null 2>&1 || true ) &

echo "[ok] the site will open at $URL"
echo "----------------------------------------------"
echo "  Choose the profile \"Lenny\" for an immediate demo."
echo "  To stop : Ctrl + C here."
echo "----------------------------------------------"

# 5) Launch the backend server with Uvicorn (FastAPI) — it will serve the API for the frontend.
exec uvicorn backend.main:app
