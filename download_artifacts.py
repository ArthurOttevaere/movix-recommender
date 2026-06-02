"""
Download the pre-trained model artifacts so the app can run without retraining.

The backend reads everything it needs from ``backend/artifacts/``. Those files are
too large for git (``content_features.pkl`` alone is ~290 MB), so they are published
as a single ``artifacts.zip`` asset on a GitHub Release instead. This script fetches
and extracts that archive — the files are byte-identical to the trained models, so
model performance is exactly the same as the authors' (no retraining involved).

Usage:
    python download_artifacts.py            # download + extract (skips if present)
    python download_artifacts.py --force    # re-download even if files exist

The download URL can be overridden with the MOVIX_ARTIFACTS_URL environment variable:
    MOVIX_ARTIFACTS_URL=https://.../artifacts.zip python download_artifacts.py
"""

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

# ── Where to get the archive ─────────────────────────────────────────────────
# URL of the `artifacts.zip` asset attached to the project's GitHub Release.
# Already configured — no action needed. Can be overridden with the
# MOVIX_ARTIFACTS_URL environment variable.
ARTIFACTS_URL = os.environ.get(
    "MOVIX_ARTIFACTS_URL",
    "https://github.com/ArthurOttevaere/Recommender_System_Assignments/releases/download/v1.0-artifacts/artifacts.zip",
)

ARTIFACTS_DIR = Path(__file__).parent / "backend" / "artifacts"

# Files the backend needs at runtime (used to detect "already downloaded").
REQUIRED_FILES = [
    "content_features.pkl",
    "ials_model.pkl",
    "bpr_model.pkl",
    "userbased_model.pkl",
    "rating_matrix.npz",
    "movies.csv",
    "links.csv",
    "popularity.csv",
]


def _already_present() -> bool:
    return all((ARTIFACTS_DIR / name).exists() for name in REQUIRED_FILES)


def _download(url: str, dest: Path) -> None:
    """Stream `url` to `dest`, printing a simple progress bar."""
    req = Request(url, headers={"User-Agent": "movix-artifacts-downloader"})
    with urlopen(req) as resp:  # noqa: S310 (trusted, user-provided URL)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        chunk = 1 << 20  # 1 MB
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = done / total * 100
                    bar = "█" * int(pct // 4)
                    print(f"\r  [{bar:<25}] {pct:5.1f}%  ({done >> 20}/{total >> 20} MB)",
                          end="", flush=True)
                else:
                    print(f"\r  {done >> 20} MB", end="", flush=True)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download pre-trained model artifacts.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if the artifacts already exist.")
    args = parser.parse_args()

    if _already_present() and not args.force:
        print(f"Artifacts already present in {ARTIFACTS_DIR} — nothing to do.")
        print("Use --force to re-download.")
        return 0

    if "<owner>" in ARTIFACTS_URL:
        print("ERROR: no download URL configured.", file=sys.stderr)
        print("Set MOVIX_ARTIFACTS_URL, or edit ARTIFACTS_URL at the top of this file,",
              file=sys.stderr)
        print("to point at the artifacts.zip asset of your GitHub Release.", file=sys.stderr)
        return 1

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading artifacts from:\n  {ARTIFACTS_URL}")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _download(ARTIFACTS_URL, tmp_path)
        print(f"Extracting into {ARTIFACTS_DIR} ...")
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(ARTIFACTS_DIR)
    finally:
        tmp_path.unlink(missing_ok=True)

    missing = [name for name in REQUIRED_FILES if not (ARTIFACTS_DIR / name).exists()]
    if missing:
        print(f"WARNING: archive extracted but these expected files are missing: {missing}",
              file=sys.stderr)
        return 1

    print("Done. All required artifacts are in place — you can now run the app.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
