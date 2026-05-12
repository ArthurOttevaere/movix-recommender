"""Submit a predictions DataFrame to the MLSMM2156 hackathon API.

Usage in notebook:

    from python_helper import submit_predictions
    submit_predictions(df_predictions)

Reads HACKATHON_URL and HACKATHON_TOKEN from the repo's `.env` file (loaded
automatically on import) or from the process environment. They can also be
passed as keyword args.
"""
from __future__ import annotations

import io
import os
from typing import Optional

import pandas as pd

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "requests is required to submit predictions. "
        "Install it with: pip install requests"
    ) from exc

# Auto-load .env from the working directory (or any parent), so students just
# put HACKATHON_URL + HACKATHON_TOKEN in .env and the notebook "just works".
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; fall back to plain os.environ


REQUIRED_COLUMNS = ["userId", "movieId", "rating"]


def submit_predictions(
    df: pd.DataFrame,
    token: Optional[str] = None,
    url: Optional[str] = None,
    timeout: int = 60,
) -> dict:
    """POST a predictions DataFrame to /api/submit.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ['userId', 'movieId', 'rating'] in that order, with
        rows matching the hidden test set's order.
    token : str, optional
        Bearer token for the group. Defaults to env var HACKATHON_TOKEN.
    url : str, optional
        Base URL of the deployed app. Defaults to env var HACKATHON_URL.
    timeout : int
        HTTP timeout in seconds.

    Returns
    -------
    dict
        Parsed JSON response from the server.
    """
    token = token or os.environ.get("HACKATHON_TOKEN")
    url = url or os.environ.get("HACKATHON_URL")
    if not token:
        raise ValueError(
            "No HACKATHON_TOKEN found. Add it to your repo-root .env file "
            "(e.g. `HACKATHON_TOKEN=...`) and restart the kernel, or pass "
            "token=... explicitly."
        )
    if not url:
        raise ValueError(
            "No HACKATHON_URL found. Add it to your repo-root .env file "
            "(e.g. `HACKATHON_URL=https://recsys-hackathon.vercel.app`) and "
            "restart the kernel, or pass url=... explicitly."
        )

    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"DataFrame columns must be {REQUIRED_COLUMNS} in that order; "
            f"got {list(df.columns)}"
        )

    csv_text = df.to_csv(index=False)
    endpoint = url.rstrip("/") + "/api/submit"

    try:
        resp = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            files={
                "file": (
                    "ratings_predictions.csv",
                    csv_text,
                    "text/csv",
                ),
            },
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        print(f"Network error: {exc}")
        return {"error": str(exc)}

    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text}

    _print_summary(resp.status_code, body)
    return body


def _print_summary(status_code: int, body: dict) -> None:
    line = "-" * 60
    print(line)
    if status_code == 200 and body.get("status") == "ok":
        rmse = body.get("rmse")
        rank = body.get("rank")
        best = body.get("best_rmse")
        rem = body.get("remaining", {}) or {}
        print("  [OK] Submission accepted")
        if rmse is not None:
            print(f"       RMSE this submission   {rmse:.4f}")
        if best is not None:
            print(f"       Best RMSE for group    {best:.4f}")
        if rank is not None:
            print(f"       Current rank           #{rank}")
        print(f"       Remaining this hour    {rem.get('hour', '?')}/10")
        print(f"       Remaining total        {rem.get('total', '?')}/300")
    elif status_code == 200 and body.get("status") == "error":
        rem = body.get("remaining", {}) or {}
        print("  [VALIDATION ERROR] Server scored your submission as failed")
        err = body.get("error") or "unknown"
        for line_err in err.split("\n"):
            print(f"       {line_err}")
        print(f"       Remaining this hour    {rem.get('hour', '?')}/10")
        print(f"       Remaining total        {rem.get('total', '?')}/300")
    elif status_code == 401:
        print("  [UNAUTHORIZED] Check that HACKATHON_TOKEN is correct.")
    elif status_code == 403:
        print(f"  [FORBIDDEN] {body.get('error', 'hackathon is not open')}")
    elif status_code == 429:
        retry = body.get("retry_after_seconds")
        print(f"  [RATE LIMITED] {body.get('error', '')}")
        if retry is not None:
            print(f"       Retry after {retry}s")
        print(f"       Hour remaining   {body.get('hour_remaining', '?')}/10")
        print(f"       Total remaining  {body.get('total_remaining', '?')}/300")
    else:
        print(f"  [HTTP {status_code}] {body.get('error', body)}")
    print(line)


def check_quota(
    token: Optional[str] = None,
    url: Optional[str] = None,
    timeout: int = 30,
) -> dict:
    """Read-only check of your group's remaining hackathon quota.

    Hits GET /api/quota — does not count toward your limits. Useful before
    a submission to know whether you're close to a cap or in cooldown.

    Returns the parsed JSON response and prints a one-line summary.
    """
    token = token or os.environ.get("HACKATHON_TOKEN")
    url = url or os.environ.get("HACKATHON_URL")
    if not token:
        raise ValueError(
            "No HACKATHON_TOKEN found. Add it to your repo-root .env file "
            "(e.g. `HACKATHON_TOKEN=...`) and restart the kernel."
        )
    if not url:
        raise ValueError(
            "No HACKATHON_URL found. Add it to your repo-root .env file "
            "(e.g. `HACKATHON_URL=https://recsys-hackathon.vercel.app`) and "
            "restart the kernel."
        )

    endpoint = url.rstrip("/") + "/api/quota"
    try:
        resp = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except requests.exceptions.RequestException as exc:
        print(f"Network error: {exc}")
        return {"error": str(exc)}

    try:
        body = resp.json()
    except ValueError:
        body = {"error": resp.text}

    _print_quota(resp.status_code, body)
    return body


def _print_quota(status_code: int, body: dict) -> None:
    line = "-" * 60
    print(line)
    if status_code == 200:
        group_name = (body.get("group") or {}).get("name", "your group")
        print(f"  [QUOTA] {group_name}")
        print(
            f"       This hour          {body.get('hour_used', '?')}/{body.get('hour_limit', '?')}"
            f" used  ({body.get('hour_remaining', '?')} remaining)"
        )
        print(
            f"       Total              {body.get('total_used', '?')}/{body.get('total_limit', '?')}"
            f" used  ({body.get('total_remaining', '?')} remaining)"
        )
        last = body.get("last_submitted_at")
        nxt = body.get("next_allowed_at")
        print(f"       Last submission    {last or '(none yet)'}")
        if nxt:
            print(f"       Next allowed at    {nxt}  (cooldown active)")
        else:
            print(f"       Cooldown           clear — you can submit now")
    elif status_code == 401:
        print("  [UNAUTHORIZED] Check that HACKATHON_TOKEN is correct.")
    else:
        print(f"  [HTTP {status_code}] {body.get('error', body)}")
    print(line)


__all__ = ["submit_predictions", "check_quota"]
