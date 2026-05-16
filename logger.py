"""
logger.py — Action logging for AI Business Intelligence App
Logs every prediction, upload, training run, and Gemini query to a JSONL file.
"""

import json
import os
import datetime
from typing import Any, Dict, Optional

from config import LOG_FILE


def _now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


def log_action(
    action: str,
    details: Optional[Dict[str, Any]] = None,
    status: str = "success",
) -> None:
    """Append one log entry (JSON line) to LOG_FILE."""
    entry = {
        "timestamp": _now(),
        "action": action,
        "status": status,
        "details": details or {},
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Never let logging crash the app
        print(f"[logger] WARNING: could not write log — {e}")


def read_logs() -> list:
    """Return all log entries as a list of dicts."""
    if not os.path.exists(LOG_FILE):
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries
