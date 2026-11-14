from __future__ import annotations

import subprocess


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"
