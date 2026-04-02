from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_env(start_dir: Optional[str] = None) -> Optional[Path]:
    """Find a .env file, preferring attacks/.env then repo-root .env."""

    start = Path(start_dir).resolve() if start_dir else Path(__file__).resolve().parent

    # Prefer attacks/.env
    candidate = (start / ".env") if start.is_dir() else None
    if candidate and candidate.exists():
        return candidate

    # Walk upwards looking for .env
    for p in [start, *start.parents]:
        c = p / ".env"
        if c.exists():
            return c

    return None


def load_env(start_dir: Optional[str] = None, override: bool = False) -> bool:
    """Load environment variables from the nearest .env (best-effort)."""

    try:
        from dotenv import load_dotenv
    except Exception:
        return False

    env_path = find_env(start_dir=start_dir)
    if not env_path:
        return False

    return bool(load_dotenv(dotenv_path=env_path, override=override))
