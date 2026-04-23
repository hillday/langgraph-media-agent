from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .models import SessionData


def create_session_dir(runs_dir: Path) -> tuple[str, Path]:
    session_id = uuid4().hex[:12]
    session_dir = runs_dir / session_id
    (session_dir / "uploads").mkdir(parents=True, exist_ok=True)
    (session_dir / "project").mkdir(parents=True, exist_ok=True)
    return session_id, session_dir


def session_file(session_dir: Path) -> Path:
    return session_dir / "session.json"


def save_session(session_dir: Path, session: SessionData) -> None:
    session_file(session_dir).write_text(session.model_dump_json(indent=2), encoding="utf-8")


def load_session(session_dir: Path) -> SessionData:
    return SessionData.model_validate(json.loads(session_file(session_dir).read_text(encoding="utf-8")))


def append_progress(session: SessionData, message: str) -> SessionData:
    progress = [*session.progress, message]
    if len(progress) > 200:
        progress = progress[-200:]
    return session.model_copy(update={"progress": progress, "status_message": message})
