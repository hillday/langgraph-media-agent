from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from .config import Settings

def npx_program() -> str:
    return "npx.cmd" if os.name == "nt" else "npx"


def hyperframes_program() -> str:
    return "hyperframes.cmd" if os.name == "nt" else "hyperframes"


def resolve_hyperframes_command(settings: Settings) -> list[str]:
    if settings.hyperframes_bin:
        return [settings.hyperframes_bin]

    # Prefer a project-local installation (e.g. `npm i -D hyperframes`) to avoid `npx` fetching
    # and to work consistently across Windows/macOS/Linux and CI.
    local_bin = settings.repo_root / "node_modules" / ".bin" / hyperframes_program()
    if local_bin.exists():
        return [str(local_bin)]

    installed_hyperframes = shutil.which(hyperframes_program()) or shutil.which("hyperframes")
    if installed_hyperframes:
        return [installed_hyperframes]

    return [npx_program(), "--yes", "hyperframes"]


def run_hyperframes_command(
    settings: Settings,
    args: list[str],
    cwd: Path,
    *,
    check: bool = True,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [*resolve_hyperframes_command(settings), *args]
    try:
        return subprocess.run(
            command,
            cwd=settings.repo_root,
            text=True,
            capture_output=True,
            check=check,
            timeout=timeout_seconds or settings.hyperframes_command_timeout_seconds,
        )
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        output = ((stdout + "\n" + stderr).strip() or "(no output)")
        raise RuntimeError(
            f"HyperFrames command failed with exit code {exc.returncode}: {' '.join(command)}\n{output}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        def _to_text(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, bytes):
                # Be resilient: TimeoutExpired may carry bytes in some environments.
                return value.decode("utf-8", errors="replace")
            return str(value)

        stdout = _to_text(exc.stdout)
        stderr = _to_text(exc.stderr)
        output = ((stdout + "\n" + stderr).strip() or "(no output)")
        raise RuntimeError(
            f"HyperFrames command timed out after {timeout_seconds or settings.hyperframes_command_timeout_seconds}s: "
            f"{' '.join(command)}\n{output}"
        ) from exc


def render_video(settings: Settings, project_dir: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    return run_hyperframes_command(
        settings,
        ["render", str(project_dir), "--output", str(output_path)],
        project_dir,
        check=True,
        timeout_seconds=settings.hyperframes_render_timeout_seconds,
    )
