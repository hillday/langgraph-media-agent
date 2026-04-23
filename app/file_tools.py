from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ReadFileInput(BaseModel):
    path: str = Field(description="Absolute or relative file path to read.")


class WriteFileInput(BaseModel):
    path: str = Field(description="Absolute or relative file path to write.")
    content: str = Field(description="Full file contents to write.")


class ListDirInput(BaseModel):
    path: str = Field(description="Absolute or relative directory path to list.")


class PatchFileInput(BaseModel):
    path: str = Field(description="Absolute or relative file path to patch.")
    old_text: str = Field(description="Exact old text to replace.")
    new_text: str = Field(description="Replacement text.")
    replace_all: bool = Field(
        default=False,
        description="If true, replace all matches. If false, replace exactly one match.",
    )


class RunScriptInput(BaseModel):
    script_path: str = Field(description="Path to a Python script to run.")
    args: list[str] = Field(default_factory=list, description="Command-line arguments for the script.")
    timeout_seconds: int = Field(default=180, description="Max execution time.")


def _resolve_path(raw_path: str, *, base_dir: Path, allowed_roots: list[Path]) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    for root in allowed_roots:
        try:
            candidate.relative_to(root.resolve())
            return candidate
        except ValueError:
            continue
    raise PermissionError(f"Path is outside allowed roots: {candidate}")


def build_file_tools(
    *,
    base_dir: Path,
    writable_root: Path,
    readable_roots: list[Path],
    allowed_scripts: list[Path] | None = None,
) -> list[StructuredTool]:
    allowed_read_roots = [root.resolve() for root in readable_roots]
    allowed_write_roots = [writable_root.resolve()]
    allowed_script_paths = [p.resolve() for p in (allowed_scripts or [])]

    def read_file(path: str) -> str:
        target = _resolve_path(path, base_dir=base_dir, allowed_roots=allowed_read_roots)
        if not target.exists():
            return f"ERROR: file does not exist: {target}"
        if target.is_dir():
            return f"ERROR: path is a directory: {target}"
        return target.read_text(encoding="utf-8")

    def write_file(path: str, content: str) -> str:
        target = _resolve_path(path, base_dir=base_dir, allowed_roots=allowed_write_roots)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"WROTE: {target}"

    def list_dir(path: str) -> str:
        target = _resolve_path(path, base_dir=base_dir, allowed_roots=allowed_read_roots)
        if not target.exists():
            return f"ERROR: directory does not exist: {target}"
        if not target.is_dir():
            return f"ERROR: path is not a directory: {target}"

        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
            marker = "/" if child.is_dir() else ""
            size = "" if child.is_dir() else f" ({child.stat().st_size} bytes)"
            entries.append(f"{child.name}{marker}{size}")
        return "\n".join(entries) if entries else "(empty directory)"

    def patch_file(path: str, old_text: str, new_text: str, replace_all: bool = False) -> str:
        target = _resolve_path(path, base_dir=base_dir, allowed_roots=allowed_write_roots)
        if not target.exists():
            return f"ERROR: file does not exist: {target}"
        if target.is_dir():
            return f"ERROR: path is a directory: {target}"

        content = target.read_text(encoding="utf-8")
        occurrences = content.count(old_text)
        if occurrences == 0:
            return f"ERROR: old_text not found in {target}"
        if not replace_all and occurrences != 1:
            return (
                f"ERROR: old_text matched {occurrences} times in {target}. "
                "Refine the patch or set replace_all=true."
            )

        updated = content.replace(old_text, new_text, -1 if replace_all else 1)
        target.write_text(updated, encoding="utf-8")
        replaced_count = occurrences if replace_all else 1
        return f"PATCHED: {target} ({replaced_count} replacement(s))"

    def run_script(script_path: str, args: list[str], timeout_seconds: int = 180) -> str:
        if timeout_seconds < 1 or timeout_seconds > 1800:
            return "ERROR: timeout_seconds must be between 1 and 1800"

        script = _resolve_path(script_path, base_dir=base_dir, allowed_roots=allowed_read_roots)
        if script.resolve() not in allowed_script_paths:
            allowed_display = "\n".join(str(p) for p in allowed_script_paths) or "(none configured)"
            return (
                "ERROR: script is not in allowed_scripts whitelist.\n"
                f"Requested: {script}\n"
                "Allowed:\n"
                f"{allowed_display}"
            )

        if not script.exists() or script.is_dir():
            return f"ERROR: script does not exist or is a directory: {script}"

        try:
            result = subprocess.run(
                [sys.executable, str(script), *args],
                cwd=str(base_dir),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: script timed out after {timeout_seconds}s"

        out = (result.stdout or "") + (result.stderr or "")
        if len(out) > 8000:
            out = out[:8000] + "\n...(truncated)...\n"
        return f"EXIT_CODE: {result.returncode}\nOUTPUT:\n{out}"

    return [
        StructuredTool.from_function(
            func=list_dir,
            name="list_dir",
            description=(
                "List files and folders under a directory. Use before reading files when you need to discover "
                "available project, skill, or metadata files."
            ),
            args_schema=ListDirInput,
        ),
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description=(
                "Read a file that may contain skill references, pipeline metadata, creative briefs, "
                "or current project HTML. Use when you need exact file contents before authoring."
            ),
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            func=write_file,
            name="write_file",
            description=(
                "Write a file inside the current project directory. Use when you need to create or update "
                "index.html, helper JSON, or draft script files."
            ),
            args_schema=WriteFileInput,
        ),
        StructuredTool.from_function(
            func=patch_file,
            name="patch_file",
            description=(
                "Patch an existing file by replacing exact text. Prefer this for small targeted edits to "
                "index.html or helper files instead of rewriting the full file."
            ),
            args_schema=PatchFileInput,
        ),
        StructuredTool.from_function(
            func=run_script,
            name="run_script",
            description=(
                "Run a Python script from a strict allowlist and return stdout/stderr. "
                "Use for safe, deterministic pipeline steps (e.g. generating assets or building HTML) "
                "instead of executing arbitrary shell commands."
            ),
            args_schema=RunScriptInput,
        ),
    ]
