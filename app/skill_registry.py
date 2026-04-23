from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)
FIELD_RE = re.compile(r'^([A-Za-z0-9_-]+):\s*"?(.+?)"?$', re.MULTILINE)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass
class SkillDefinition:
    name: str
    description: str
    body: str
    expanded_body: str
    referenced_paths: tuple[Path, ...]
    path: Path


class SkillRegistry:
    def __init__(self, skill_dirs: list[Path] | tuple[Path, ...]):
        self.skill_dirs = list(skill_dirs)
        self.skills = self._load_skills()

    def _load_skills(self) -> dict[str, SkillDefinition]:
        loaded: dict[str, SkillDefinition] = {}
        for skill_dir in self.skill_dirs:
            if not skill_dir.exists():
                continue
            for path in skill_dir.glob("*/SKILL.md"):
                parsed = self._parse_skill(path)
                if parsed:
                    loaded[parsed.name] = parsed
        return loaded

    def _parse_skill(self, path: Path) -> SkillDefinition | None:
        content = path.read_text(encoding="utf-8")
        match = FRONTMATTER_RE.match(content)
        if not match:
            return None
        frontmatter, body = match.groups()
        fields = {key: value.strip() for key, value in FIELD_RE.findall(frontmatter)}
        name = fields.get("name", path.parent.name)
        description = fields.get("description", "")
        body = body.strip()
        referenced_paths = tuple(self._collect_referenced_markdown(path, body))
        expanded_body = self._expand_skill_body(path, body, referenced_paths)
        return SkillDefinition(
            name=name,
            description=description,
            body=body,
            expanded_body=expanded_body,
            referenced_paths=referenced_paths,
            path=path,
        )

    def _collect_referenced_markdown(self, skill_path: Path, body: str) -> list[Path]:
        skill_root = skill_path.parent
        seen: set[Path] = set()
        referenced: list[Path] = []

        for raw_target in MARKDOWN_LINK_RE.findall(body):
            target = raw_target.strip()
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue

            target = target.split("#", 1)[0].strip()
            if not target.endswith(".md"):
                continue

            candidate = (skill_root / target).resolve()
            try:
                candidate.relative_to(skill_root.resolve())
            except ValueError:
                continue

            if not candidate.exists() or not candidate.is_file() or candidate in seen:
                continue

            seen.add(candidate)
            referenced.append(candidate)

        return referenced

    def _expand_skill_body(self, skill_path: Path, body: str, referenced_paths: tuple[Path, ...]) -> str:
        if not referenced_paths:
            return body

        sections = [body, "## Referenced Documents (Auto-loaded)"]
        for referenced_path in referenced_paths:
            relative_path = referenced_path.relative_to(skill_path.parent).as_posix()
            sections.append(
                f"### {relative_path}\nPath: {referenced_path}\n\n{referenced_path.read_text(encoding='utf-8').strip()}"
            )
        return "\n\n".join(sections).strip()

    def list_brief(self) -> list[dict[str, str]]:
        return [
            {"name": skill.name, "description": skill.description, "path": str(skill.path)}
            for skill in sorted(self.skills.values(), key=lambda item: item.name)
        ]

    def get(self, name: str) -> SkillDefinition | None:
        return self.skills.get(name)

    def build_context(self, selected_skills: list[str]) -> str:
        sections: list[str] = []
        for skill_name in selected_skills:
            skill = self.get(skill_name)
            if not skill:
                continue
            referenced_docs = ", ".join(path.relative_to(skill.path.parent).as_posix() for path in skill.referenced_paths) or "(none)"
            sections.append(
                f"## Skill: {skill.name}\nPath: {skill.path}\nDescription: {skill.description}\nReferenced docs: {referenced_docs}\n\n{skill.expanded_body}"
            )
        return "\n\n".join(sections)
