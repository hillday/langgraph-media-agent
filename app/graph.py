from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict, cast

from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .config import Settings
from .file_tools import build_file_tools
from .hyperframes_runner import render_video, run_hyperframes_command
from .models import PlanResult, VerificationResult
from .pipeline_tools import build_pipeline_payload, write_pipeline_file
from .skill_registry import SkillRegistry
from .storage import append_progress, load_session, save_session

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AgentState(TypedDict, total=False):
    session_id: str
    session_dir: str
    user_request: str
    uploaded_images: list[str]
    feedback_history: list[str]
    stage: str
    plan: dict[str, Any]
    clarification_needed: bool
    clarification_questions: list[str]
    selected_skills: list[str]
    project_dir: str
    pipeline_path: str
    resolved_pipeline_path: str
    creative_brief_path: str
    render_output_path: str
    lint_output: str
    validate_output: str
    verification: dict[str, Any]
    html_revision_count: int
    last_error: str


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return normalized or f"composition-{int(time.time())}"


def _serialize_for_log(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        return repr(value)


def _message_to_log_dict(message: Any) -> dict[str, Any]:
    return {
        "type": message.__class__.__name__,
        "content": getattr(message, "content", None),
        "name": getattr(message, "name", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "tool_calls": getattr(message, "tool_calls", None),
        "additional_kwargs": getattr(message, "additional_kwargs", None),
    }


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(candidate[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object.")
    return parsed


def extract_html_document(text: str) -> str:
    candidate = text.strip()
    fenced_match = re.search(r"```(?:html)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidate = fenced_match.group(1).strip()

    doctype_match = re.search(r"(<!doctype\s+html[\s\S]*?</html>)", candidate, re.IGNORECASE)
    if doctype_match:
        return doctype_match.group(1).strip()

    html_match = re.search(r"(<html[\s\S]*?</html>)", candidate, re.IGNORECASE)
    if html_match:
        return html_match.group(1).strip()

    raise ValueError("Model response does not contain a complete HTML document.")


def image_path_to_data_url(image_path: str) -> str:
    file_path = Path(image_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Uploaded image not found: {image_path}")
    mime_type, _ = mimetypes.guess_type(file_path.name)
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def invoke_json_prompt(
    model: Any,
    prompt: str,
    schema: type[Any],
    *,
    image_paths: list[str] | None = None,
) -> Any:
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    full_prompt = f"""
{prompt}

Return only a valid JSON object.
- Do not use markdown code fences.
- Do not add any explanation before or after the JSON.
- The JSON must follow this schema exactly:
{schema_json}
""".strip()
    image_paths = image_paths or []
    if image_paths:
        human_content: str | list[str | dict[Any, Any]] = [
            {
                "type": "text",
                "text": (
                    "The uploaded images are attached in this message. "
                    "Use them as visual grounding when planning.\n\n"
                    f"{full_prompt}"
                ),
            },
            *[
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(image_path)},
                }
                for image_path in image_paths
            ],
        ]
        human_content = cast(list[str | dict[Any, Any]], human_content)
    else:
        human_content = full_prompt

    messages = [
        SystemMessage(content="You must continue the assistant prefill and return only a valid JSON object."),
        HumanMessage(content=human_content),
        AIMessage(content="{"),
    ]
    logger.debug(
        "LLM request `invoke_json_prompt` schema=%s messages=\n%s",
        schema.__name__,
        _serialize_for_log([_message_to_log_dict(message) for message in messages]),
    )
    try:
        response = model.invoke(messages)
        raw_content = response.content if isinstance(response.content, str) else json.dumps(response.content, ensure_ascii=False)
        normalized_content = raw_content.lstrip()
        content = normalized_content if normalized_content.startswith("{") else "{" + normalized_content
        logger.debug("LLM response `invoke_json_prompt` schema=%s\n%s", schema.__name__, content)
        parsed = schema.model_validate(extract_json_object(content))
        logger.debug(
            "LLM parsed `invoke_json_prompt` schema=%s\n%s",
            schema.__name__,
            _serialize_for_log(parsed.model_dump()),
        )
        return parsed
    except Exception:
        logger.exception("LLM `invoke_json_prompt` failed for schema=%s", schema.__name__)
        raise


def run_file_tool_authoring_loop(
    *,
    model: Any,
    system_prompt: str,
    user_prompt: str,
    base_dir: Path,
    readable_roots: list[Path],
    allowed_scripts: list[Path],
    max_tool_call_steps: int,
) -> str:
    logger.info("Starting file-tool authoring loop for `%s`", base_dir)
    tools = build_file_tools(
        base_dir=base_dir,
        writable_root=base_dir,
        readable_roots=readable_roots,
        allowed_scripts=allowed_scripts,
    )
    tool_map = {tool.name: tool for tool in tools}
    llm = model.bind_tools(tools)
    messages: list[Any] = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    for step_index in range(max_tool_call_steps):
        logger.debug(
            "LLM request `run_file_tool_authoring_loop` step=%s messages=\n%s",
            step_index + 1,
            _serialize_for_log([_message_to_log_dict(message) for message in messages]),
        )
        response = llm.invoke(messages)
        logger.debug(
            "LLM response `run_file_tool_authoring_loop` step=%s\n%s",
            step_index + 1,
            _serialize_for_log(_message_to_log_dict(response)),
        )
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", [])
        if not tool_calls:
            content_str = str(response.content).strip()
            try:
                html_doc = extract_html_document(content_str)
                logger.info("Authoring loop finished at step=%s with valid HTML", step_index + 1)
                return html_doc
            except ValueError as exc:
                logger.warning("Authoring loop returned invalid HTML at step=%s: %s", step_index + 1, exc)
                # 如果没超过最大轮次，把错误塞回给模型让它重试
                error_msg = (
                    "Your response did not contain a valid HTML document. "
                    "You MUST output the complete final HTML code (starting with <!doctype html> or <html>). "
                    "Do not just output explanations."
                )
                messages.append(HumanMessage(content=error_msg))
                continue

        for tool_call in tool_calls:
            logger.info(
                "Authoring loop tool call step=%s tool=%s",
                step_index + 1,
                tool_call["name"],
            )
            logger.debug(
                "Authoring loop tool args step=%s tool=%s\n%s",
                step_index + 1,
                tool_call["name"],
                _serialize_for_log(tool_call["args"]),
            )
            tool = tool_map[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            logger.debug(
                "Authoring loop tool result step=%s tool=%s\n%s",
                step_index + 1,
                tool_call["name"],
                _serialize_for_log(result),
            )
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))

    logger.error("Authoring loop exceeded maximum steps: %s", max_tool_call_steps)
    raise RuntimeError("Tool-calling authoring loop exceeded the maximum number of steps.")


def planner_node(state: AgentState, *, settings: Settings, registry: SkillRegistry, model: Any) -> AgentState:
    logger.info("Entering planner node for session `%s`", state.get("session_id", "unknown"))
    prompt = f"""
You are planning a web-based prompt-to-video agent.

User request:
{state["user_request"]}

Uploaded images:
{json.dumps(state.get("uploaded_images", []), ensure_ascii=False)}

Feedback history:
{json.dumps(state.get("feedback_history", []), ensure_ascii=False)}

Available skills:
{json.dumps(registry.list_brief(), ensure_ascii=False)}

Return a structured plan.
Rules:
- Ask clarification only if truly necessary.
- Always include hyperframes-media-pipeline and hyperframes in selected_skills.
- Use ONE asset per scene. Do NOT plan separate background and hero assets.
- Plan shorter scenes (2-4 seconds each) to make the video feel dynamic. A 15s video should have 4-6 scenes.
- Plan enough assets to cover all scenes. Each scene gets exactly one asset (either image or video).
- Focus on scene transitions and text animations to make the composition engaging. Provide hints for `transition_in` (e.g. "slide from left", "fade", "zoom in") and `text_animation` (e.g. "typewriter", "staggered fade", "glitch") in the scene plan.
- Optimize for balanced generation cost.
- Prefer image assets over video assets by default. Use images with panning/zooming effects in the HTML to simulate motion.
- CRITICAL: If the requested video duration is greater than 5 seconds, you MUST include AT LEAST ONE video asset in your plan. Do not make a 10s+ video entirely out of images.
- Use video assets when complex motion is important to communicate the idea (e.g. garment movement, walking, camera motion).
- Use local asset targets under assets/.
- If uploaded images exist, prioritize reusing them as direct final image assets wherever reasonable before planning AI image generation.
- Use `asset_source` to explicitly distinguish each asset:
  - `local`: directly use one uploaded image as the final image asset. Only valid for `type="image"`.
  - `generated`: generate a new asset without uploaded-image references.
  - `generated_with_reference`: generate a new asset using selected uploaded images as references.
- When `asset_source` is `local`, set `uploaded_image_index` to the zero-based uploaded image index to use.
- When `asset_source` is `generated_with_reference`, set `reference_image_indexes` to the specific zero-based uploaded image indexes to reference. Do NOT reference all uploaded images unless they are all truly needed.
- If the planned image slots exceed the number of uploaded images, only the remaining image slots should require generated images.
- Video assets may use uploaded images as references when helpful, but this is optional, and you should usually choose only the most relevant 1-2 images instead of all uploads.
- Scene text should be concise and render-friendly.
"""
    try:
        plan = invoke_json_prompt(model, prompt, PlanResult, image_paths=state.get("uploaded_images", []))
    except Exception:
        logger.exception("Planner node failed for session `%s`", state.get("session_id", "unknown"))
        raise
    selected_skills = list(dict.fromkeys(["hyperframes-media-pipeline", "hyperframes", *plan.selected_skills]))
    logger.info(
        "Planner node completed for session `%s` with %s assets and %s scenes",
        state.get("session_id", "unknown"),
        len(plan.assets),
        len(plan.scenes),
    )
    return {
        "stage": "planning",
        "plan": plan.model_dump(),
        "clarification_needed": plan.needs_clarification,
        "clarification_questions": plan.clarification_questions,
        "selected_skills": selected_skills,
        "html_revision_count": 0,
    }


def clarification_router(state: AgentState) -> str:
    return "clarify" if state.get("clarification_needed") else "generate_assets"


def clarify_node(state: AgentState) -> AgentState:
    logger.info("Clarification required for session `%s`", state.get("session_id", "unknown"))
    return {"stage": "clarification_needed"}


def generate_assets_node(state: AgentState, *, settings: Settings) -> AgentState:
    logger.info("Entering generate_assets node for session `%s`", state.get("session_id", "unknown"))
    session_dir = Path(state["session_dir"])
    project_dir = session_dir / "project"
    pipeline_path = session_dir / "pipeline.json"
    plan = PlanResult.model_validate(state["plan"])
    payload = build_pipeline_payload(settings, plan, state.get("uploaded_images", []), str(project_dir))
    logger.debug("Media pipeline payload for session `%s`\n%s", state.get("session_id", "unknown"), _serialize_for_log(payload))
    write_pipeline_file(pipeline_path, payload)

    env = os.environ.copy()
    proc = subprocess.Popen(
        [sys.executable, str(settings.media_pipeline_script), "--pipeline", str(pipeline_path), "--output-dir", str(project_dir)],
        cwd=settings.app_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    progress_lines: list[str] = []
    full_output: list[str] = []
    if proc.stdout is not None:
        for line in proc.stdout:
            line = line.rstrip("\n")
            full_output.append(line)
            logger.debug("Media pipeline output: %s", line)
            if "[PROGRESS]" in line:
                progress_text = line.split("[PROGRESS]", 1)[1].strip()
                if progress_text:
                    progress_lines.append(progress_text)
                    session = load_session(session_dir)
                    session = append_progress(session, progress_text)
                    save_session(session_dir, session)
    proc.wait()
    if proc.returncode != 0:
        error_msg = "\n".join(full_output)
        logger.error("Media pipeline failed for session `%s`:\n%s", state.get("session_id", "unknown"), error_msg)
        raise RuntimeError(f"Media pipeline failed with exit code {proc.returncode}:\n{error_msg}")
    logger.info(
        "Generate assets completed for session `%s` with %s progress updates",
        state.get("session_id", "unknown"),
        len(progress_lines),
    )

    return {
        "stage": "executing",
        "project_dir": str(project_dir),
        "pipeline_path": str(pipeline_path),
        "resolved_pipeline_path": str(project_dir / "pipeline.resolved.json"),
        "creative_brief_path": str(project_dir / "creative-brief.md"),
    }


def verify_assets_node(state: AgentState, *, model: Any) -> AgentState:
    logger.info("Entering verify_assets node for session `%s`", state.get("session_id", "unknown"))
    resolved_pipeline = Path(state["resolved_pipeline_path"]).read_text(encoding="utf-8")
    verification = invoke_json_prompt(
        model,
        f"""
You are the verifier in a planner/executor/verifier loop.
Check whether the asset stage produced enough information for HyperFrames HTML authoring.

Resolved pipeline:
{resolved_pipeline}

Return:
- continue if assets and scenes are sufficient
- replan_required if the plan itself should change
- blocked if generation clearly failed
""",
        VerificationResult,
    )
    logger.info(
        "Verify assets decision for session `%s`: %s",
        state.get("session_id", "unknown"),
        verification.decision,
    )
    return {"verification": verification.model_dump()}


def verification_router(state: AgentState) -> str:
    decision = state.get("verification", {}).get("decision", "continue")
    if decision == "continue":
        return "build_html"
    if decision == "replan_required":
        return "planner"
    return "fail"


def build_html_node(state: AgentState, *, settings: Settings, registry: SkillRegistry, model: Any) -> AgentState:
    logger.info("Entering build_html node for session `%s`", state.get("session_id", "unknown"))
    project_dir = Path(state["project_dir"])
    resolved_pipeline_path = Path(state["resolved_pipeline_path"])
    creative_brief_path = Path(state["creative_brief_path"])
    resolved_pipeline = resolved_pipeline_path.read_text(encoding="utf-8")
    creative_brief = creative_brief_path.read_text(encoding="utf-8")
    skill_context = registry.build_context(state.get("selected_skills", []))

    system_prompt = """
You are authoring a final HyperFrames HTML composition.

You may use tools:
- `list_dir` to discover project and support files
- `read_file` to inspect exact project, pipeline, brief, or skill-support files
- `write_file` to write or update files inside the current project directory
- `patch_file` for focused edits to existing files
- `run_script` to run whitelisted pipeline scripts and use their output

Rules:
- Final answer should be the final HTML only if you are not already writing it via write_file
- Prefer reading exact files instead of guessing content
- Only write files inside the project directory
- Final `index.html` must be deterministic and HyperFrames-compatible
- Treat the injected skill references as authoritative implementation guidance.
- Prefer patterns, structure, and constraints from the injected skill references over your own ad hoc HTML patterns.
- If a skill reference provides a concrete rule or pattern, follow it unless it conflicts with an explicit hard constraint in this prompt.
"""

    prompt = f"""
Project directory:
{project_dir}

Primary files:
- {resolved_pipeline_path}
- {creative_brief_path}

You should usually write the final result to `index.html`.

Use these skill references:
{skill_context}

Use the skill references fully, not superficially:
- Read and apply the relevant structural rules, runtime conventions, and media patterns from the injected skill context.
- Prefer the skill-provided HyperFrames patterns before inventing your own structure.
- If the skill context is relevant but incomplete, extend it minimally instead of replacing it with a custom approach.
- If the skill context contains an "Referenced Documents (Auto-loaded)" section, treat it as part of the skill and follow it.

User request:
{state["user_request"]}

Feedback history:
{json.dumps(state.get("feedback_history", []), ensure_ascii=False)}

Creative brief:
{creative_brief}

Resolved pipeline:
{resolved_pipeline}

Requirements:
- Output only final HTML
- Use HyperFrames-compatible root composition rules
- Use local assets only from the resolved pipeline
- Include text, transitions, and visual effects
- Register paused GSAP timeline on window.__timelines
- Keep the output deterministic
- Use relative paths starting with `./assets/` for all media resources (e.g. `./assets/video.mp4`)
- HARD CONSTRAINT: Every timed scene container MUST include the `clip` class (for example: `<div class="scene clip" data-start="..." data-duration="...">`).
- HARD CONSTRAINT: Image assets MUST use normal `<img>` elements with local `./assets/...` paths.
- HARD CONSTRAINT: Video assets MUST use normal `<video>` elements with local `./assets/...` paths, and every video element MUST include `muted playsinline`.
- HARD CONSTRAINT: Every video asset MUST be represented as a timed media node with its own `data-start`, `data-duration`, and `data-track-index` so the HyperFrames runtime can control playback by timeline time.
- HARD CONSTRAINT: Do NOT rely on page-load playback for videos. Videos must be positioned so they become visible at the right scene time while playback remains runtime-controlled.
- HARD CONSTRAINT: Do NOT nest a playing `<video>` as an untimed background inside a separately timed scene pattern that assumes the browser starts playback on load. Use the video itself as the timed clip, or place it inside a non-timed wrapper while timing is carried by the media node.
- HARD CONSTRAINT: Do NOT add a global background-music `<audio>` element, and do NOT use `<audio src="...mp4">`.
- HARD CONSTRAINT: Do NOT call `play()`, `pause()`, or force media sync by repeatedly setting `currentTime` inside GSAP `onUpdate`. The runtime owns media playback.
- HARD CONSTRAINT: Do NOT use non-deterministic code such as `Date`, `new Date()`, `Math.random()`, timers that change output across runs, or runtime-generated IDs.
- HARD CONSTRAINT: Prefer `loop` for short visual video backgrounds when needed. Do NOT add `autoplay` or custom `video.play()` bootstrap code in the generated HTML.
- HARD CONSTRAINT: If audio is intentionally needed, it must be a separate timed `<audio>` element. Otherwise keep videos muted and visual-only.
"""
    html = run_file_tool_authoring_loop(
        model=model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        base_dir=project_dir,
        readable_roots=[project_dir, resolved_pipeline_path.parent, settings.repo_root, *settings.skills_dirs],
        allowed_scripts=[settings.media_pipeline_script, settings.html_builder_script],
        max_tool_call_steps=settings.max_tool_call_steps,
    )
    index_path = project_dir / "index.html"
    if not index_path.exists() or html:
        index_path.write_text(html.strip() + "\n", encoding="utf-8")
    logger.info("HTML authored for session `%s` at `%s`", state.get("session_id", "unknown"), index_path)
    meta_path = project_dir / "meta.json"
    if not meta_path.exists():
        resolved = json.loads(resolved_pipeline)
        meta = {
            "id": slugify(resolved.get("project_name", "media-project")),
            "name": resolved.get("project_name", "media-project"),
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"stage": "executing"}


def validate_html_node(state: AgentState, *, settings: Settings) -> AgentState:
    logger.info("Entering validate_html node for session `%s`", state.get("session_id", "unknown"))
    project_dir = Path(state["project_dir"])
    index_path = project_dir / "index.html"
    if not index_path.exists():
        return {
            "lint_output": "error: index.html was not generated",
            "validate_output": "error: index.html was not generated",
        }
    try:
        extract_html_document(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.exception("Generated index.html is not a valid HTML document for session `%s`", state.get("session_id", "unknown"))
        return {
            "lint_output": f"error: invalid html output: {exc}",
            "validate_output": f"error: invalid html output: {exc}",
        }
    try:
        lint_result = run_hyperframes_command(settings, ["lint", str(project_dir)], project_dir, check=False)
        lint_output = (lint_result.stdout or "") + (lint_result.stderr or "")
    except Exception as exc:
        lint_output = f"lint error: {exc}"
        logger.exception("Lint command failed for session `%s`", state.get("session_id", "unknown"))
    try:
        validate_result = run_hyperframes_command(
            settings,
            ["validate", str(project_dir)],
            project_dir,
            check=False,
        )
        validate_output = (validate_result.stdout or "") + (validate_result.stderr or "")
    except Exception as exc:
        validate_output = f"validate error: {exc}"
        logger.exception("Validate command failed for session `%s`", state.get("session_id", "unknown"))
    logger.debug("Lint output for session `%s`\n%s", state.get("session_id", "unknown"), lint_output)
    logger.debug("Validate output for session `%s`\n%s", state.get("session_id", "unknown"), validate_output)
    return {
        "lint_output": lint_output,
        "validate_output": validate_output,
    }


def validate_router(state: AgentState) -> str:
    text = f"{state.get('lint_output', '')}\n{state.get('validate_output', '')}".lower()

    # Only treat real validation failures as blocking. Outputs like `0 error(s)`
    # or `0 errors` should not be considered failures.
    has_nonzero_errors = bool(
        re.search(r"\b[1-9]\d*\s+error\(s\)\b", text)
        or re.search(r"\b[1-9]\d*\s+errors\b", text)
        or re.search(r"\bx\s+\[error\]\b", text)
        or re.search(r"\bfailed\b", text)
        or ("✗" in text)
    )
    if has_nonzero_errors:
        if state.get("html_revision_count", 0) < 1:
            return "repair_html"
        return "fail"
    return "render"


def repair_html_node(state: AgentState, *, settings: Settings, registry: SkillRegistry, model: Any) -> AgentState:
    logger.info("Entering repair_html node for session `%s`", state.get("session_id", "unknown"))
    project_dir = Path(state["project_dir"])
    skill_context = registry.build_context(["hyperframes", "gsap"])
    system_prompt = """
You are repairing a HyperFrames project.

You may use:
- `list_dir` to inspect available files in the project
- `read_file` to inspect current HTML or project support files
- `write_file` to directly update files in the project directory
- `patch_file` for precise edits to the current HTML or helper files
- `run_script` to run whitelisted pipeline scripts and use their output

Prefer fixing `index.html` in place.
Return final HTML only if you are not already writing it with write_file.
- Treat the injected skill references as authoritative repair guidance.
- Prefer repairing toward the skill-provided HyperFrames patterns instead of inventing a new structure.
- Fix the HTML to satisfy these hard constraints:
- Every timed scene container MUST include the `clip` class.
- Keep image assets as normal `<img>` elements and video assets as normal timed `<video>` elements with `muted playsinline`, `data-start`, `data-duration`, and `data-track-index`.
- Repair any scene where a video depends on page-load playback instead of runtime-controlled timed playback.
- Prefer the video element itself to carry media timing; do not leave the video as an untimed background that starts independently of the scene schedule.
- Remove any global background-music `<audio>` element (especially `<audio src="...mp4">`).
- Remove any GSAP/media logic that calls `play()`, `pause()`, or repeatedly sets `currentTime`.
- Remove any non-deterministic code such as `Date`, `new Date()`, `Math.random()`, or runtime-generated IDs.
- Do NOT add `autoplay` or custom `video.play()` bootstrap code during repair unless the user explicitly asks for that behavior.
- Keep `<video>` muted unless the project intentionally introduces a separate timed audio track.
"""
    prompt = f"""
Project directory:
{project_dir}

Repair this HyperFrames HTML.

Skill references:
{skill_context}

Use the skill references fully during repair:
- Compare the current HTML against the injected skill patterns and repair toward those patterns first.
- Prefer minimal fixes that move the HTML closer to the skill-provided HyperFrames conventions.
- If the skill context contains an "Referenced Documents (Auto-loaded)" section, treat it as part of the skill and follow it.

Lint output:
{state.get("lint_output", "")}

Validate output:
{state.get("validate_output", "")}

Read the current project files before fixing if needed.
Treat `timed_element_missing_clip_class` as a real issue that must be fixed, not as an ignorable warning.
"""

    html = run_file_tool_authoring_loop(
        model=model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        base_dir=project_dir,
        readable_roots=[project_dir, settings.repo_root, *settings.skills_dirs],
        allowed_scripts=[settings.media_pipeline_script, settings.html_builder_script],
        max_tool_call_steps=settings.max_tool_call_steps,
    )

    index_path = project_dir / "index.html"
    if not index_path.exists() or html:
        index_path.write_text(html.strip() + "\n", encoding="utf-8")
    logger.info("HTML repaired for session `%s` at `%s`", state.get("session_id", "unknown"), index_path)
    return {"html_revision_count": state.get("html_revision_count", 0) + 1}


def render_node(state: AgentState, *, settings: Settings) -> AgentState:
    logger.info("Entering render node for session `%s`", state.get("session_id", "unknown"))
    project_dir = Path(state["project_dir"])
    output_path = project_dir / "output.mp4"
    render_video(settings, project_dir, output_path)
    logger.info("Render completed for session `%s`: %s", state.get("session_id", "unknown"), output_path)
    return {"stage": "done", "render_output_path": str(output_path)}


def fail_node(state: AgentState) -> AgentState:
    logger.error("Workflow failed for session `%s`", state.get("session_id", "unknown"))
    return {"stage": "failed"}


def build_graph(settings: Settings, registry: SkillRegistry, model: Any):
    graph = StateGraph(AgentState)
    graph.add_node("planner", lambda state: planner_node(state, settings=settings, registry=registry, model=model))
    graph.add_node("clarify", clarify_node)
    graph.add_node("generate_assets", lambda state: generate_assets_node(state, settings=settings))
    graph.add_node("verify_assets", lambda state: verify_assets_node(state, model=model))
    graph.add_node("build_html", lambda state: build_html_node(state, settings=settings, registry=registry, model=model))
    graph.add_node("validate_html", lambda state: validate_html_node(state, settings=settings))
    graph.add_node("repair_html", lambda state: repair_html_node(state, settings=settings, registry=registry, model=model))
    graph.add_node("render", lambda state: render_node(state, settings=settings))
    graph.add_node("fail", fail_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", clarification_router, {"clarify": "clarify", "generate_assets": "generate_assets"})
    graph.add_edge("generate_assets", "verify_assets")
    graph.add_conditional_edges("verify_assets", verification_router, {"build_html": "build_html", "planner": "planner", "fail": "fail"})
    graph.add_edge("build_html", "validate_html")
    graph.add_conditional_edges("validate_html", validate_router, {"repair_html": "repair_html", "render": "render", "fail": "fail"})
    graph.add_edge("repair_html", "validate_html")
    graph.add_edge("clarify", END)
    graph.add_edge("render", END)
    graph.add_edge("fail", END)
    return graph.compile()
