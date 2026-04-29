from __future__ import annotations

import base64
import hashlib
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
    session_stats: dict[str, Any]


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


def _is_retryable_llm_exception(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    exc_name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    retry_markers = (
        "bad gateway",
        "gateway timeout",
        "service unavailable",
        "temporarily unavailable",
        "connection reset",
        "remoteprotocolerror",
        "readtimeout",
        "connecttimeout",
        "timed out",
        "expecting value",
    )
    return "jsondecodeerror" in exc_name or any(marker in message for marker in retry_markers)


def _invoke_llm_with_retries(
    llm: Any,
    messages: list[Any],
    *,
    operation: str,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
) -> Any:
    for attempt in range(1, max_attempts + 1):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            retryable = _is_retryable_llm_exception(exc)
            if not retryable or attempt >= max_attempts:
                logger.exception(
                    "LLM invoke failed for %s on attempt %s/%s (retryable=%s)",
                    operation,
                    attempt,
                    max_attempts,
                    retryable,
                )
                raise
            delay_seconds = base_delay_seconds * attempt
            logger.warning(
                "LLM invoke failed for %s on attempt %s/%s with retryable error %s: %s; retrying in %.1fs",
                operation,
                attempt,
                max_attempts,
                exc.__class__.__name__,
                exc,
                delay_seconds,
            )
            time.sleep(delay_seconds)


def empty_session_stats() -> dict[str, Any]:
    return {
        "tokens": {"input": 0, "output": 0, "total": 0},
        "media": {"videos_generated": 0, "images_generated": 0},
    }


def normalize_session_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    normalized = empty_session_stats()
    if not isinstance(stats, dict):
        return normalized

    tokens = stats.get("tokens")
    if isinstance(tokens, dict):
        normalized["tokens"]["input"] = int(tokens.get("input", 0) or 0)
        normalized["tokens"]["output"] = int(tokens.get("output", 0) or 0)
        normalized["tokens"]["total"] = int(tokens.get("total", 0) or 0)

    media = stats.get("media")
    if isinstance(media, dict):
        normalized["media"]["videos_generated"] = int(media.get("videos_generated", 0) or 0)
        normalized["media"]["images_generated"] = int(media.get("images_generated", 0) or 0)

    return normalized


def extract_llm_token_usage(response: Any) -> dict[str, int]:
    usage: dict[str, Any] = {}
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        usage = usage_metadata

    if not usage:
        response_metadata = getattr(response, "response_metadata", None)
        if isinstance(response_metadata, dict):
            nested_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
            if isinstance(nested_usage, dict):
                usage = nested_usage

    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or (input_tokens + output_tokens))
    return {"input": input_tokens, "output": output_tokens, "total": total_tokens}


def record_llm_token_usage(stats: dict[str, Any], response: Any) -> None:
    normalized = normalize_session_stats(stats)
    usage = extract_llm_token_usage(response)
    normalized["tokens"]["input"] += usage["input"]
    normalized["tokens"]["output"] += usage["output"]
    normalized["tokens"]["total"] += usage["total"]
    stats.clear()
    stats.update(normalized)


def compute_media_stats(resolved_pipeline_path: Path) -> dict[str, int]:
    if not resolved_pipeline_path.exists():
        return {"videos_generated": 0, "images_generated": 0}

    resolved_pipeline = json.loads(resolved_pipeline_path.read_text(encoding="utf-8"))
    videos_generated = 0
    images_generated = 0
    for asset in resolved_pipeline.get("assets", []):
        if not asset.get("resolved"):
            continue
        if asset.get("type") == "video":
            videos_generated += 1
        elif asset.get("type") == "image" and asset.get("asset_source") != "local":
            images_generated += 1

    return {"videos_generated": videos_generated, "images_generated": images_generated}


def _repair_truncated_json(text: str) -> str:
    """
    Heal common LLM JSON truncation issues before parsing:
    - unescaped newlines inside string values
    - trailing content after the final closing brace
    - missing closing brace/array bracket at the end
    """
    # 1) Remove markdown fences if present
    candidate = text.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced_match:
        candidate = fenced_match.group(1).strip()

    # 2) Strip any trailing text after the last top-level '}'
    last_brace = candidate.rfind("}")
    if last_brace != -1:
        candidate = candidate[: last_brace + 1]

    # 3) Replace raw newlines that appear inside quoted string values.
    #    We walk the string char-by-char so we only touch newlines that are
    #    genuinely inside a JSON string (between unescaped double quotes).
    result_chars: list[str] = []
    in_string = False
    escape_next = False
    for ch in candidate:
        if escape_next:
            result_chars.append(ch)
            escape_next = False
            continue
        if ch == "\\":
            result_chars.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result_chars.append(ch)
            continue
        if ch in "\n\r" and in_string:
            # Replace raw newline inside a JSON string with an escaped space
            # so the string stays valid and readable.
            result_chars.append(" ")
            continue
        result_chars.append(ch)
    return "".join(result_chars)


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = _repair_truncated_json(text)

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


def auto_fix_html_violations(html: str) -> str:
    """Deterministically fix known mechanical HyperFrames violations that don't require LLM judgment."""
    # 1. Remove data-has-audio="true" from any <video> element (all videos are muted).
    html = re.sub(
        r'(<video\b[^>]*?)\s+data-has-audio=(?:"true"|\'true\')([^>]*>)',
        r'\1\2',
        html,
        flags=re.IGNORECASE,
    )

    # 2. Strip timing attributes from <video> elements nested inside a timed <section>.
    #    HyperFrames forbids a video with data-start inside another timed element
    #    (video_nested_in_timed_element → video is FROZEN in renders).
    #    The parent <section> already carries clip timing; the video is a plain visual fill.
    def _strip_timing_from_nested_video(section_match: re.Match) -> str:
        open_tag = section_match.group(1)
        body = section_match.group(2)

        def _strip_video(video_match: re.Match) -> str:
            tag = video_match.group(0)
            # Only strip timing if this video actually has data-start (i.e., was incorrectly timed)
            if not re.search(r'\bdata-start=', tag, re.IGNORECASE):
                return tag
            tag = re.sub(r'\s+data-start=(?:"[^"]*"|\'[^\']*\')', '', tag, flags=re.IGNORECASE)
            tag = re.sub(r'\s+data-duration=(?:"[^"]*"|\'[^\']*\')', '', tag, flags=re.IGNORECASE)
            tag = re.sub(r'\s+data-track-index=(?:"[^"]*"|\'[^\']*\')', '', tag, flags=re.IGNORECASE)
            # Remove 'clip' from class but preserve other classes
            tag = re.sub(r'\bclip\s*', '', tag)
            tag = re.sub(r'class="\s*"', '', tag)
            return tag

        fixed_body = re.sub(
            r'<video\b[^>]*>',
            _strip_video,
            body,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return f"{open_tag}{fixed_body}</section>"

    html = re.sub(
        r'(<section\b[^>]*\bdata-start=(?:"[^"]*"|\'[^\']*\')[^>]*>)([\s\S]*?)</section>',
        _strip_timing_from_nested_video,
        html,
        flags=re.IGNORECASE,
    )

    return html


def detect_text_visibility_risks(html: str) -> list[str]:
    # Match opacity strictly equal to 0 (not 0.x).  The negative lookahead (?![.\d])
    # ensures we don't match opacity: 0.3, 0.98, etc.
    _OP_ZERO = r"opacity\s*:\s*0(?![.\d])"

    # Extract only content inside <style>...</style> blocks for CSS checks,
    # to avoid false-positives from matching GSAP JS object literals.
    style_blocks = re.findall(r"<style\b[^>]*>(.*?)</style>", html, re.IGNORECASE | re.DOTALL)
    css_only = "\n".join(style_blocks)

    risks: list[str] = []
    css_hidden_text_patterns = [
        rf"\.(?:kicker|title|body|point|copy|headline|subtitle|caption|cta)\b[^{{}}]*\{{[^{{}}]*{_OP_ZERO}",
        rf"(?:^|[,{{])\s*(?:h1|h2|h3|h4|p|li|span)\b[^{{}}]*\{{[^{{}}]*{_OP_ZERO}",
    ]
    has_hidden_text_css = bool(css_only) and any(
        re.search(pattern, css_only, re.IGNORECASE | re.DOTALL | re.MULTILINE) for pattern in css_hidden_text_patterns
    )
    has_from_opacity_zero = bool(re.search(rf"\b(?:gsap|tl)\.from(?:To)?\s*\([\s\S]*?{_OP_ZERO}", html, re.IGNORECASE))
    has_text_from_opacity_zero = bool(
        re.search(
            rf"\b(?:gsap|tl)\.from\s*\(\s*(?:\[[^\]]*(?:kicker|title|body|point|copy|headline|subtitle|caption|cta|vo|note|pill)[^\]]*\]|['\"][^'\"]*(?:kicker|title|body|point|copy|headline|subtitle|caption|cta|vo|note|pill)[^'\"]*['\"])\s*,[\s\S]*?{_OP_ZERO}",
            html,
            re.IGNORECASE,
        )
    )
    has_from_to_opacity_one = bool(
        re.search(
            rf"\b(?:gsap|tl)\.fromTo\s*\([\s\S]*?\{{[\s\S]*?{_OP_ZERO}[\s\S]*?\}}\s*,\s*\{{[\s\S]*?opacity\s*:\s*1\b",
            html,
            re.IGNORECASE,
        )
    )

    if has_hidden_text_css and re.search(rf"\b(?:gsap|tl)\.from\s*\([\s\S]*?{_OP_ZERO}", html, re.IGNORECASE):
        risks.append(
            "text_visibility_risk: text-like CSS selectors default to opacity:0 while GSAP uses from(... opacity:0 ...); "
            "this usually animates from invisible to invisible, so rendered text never appears"
        )

    if has_text_from_opacity_zero:
        risks.append(
            "text_visibility_risk: readable text uses gsap/timeline from(... opacity:0 ...). Default text should remain visible; "
            "use visible base styles plus motion, or fromTo(... opacity:0 -> 1 ...)"
        )

    if has_hidden_text_css and not has_from_to_opacity_one:
        risks.append(
            "text_visibility_risk: text-like CSS selectors default to opacity:0 without a clear matching fromTo/to animation back to opacity:1"
        )

    if has_hidden_text_css and not has_from_opacity_zero and not re.search(r"\b(?:gsap|tl)\.(?:to|set)\s*\([\s\S]*?opacity\s*:\s*1\b", html, re.IGNORECASE):
        risks.append(
            "text_visibility_risk: text-like CSS selectors default to opacity:0 but no deterministic GSAP reveal to opacity:1 was found"
        )

    return risks


def detect_missing_local_asset_refs(html: str, project_dir: Path) -> list[str]:
    missing: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"""<(?:img|video|audio)\b[^>]*\bsrc=["']([^"']+)["']""", html, re.IGNORECASE):
        src = match.group(1).strip()
        if not src:
            continue
        if src.startswith(("http://", "https://", "data:", "//")):
            continue
        normalized = src[2:] if src.startswith("./") else src
        if normalized.startswith("/"):
            normalized = normalized[1:]
        if not normalized:
            continue
        target = project_dir / normalized.replace("/", os.sep)
        if not target.exists() and src not in seen:
            missing.append(f'asset_src_not_found: "{src}" does not exist under the project directory')
            seen.add(src)
    return missing


def describe_html_snapshot(index_path: Path, html_text: str) -> dict[str, Any]:
    stat = index_path.stat()
    video_match = re.search(r"""<video\b[^>]*\bid=["'](?:scene3-media|s3-media)["'][^>]*>""", html_text, re.IGNORECASE)
    if not video_match:
        video_match = re.search(r"""<video\b[^>]*>""", html_text, re.IGNORECASE)
    body_opacity_zero = bool(re.search(r"""\.body\b[^{}]*\{[^{}]*opacity\s*:\s*0\b""", html_text, re.IGNORECASE | re.DOTALL))
    title_opacity_zero = bool(re.search(r"""\.title\b[^{}]*\{[^{}]*opacity\s*:\s*0\b""", html_text, re.IGNORECASE | re.DOTALL))
    text_from_opacity_zero = bool(
        re.search(
            r"\b(?:gsap|tl)\.from\s*\(\s*(?:\[[^\]]*(?:kicker|title|body|point|copy|headline|subtitle|caption|cta|vo|note|pill)[^\]]*\]|['\"][^'\"]*(?:kicker|title|body|point|copy|headline|subtitle|caption|cta|vo|note|pill)[^'\"]*['\"])\s*,[\s\S]*?opacity\s*:\s*0\b",
            html_text,
            re.IGNORECASE,
        )
    )
    return {
        "path": str(index_path),
        "mtime_epoch": round(stat.st_mtime, 3),
        "size_bytes": stat.st_size,
        "sha256": hashlib.sha256(html_text.encode("utf-8")).hexdigest(),
        "scene3_media_tag": video_match.group(0) if video_match else "",
        "has_data_has_audio_true": 'data-has-audio="true"' in html_text or "data-has-audio='true'" in html_text,
        "has_text_css_opacity_zero": body_opacity_zero or title_opacity_zero,
        "has_text_from_opacity_zero": text_from_opacity_zero,
    }


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
    session_stats: dict[str, Any] | None = None,
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
        response = _invoke_llm_with_retries(
            model,
            messages,
            operation=f"invoke_json_prompt:{schema.__name__}",
        )
        if session_stats is not None:
            record_llm_token_usage(session_stats, response)
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
    session_stats: dict[str, Any] | None = None,
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
        response = _invoke_llm_with_retries(
            llm,
            messages,
            operation=f"run_file_tool_authoring_loop:step_{step_index + 1}",
        )
        if session_stats is not None:
            record_llm_token_usage(session_stats, response)
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


def run_direct_html_authoring_loop(
    *,
    model: Any,
    system_prompt: str,
    user_prompt: str,
    current_html: str,
    max_attempts: int = 3,
    session_stats: dict[str, Any] | None = None,
) -> str:
    logger.info("Starting direct HTML authoring fallback without tools")
    fallback_system_prompt = (
        system_prompt
        + "\n\nTool fallback mode: file tools are unavailable for this request. "
        + "Work only from the provided HTML/pipeline/brief context and return the complete final HTML document."
    )
    fallback_user_prompt = (
        user_prompt
        + "\n\nCurrent `index.html` to enhance:\n```html\n"
        + current_html
        + "\n```"
        + "\n\nReturn only the complete final HTML document."
    )
    messages: list[Any] = [SystemMessage(content=fallback_system_prompt), HumanMessage(content=fallback_user_prompt)]

    for attempt in range(1, max_attempts + 1):
        logger.debug(
            "LLM request `run_direct_html_authoring_loop` attempt=%s messages=\n%s",
            attempt,
            _serialize_for_log([_message_to_log_dict(message) for message in messages]),
        )
        response = _invoke_llm_with_retries(
            model,
            messages,
            operation=f"run_direct_html_authoring_loop:attempt_{attempt}",
        )
        if session_stats is not None:
            record_llm_token_usage(session_stats, response)
        logger.debug(
            "LLM response `run_direct_html_authoring_loop` attempt=%s\n%s",
            attempt,
            _serialize_for_log(_message_to_log_dict(response)),
        )
        content_str = str(response.content).strip()
        try:
            html_doc = extract_html_document(content_str)
            logger.info("Direct HTML authoring fallback finished on attempt=%s", attempt)
            return html_doc
        except ValueError as exc:
            logger.warning("Direct HTML authoring fallback returned invalid HTML on attempt=%s: %s", attempt, exc)
            messages.append(response)
            messages.append(
                HumanMessage(
                    content=(
                        "Your response did not contain a complete valid HTML document. "
                        "Return only the full final HTML code starting with <!doctype html> or <html>."
                    )
                )
            )

    raise RuntimeError("Direct HTML authoring fallback exceeded the maximum number of attempts.")


def planner_node(state: AgentState, *, settings: Settings, registry: SkillRegistry, model: Any) -> AgentState:
    logger.info("Entering planner node for session `%s`", state.get("session_id", "unknown"))
    session_stats = normalize_session_stats(state.get("session_stats"))
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
- Only plan narration audio for image scenes that would otherwise be silent. Video scenes should rely on extracted native video audio by default and should NOT get extra narration audio unless the user explicitly asks for dubbing.
- Every scene should have audible content. Image scenes should usually use TTS narration; video scenes should usually rely on an extracted local audio track from the generated video rather than extra TTS dubbing.
- Plan shorter scenes (2-4 seconds each) to make the video feel dynamic. A 15s video should have 4-6 scenes.
- Plan enough assets to cover all scenes. Each scene gets exactly one asset (either image or video).
- If a scene uses an image asset, you should usually include a matching audio asset with concise spoken text.
- If a scene uses a video asset, do NOT add a narration audio asset for that scene unless the user explicitly asks for dubbing or voiceover on top of the video.
- Focus on scene transitions and text animations to make the composition engaging. Provide concrete hints for `transition_in` (e.g. "diagonal wipe with parallax", "masked push reveal", "radial flash zoom", "split-panel slide", "liquid blur dissolve") and `text_animation` (e.g. "typewriter", "staggered fade", "glitch", "kinetic word swap", "price pulse pop").
- Optimize for balanced generation cost.
- Prefer image assets over video assets by default. Use images with panning/zooming effects in the HTML to simulate motion.
- CRITICAL: If the requested video duration is greater than 5 seconds, you MUST include AT LEAST ONE video asset in your plan. Do not make a 10s+ video entirely out of images.
- CRITICAL: When planning a video asset, you MUST specify its `duration` as an integer. Do not leave it null, and do not use floats. The MINIMUM allowed duration for a video asset is 4. Do not generate video assets shorter than 4 seconds.
- Use video assets when complex motion is important to communicate the idea (e.g. garment movement, walking, camera motion).
- Use local asset targets under assets/.
- If uploaded images exist, prioritize reusing them as direct final image assets wherever reasonable before planning AI image generation.
- Use `asset_source` to explicitly distinguish each asset:
  - `local`: directly use one uploaded image as the final image asset. Only valid for `type="image"`.
  - `generated`: generate a new asset without uploaded-image references.
  - `generated_with_reference`: generate a new asset using selected uploaded images as references.
- CRITICAL: If an image asset is meant to reuse an uploaded image as the final visual, you MUST use `asset_source="local"` and set `uploaded_image_index`. Do NOT mark it as `generated`.
- CRITICAL: If `asset_source` is `generated` or `generated_with_reference` for an image asset, `prompt` MUST be non-empty.
- When `asset_source` is `local`, set `uploaded_image_index` to the zero-based uploaded image index to use.
- When `asset_source` is `generated_with_reference`, set `reference_image_indexes` to the specific zero-based uploaded image indexes to reference. Do NOT reference all uploaded images unless they are all truly needed.
- If the planned image slots exceed the number of uploaded images, only the remaining image slots should require generated images.
- Video assets may use uploaded images as references when helpful, but this is optional, and you should usually choose only the most relevant 1-2 images instead of all uploads.
- Scene text should be concise and render-friendly.
- Every scene should have clear on-screen messaging. Do NOT leave most scenes text-empty.
- Unless the user explicitly requests a pure visual beat, each scene should include at least a strong `title`, and usually also a `kicker` or `body`.
- CTA / price / selling-point scenes must have explicit readable copy in the plan, not just implied visuals.
- Keep copy compact enough to render large and legible on mobile-first vertical video.
- Plan the scene durations so the requested runtime is covered continuously with no empty tail at the end.
- Make sure the final scene still has meaningful visible media and text, not just a brief flash before the composition ends.
- Transitions should feel premium and noticeable, not generic. Avoid planning a whole video with only simple fades or basic slide-ins.
- Across the full plan, vary the transition language. Mix 2-4 stronger transition styles instead of repeating the same move every scene.
- For every planned audio asset:
  - use `type="audio"`
  - also include `prompt`; it can be a short style hint such as voice/tone guidance, and it MUST be present even if brief
  - set `text` to the exact narration script to synthesize
  - set `target` under `assets/` with an audio extension such as `.mp3`
  - set `target_duration` close to the linked scene duration
- For every scene:
  - set `audio_asset_id` only for image scenes that need narration
  - set `voiceover_text` to the concise spoken text that should be heard in that scene
  - keep `voiceover_text` semantically aligned with the scene's visible copy
- If a scene uses a video asset and has no `audio_asset_id`, that means the final HTML should use a separate timed audio track sourced from the generated video's extracted local audio file, not embedded `<video>` audio and not extra TTS.
- Audio assets are generated by TTS, not uploaded images, so do not use `asset_source="local"` for audio assets.
"""
    try:
        plan = invoke_json_prompt(
            model,
            prompt,
            PlanResult,
            image_paths=state.get("uploaded_images", []),
            session_stats=session_stats,
        )
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
        "session_stats": session_stats,
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
    session_stats = normalize_session_stats(state.get("session_stats"))
    session_stats["media"] = compute_media_stats(project_dir / "pipeline.resolved.json")

    return {
        "stage": "executing",
        "project_dir": str(project_dir),
        "pipeline_path": str(pipeline_path),
        "resolved_pipeline_path": str(project_dir / "pipeline.resolved.json"),
        "creative_brief_path": str(project_dir / "creative-brief.md"),
        "session_stats": session_stats,
    }


def verify_assets_node(state: AgentState, *, model: Any) -> AgentState:
    logger.info("Entering verify_assets node for session `%s`", state.get("session_id", "unknown"))
    session_stats = normalize_session_stats(state.get("session_stats"))
    resolved_pipeline = Path(state["resolved_pipeline_path"]).read_text(encoding="utf-8")
    verification = invoke_json_prompt(
        model,
        f"""
You are the verifier in a planner/executor/verifier loop.
Inspect the resolved pipeline and decide whether asset generation succeeded well enough to proceed to HTML authoring.

Default to `continue`. Only return `replan_required` or `blocked` when there is a concrete, observable problem in the resolved pipeline data itself — not a prediction about future HTML quality.

Return `replan_required` only if:
- One or more assets have `resolved: false` (generation failed) AND the missing asset is required for a scene
- Scene timing has obvious gaps (a scene ends before the next starts with no coverage) or the total covered duration is significantly shorter than the planned composition duration
- A required audio asset has no resolved path and leaves an image scene completely silent with no fallback

Return `blocked` only if:
- Asset generation clearly crashed or produced no usable output at all

Return `continue` in all other cases, including:
- Scene copy that could be stronger (HTML authoring will handle this)
- Aesthetic or structural concerns about future HTML layout (HTML authoring and validation handle this)
- Video scenes without explicit narration (they will use extracted audio in HTML)
- Minor timing imprecision that is within 1-2 seconds

Resolved pipeline:
{resolved_pipeline}
""",
        VerificationResult,
        session_stats=session_stats,
    )
    logger.info(
        "Verify assets decision for session `%s`: %s",
        state.get("session_id", "unknown"),
        verification.decision,
    )
    return {"verification": verification.model_dump(), "session_stats": session_stats}


def verification_router(state: AgentState) -> str:
    decision = state.get("verification", {}).get("decision", "continue")
    if decision == "continue":
        return "build_html"
    if decision == "replan_required":
        return "planner"
    return "fail"


def _generate_html_skeleton(settings: Settings, resolved_pipeline_path: Path, project_dir: Path) -> None:
    """Run build_hyperframes_html.py to generate a structurally correct index.html skeleton."""
    import subprocess as _subprocess
    import sys as _sys
    proc = _subprocess.run(
        [
            _sys.executable,
            str(settings.html_builder_script),
            "--pipeline",
            str(resolved_pipeline_path),
            "--output-dir",
            str(project_dir),
        ],
        cwd=settings.app_root,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        logger.warning("HTML skeleton script failed (will let LLM author from scratch): %s", proc.stderr or proc.stdout)


def build_html_node(state: AgentState, *, settings: Settings, registry: SkillRegistry, model: Any) -> AgentState:
    logger.info("Entering build_html node for session `%s`", state.get("session_id", "unknown"))
    project_dir = Path(state["project_dir"])
    resolved_pipeline_path = Path(state["resolved_pipeline_path"])
    creative_brief_path = Path(state["creative_brief_path"])
    resolved_pipeline = resolved_pipeline_path.read_text(encoding="utf-8")
    creative_brief = creative_brief_path.read_text(encoding="utf-8")

    # Step A: generate structurally correct skeleton via script (no LLM tokens)
    _generate_html_skeleton(settings, resolved_pipeline_path, project_dir)
    skeleton_exists = (project_dir / "index.html").exists()
    logger.info("HTML skeleton generated=%s for session `%s`", skeleton_exists, state.get("session_id", "unknown"))

    # Step B: LLM enhances style/animation only — use slim context to save tokens
    slim_context, ref_paths = registry.build_slim_context(state.get("selected_skills", []))

    skeleton_note = (
        "A structurally correct `index.html` skeleton has already been generated for you by the build script. "
        "Read it first, then enhance the visual design: improve CSS styling, add premium transitions, enrich GSAP animations, and ensure text is bold and readable. "
        "Do NOT restructure the HyperFrames scene/clip skeleton — preserve the existing `clip` class, `data-start`, `data-duration`, `data-track-index` attributes, and media element placement."
        if skeleton_exists else
        "No skeleton was pre-generated. Author the complete index.html from scratch following HyperFrames conventions."
    )

    ref_paths_note = ""
    if ref_paths:
        ref_paths_note = "Available skill reference documents (use read_file to load any you need):\n" + "\n".join(ref_paths)

    system_prompt = f"""
You are enhancing a HyperFrames HTML composition.

{skeleton_note}

You may use tools:
- `list_dir` to discover project and support files
- `read_file` to inspect exact project, pipeline, brief, or skill-support files
- `write_file` to write or update files inside the current project directory
- `patch_file` for focused edits to existing files
- `run_script` to run whitelisted pipeline scripts and use their output

Core rules (strictly enforced — do not override these during enhancement):
- Preserve every `clip` class, `data-start`, `data-duration`, `data-track-index` attribute already in the skeleton — do NOT remove or change them.
- `<video>` elements inside a timed `<section>` are plain visual fills — they have NO `data-start`/`data-duration`/`data-track-index` and NO `clip` class. Do NOT add timing attributes to them. Adding data-start to a nested video causes `video_nested_in_timed_element` lint error and freezes the video in renders.
- Do NOT set `data-has-audio="true"` on any `<video>` element. All videos are `muted`; set `data-has-audio="false"` if the attribute must be present.
- Audible scenes use separate timed `<audio>` elements with their own `data-start`/`data-duration`/`data-track-index`.
- Do NOT call `play()`, `pause()`, or set `currentTime` in JS.
- Do NOT use `Date`, `Math.random()`, or runtime-generated IDs.
- Do NOT animate `visibility`, `display`, or `autoAlpha` on `clip` elements.
- Register the paused GSAP timeline on `window.__timelines`.
- CRITICAL — text animation rule: NEVER use `gsap.from(selector, {{opacity:0, ...}})` on readable text (kicker, title, h1, h2, p, li, .copy, .points, .body, .headline, .subtitle, .cta). `from` with opacity:0 leaves text invisible when the timeline is paused. ALWAYS use `gsap.fromTo(selector, {{opacity:0, ...}}, {{opacity:1, ...}})` if a fade-in is needed, or animate only positional properties (y, x, scale) without touching opacity.
- Readable text CSS must NOT have `opacity:0` as its resting state.
- Use only exact `./assets/...` paths from the resolved pipeline.
- Favor premium ad direction: bold typography, clean layering, intentional transitions.
"""

    prompt = f"""
Project directory:
{project_dir}

Primary files:
- {resolved_pipeline_path}
- {creative_brief_path}

{ref_paths_note}

Skill references (core patterns only — load reference docs above as needed):
{slim_context}

User request:
{state["user_request"]}

Feedback history:
{json.dumps(state.get("feedback_history", []), ensure_ascii=False)}

Creative brief:
{creative_brief}

Resolved pipeline:
{resolved_pipeline}

Your task:
1. Read the current `index.html` (if it exists from the skeleton generator).
2. Enhance its visual design: improve CSS layout, typography, color, overlay contrast, and GSAP animations.
3. Add premium transitions between scenes (wipes, masked reveals, parallax pushes, blur dissolves).
4. Ensure every scene has readable, high-contrast text that is visible by default.
5. Write the final enhanced HTML back to `index.html`.

Do NOT restructure the clip/scene skeleton. Patch or replace style and animation sections only.
"""
    session_stats = normalize_session_stats(state.get("session_stats"))
    index_path = project_dir / "index.html"
    try:
        html = run_file_tool_authoring_loop(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            base_dir=project_dir,
            readable_roots=[project_dir, resolved_pipeline_path.parent, settings.repo_root, *settings.skills_dirs],
            allowed_scripts=[settings.media_pipeline_script, settings.html_builder_script],
            max_tool_call_steps=settings.max_tool_call_steps,
            session_stats=session_stats,
        )
    except Exception:
        if not skeleton_exists or not index_path.exists():
            raise
        logger.exception(
            "HTML enhancement via tool-calling failed for session `%s`; retrying with direct HTML fallback",
            state.get("session_id", "unknown"),
        )
        html = run_direct_html_authoring_loop(
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            current_html=index_path.read_text(encoding="utf-8"),
            session_stats=session_stats,
        )

    if not index_path.exists() or html:
        fixed_html = auto_fix_html_violations(html.strip())
        index_path.write_text(fixed_html + "\n", encoding="utf-8")
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
    return {"stage": "executing", "session_stats": session_stats}


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
    html_text = index_path.read_text(encoding="utf-8")
    logger.debug(
        "Validate HTML snapshot for session `%s`\n%s",
        state.get("session_id", "unknown"),
        _serialize_for_log(describe_html_snapshot(index_path, html_text)),
    )
    text_visibility_risks = detect_text_visibility_risks(html_text)
    missing_asset_risks = detect_missing_local_asset_refs(html_text, project_dir)
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
    if text_visibility_risks:
        # Only CSS resting-state opacity:0 is a hard error (text guaranteed invisible).
        # GSAP from(opacity:0) on text is a warning — it animates correctly during playback
        # but text is invisible while timeline is paused (not fatal for video renders).
        hard_risks = [r for r in text_visibility_risks if "css" in r.lower() or "resting" in r.lower() or ("opacity:0" in r and "gsap" not in r.lower())]
        soft_risks = [r for r in text_visibility_risks if r not in hard_risks]
        if hard_risks:
            risk_output = "\n".join(f"error: {risk}" for risk in hard_risks)
            validate_output = ((validate_output.rstrip() + "\n") if validate_output.strip() else "") + risk_output
        if soft_risks:
            risk_output = "\n".join(f"warning: {risk}" for risk in soft_risks)
            validate_output = ((validate_output.rstrip() + "\n") if validate_output.strip() else "") + risk_output
    if missing_asset_risks:
        risk_output = "\n".join(f"error: {risk}" for risk in missing_asset_risks)
        validate_output = ((validate_output.rstrip() + "\n") if validate_output.strip() else "") + risk_output
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
    # Some validators emit machine-readable issues as `error: ...` lines even when the summary says `0 error(s)`.
    has_error_lines = bool(re.search(r"(?m)^\s*error:\s+", text))

    # HyperFrames StaticGuard contract violations are always blocking.
    has_staticguard_contract_violation = ("staticguard" in text) or ("invalid hyperframe contract" in text)

    if has_nonzero_errors or has_error_lines or has_staticguard_contract_violation:
        if state.get("html_revision_count", 0) < 1:
            return "repair_html"
        return "fail"
    return "render"


def repair_html_node(state: AgentState, *, settings: Settings, registry: SkillRegistry, model: Any) -> AgentState:
    logger.info("Entering repair_html node for session `%s`", state.get("session_id", "unknown"))
    project_dir = Path(state["project_dir"])
    slim_context, ref_paths = registry.build_slim_context(["hyperframes", "gsap"])
    ref_paths_note = ""
    if ref_paths:
        ref_paths_note = "Available skill reference documents (use read_file to load any you need):\n" + "\n".join(ref_paths)
    feedback_history = state.get("feedback_history", [])
    feedback_note = ""
    if feedback_history:
        feedback_note = f"\nUser feedback to address during repair:\n{json.dumps(feedback_history, ensure_ascii=False)}\n"
    system_prompt = """
You are repairing a HyperFrames project.

You may use:
- `list_dir` to inspect available files in the project
- `read_file` to inspect current HTML or project support files
- `write_file` to directly update files in the project directory
- `patch_file` for precise edits to the current HTML or helper files
- `run_script` to run whitelisted pipeline scripts and use their output

Prefer fixing `index.html` in place with patch_file.
Return final HTML only if you are not already writing it with write_file.
Hard constraints to enforce:
- Every timed scene container MUST include the `clip` class.
- Keep image assets as `<img>` and keep video assets as timed `<video class="clip" muted playsinline>` elements with local `./assets/...` paths.
- If a scene contains a visual `<video>`, ensure that the video has `data-start`/`data-duration`/`data-track-index` aligned with its parent scene timing.
- Keep narration as separate timed `<audio>` elements with local `./assets/...` paths and their own timing attributes.
- Remove `data-has-audio="true"` from any `<video>` element — all videos are muted.
- Remove any global background-music `<audio>` and any `<audio src="...mp4">`.
- Remove GSAP/media logic that calls `play()`, `pause()`, or repeatedly sets `currentTime`.
- Remove non-deterministic code: `Date`, `new Date()`, `Math.random()`, runtime-generated IDs.
- Do NOT animate `visibility`/`display`/`autoAlpha` on `clip` elements.
- Do NOT invent or rename local asset paths.
- CRITICAL — text animation: Replace every `gsap.from(textSelector, {{opacity:0, ...}})` on readable text (kicker, title, h1, h2, p, li, .copy, .points) with `gsap.fromTo(textSelector, {{opacity:0, ...}}, {{opacity:1, ...}})` so the text has a clear visible end state. Alternatively, remove opacity from the from() call and animate only positional/scale properties.
- Ensure readable text CSS does NOT have `opacity:0` as its resting state.
"""
    prompt = f"""
Project directory:
{project_dir}

Repair this HyperFrames HTML.

{ref_paths_note}

Skill references (core patterns):
{slim_context}
{feedback_note}
Lint output:
{state.get("lint_output", "")}

Validate output:
{state.get("validate_output", "")}

The lint/validate outputs above are the primary repair targets. Use them explicitly.
You MUST read the current `index.html` before fixing so your repair is grounded in the actual generated code.
When a validation message names a concrete risk such as hidden text, trace that risk back to the exact CSS/GSAP pattern in the file and repair that pattern directly.
Treat `timed_element_missing_clip_class` as a real issue that must be fixed, not as an ignorable warning.
"""

    session_stats = normalize_session_stats(state.get("session_stats"))
    html = run_file_tool_authoring_loop(
        model=model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        base_dir=project_dir,
        readable_roots=[project_dir, settings.repo_root, *settings.skills_dirs],
        allowed_scripts=[settings.media_pipeline_script, settings.html_builder_script],
        max_tool_call_steps=settings.max_tool_call_steps,
        session_stats=session_stats,
    )

    index_path = project_dir / "index.html"
    if not index_path.exists() or html:
        fixed_html = auto_fix_html_violations(html.strip())
        index_path.write_text(fixed_html + "\n", encoding="utf-8")
    logger.info("HTML repaired for session `%s` at `%s`", state.get("session_id", "unknown"), index_path)
    return {"html_revision_count": state.get("html_revision_count", 0) + 1, "session_stats": session_stats}


def render_node(state: AgentState, *, settings: Settings) -> AgentState:
    logger.info("Entering render node for session `%s`", state.get("session_id", "unknown"))
    project_dir = Path(state["project_dir"])
    output_path = project_dir / "output.mp4"
    try:
        render_video(settings, project_dir, output_path)
    except Exception:
        logger.warning(
            "Render attempt 1 failed for session `%s`, retrying once",
            state.get("session_id", "unknown"),
            exc_info=True,
        )
        if output_path.exists():
            output_path.unlink(missing_ok=True)
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
