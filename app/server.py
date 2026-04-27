from __future__ import annotations

import asyncio
import shutil
import traceback
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import get_settings
from .graph import (
    AgentState,
    build_graph,
    build_html_node,
    clarify_node,
    planner_node,
    render_node,
    repair_html_node,
    validate_html_node,
    validate_router,
    verification_router,
    verify_assets_node,
    generate_assets_node,
)
from .llm import build_chat_model
from .models import SessionData
from .skill_registry import SkillRegistry
from .storage import create_session_dir, load_session, save_session


settings = get_settings()
registry = SkillRegistry(settings.skills_dirs)
model = build_chat_model(settings)
graph = build_graph(settings, registry, model)

app = FastAPI(title="LangGraph HyperFrames Media Agent")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/runs", StaticFiles(directory=str(settings.runs_dir)), name="runs")
MAX_UPLOAD_IMAGES = 9


def session_dir_from_id(session_id: str) -> Path:
    session_dir = settings.runs_dir / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return session_dir


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


def _append_progress(session: SessionData, message: str) -> SessionData:
    progress = [*session.progress, message]
    # Keep the file size bounded.
    if len(progress) > 200:
        progress = progress[-200:]
    return session.model_copy(update={"progress": progress, "status_message": message})


def _update_from_state(session: SessionData, state: AgentState) -> SessionData:
    update: dict[str, Any] = {
        "stage": state.get("stage", session.stage),
        "clarification_questions": state.get("clarification_questions", session.clarification_questions),
        "selected_skills": state.get("selected_skills", session.selected_skills),
        "plan_summary": state.get("plan", {}).get("summary", session.plan_summary),
        "project_dir": state.get("project_dir", session.project_dir),
        "pipeline_path": state.get("pipeline_path", session.pipeline_path),
        "resolved_pipeline_path": state.get("resolved_pipeline_path", session.resolved_pipeline_path),
        "creative_brief_path": state.get("creative_brief_path", session.creative_brief_path),
        "render_output_path": state.get("render_output_path", session.render_output_path),
        "pipeline": state.get("plan", session.pipeline),
        "last_error": state.get("last_error", session.last_error),
    }
    if state.get("session_stats"):
        update["stats"] = state["session_stats"]
    return session.model_copy(update=update)


def _run_workflow_sync(session_dir: Path, session: SessionData) -> None:
    """
    Runs the planner/executor/verifier/html/validate/render loop synchronously and
    persists progress updates to `session.json` after each stage.
    """
    try:
        state: AgentState = {
            "session_id": session.session_id,
            "session_dir": str(session_dir),
            "user_request": session.user_request,
            "uploaded_images": session.uploaded_images,
            "feedback_history": session.feedback_history,
            "stage": "input",
        }

        session = session.model_copy(update={"stage": "planning", "last_error": ""})
        session = _append_progress(session, "Planning...")
        save_session(session_dir, session)

        # planner -> (clarify | generate_assets)
        state.update(planner_node(state, settings=settings, registry=registry, model=model))
        session = _update_from_state(session, state)
        save_session(session_dir, session)
        if state.get("clarification_needed"):
            state.update(clarify_node(state))
            session = _update_from_state(session, state)
            session = _append_progress(session, "Need clarification.")
            save_session(session_dir, session)
            return

        # generate assets
        session = _append_progress(session, "Generating assets (this may take a few minutes)...")
        session = session.model_copy(update={"stage": "executing"})
        save_session(session_dir, session)
        state.update(generate_assets_node(state, settings=settings))
        session = _update_from_state(session, state)
        save_session(session_dir, session)

        # verify assets -> (build_html | planner | fail)
        session = _append_progress(session, "Verifying assets...")
        save_session(session_dir, session)
        state.update(verify_assets_node(state, model=model))
        session = _update_from_state(session, state)
        save_session(session_dir, session)
        route = verification_router(state)
        if route == "planner":
            session = _append_progress(session, "Replan required. Returning to planner...")
            save_session(session_dir, session)
            state.update(planner_node(state, settings=settings, registry=registry, model=model))
            session = _update_from_state(session, state)
            save_session(session_dir, session)
            if state.get("clarification_needed"):
                state.update(clarify_node(state))
                session = _update_from_state(session, state)
                session = _append_progress(session, "Need clarification.")
                save_session(session_dir, session)
                return
            state.update(generate_assets_node(state, settings=settings))
            session = _update_from_state(session, state)
            save_session(session_dir, session)
            state.update(verify_assets_node(state, model=model))
            session = _update_from_state(session, state)
            save_session(session_dir, session)
            route = verification_router(state)
        if route == "fail":
            session = session.model_copy(update={"stage": "failed"})
            session = _append_progress(session, "Failed during asset verification.")
            save_session(session_dir, session)
            return

        # build html
        session = _append_progress(session, "Authoring HyperFrames HTML...")
        save_session(session_dir, session)
        state.update(build_html_node(state, settings=settings, registry=registry, model=model))
        session = _update_from_state(session, state)
        save_session(session_dir, session)

        # validate -> maybe repair -> validate
        session = _append_progress(session, "Validating project...")
        save_session(session_dir, session)
        state.update(validate_html_node(state, settings=settings))
        session = _update_from_state(session, state)
        save_session(session_dir, session)
        route = validate_router(state)
        if route == "repair_html":
            session = _append_progress(session, "Repairing HTML...")
            save_session(session_dir, session)
            state.update(repair_html_node(state, settings=settings, registry=registry, model=model))
            session = _update_from_state(session, state)
            save_session(session_dir, session)
            state.update(validate_html_node(state, settings=settings))
            session = _update_from_state(session, state)
            save_session(session_dir, session)
            route = validate_router(state)
        if route == "fail":
            session = session.model_copy(update={"stage": "failed"})
            session = _append_progress(session, "Failed validation.")
            save_session(session_dir, session)
            return

        # render
        session = _append_progress(session, "Rendering final video...")
        save_session(session_dir, session)
        session = session.model_copy(update={"stage": "rendering"})
        save_session(session_dir, session)
        state.update(render_node(state, settings=settings))
        session = _update_from_state(session, state)
        session = session.model_copy(update={"stage": "done"})
        session = _append_progress(session, f"Render done: {session.render_output_path}")
        save_session(session_dir, session)
    except Exception:
        session = session.model_copy(update={"stage": "failed", "last_error": traceback.format_exc()})
        session = _append_progress(session, "Failed with an exception. See last_error.")
        save_session(session_dir, session)


async def _run_workflow_background(session_id: str) -> None:
    session_dir = session_dir_from_id(session_id)
    session = load_session(session_dir)
    await asyncio.to_thread(_run_workflow_sync, session_dir, session)


@app.post("/api/sessions")
async def create_session(
    request: str = Form(...),
    images: list[UploadFile] | None = File(default=None),
):
    if images and len(images) > MAX_UPLOAD_IMAGES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_UPLOAD_IMAGES} images are supported.")

    session_id, session_dir = create_session_dir(settings.runs_dir)
    upload_paths: list[str] = []
    uploads_dir = session_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    for image in images or []:
        filename = Path(image.filename or "").name
        if not filename:
            continue
        unique_name = f"{uuid4().hex[:8]}-{filename}"
        target = uploads_dir / unique_name
        with target.open("wb") as file_obj:
            shutil.copyfileobj(image.file, file_obj)
        upload_paths.append(str(target))

    session = SessionData(
        session_id=session_id,
        user_request=request,
        uploaded_images=upload_paths,
        stage="planning",
        status_message="Queued.",
        progress=["Queued."],
    )
    save_session(session_dir, session)

    asyncio.create_task(_run_workflow_background(session_id))
    return session.model_dump()


@app.get("/api/sessions")
def list_sessions():
    items: list[dict] = []
    if not settings.runs_dir.exists():
        return items
    for child in sorted(settings.runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir():
            continue
        session_path = child / "session.json"
        if not session_path.exists():
            continue
        try:
            session = load_session(child)
            items.append(
                {
                    "session_id": session.session_id,
                    "stage": session.stage,
                    "user_request": session.user_request,
                    "plan_summary": session.plan_summary,
                    "render_output_path": session.render_output_path,
                }
            )
        except Exception:
            continue
    return items


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = load_session(session_dir_from_id(session_id))
    return session.model_dump()


@app.post("/api/sessions/{session_id}/feedback")
async def submit_feedback(session_id: str, feedback: str = Form(...)):
    session_dir = session_dir_from_id(session_id)
    session = load_session(session_dir)
    feedback_history = [*session.feedback_history, feedback]
    updated = session.model_copy(update={"feedback_history": feedback_history, "stage": "planning"})
    updated = _append_progress(updated, f"User feedback: {feedback}")
    updated = _append_progress(updated, "Queued.")
    save_session(session_dir, updated)
    asyncio.create_task(_run_workflow_background(session_id))
    return updated.model_dump()

