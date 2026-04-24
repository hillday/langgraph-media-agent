from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Stage = Literal[
    "input",
    "clarification_needed",
    "planning",
    "executing",
    "rendering",
    "done",
    "failed",
]


class AssetPlan(BaseModel):
    id: str
    type: Literal["image", "video", "audio"]
    prompt: str = ""
    description: str = ""
    target: str
    duration: int | None = None
    target_duration: float | None = None
    text: str = ""
    ratio: str | None = None
    use_uploaded_images_as_references: bool = False
    asset_source: Literal["local", "generated", "generated_with_reference"] | None = None
    uploaded_image_index: int | None = None
    reference_image_indexes: list[int] = Field(default_factory=list)


class ScenePlan(BaseModel):
    id: str
    start: float
    duration: float
    kicker: str = ""
    title: str
    body: str = ""
    points: list[str] = Field(default_factory=list)
    asset_id: str
    audio_asset_id: str | None = None
    voiceover_text: str = ""
    transition_in: str = ""
    text_animation: str = ""


class PlanResult(BaseModel):
    needs_clarification: bool
    clarification_questions: list[str] = Field(default_factory=list)
    project_name: str
    width: int = 1920
    height: int = 1080
    duration: int = 10
    ratio: str = "16:9"
    selected_skills: list[str] = Field(default_factory=lambda: ["hyperframes-media-pipeline", "hyperframes"])
    summary: str
    assets: list[AssetPlan]
    scenes: list[ScenePlan]


class VerificationResult(BaseModel):
    decision: Literal["continue", "replan_required", "blocked"]
    summary: str
    issues: list[str] = Field(default_factory=list)


class SessionData(BaseModel):
    session_id: str
    stage: Stage = "input"
    user_request: str
    uploaded_images: list[str] = Field(default_factory=list)
    status_message: str = ""
    progress: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    selected_skills: list[str] = Field(default_factory=list)
    plan_summary: str = ""
    project_dir: str = ""
    pipeline_path: str = ""
    resolved_pipeline_path: str = ""
    creative_brief_path: str = ""
    render_output_path: str = ""
    last_error: str = ""
    feedback_history: list[str] = Field(default_factory=list)
    pipeline: dict[str, Any] = Field(default_factory=dict)
