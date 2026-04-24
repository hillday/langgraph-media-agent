import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = APP_ROOT.parents[0]
load_dotenv(APP_ROOT / ".env")


def resolve_project_path(raw_path: str, *, app_root: Path = APP_ROOT, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == app_root.name:
        return repo_root / path
    return app_root / path


@dataclass(frozen=True)
class Settings:
    app_root: Path = APP_ROOT
    repo_root: Path = REPO_ROOT
    runs_dir: Path = APP_ROOT / "runs"
    skills_dirs: tuple[Path, ...] = (APP_ROOT / "skills", APP_ROOT / ".trae" / "skills")
    model_api_base: str = os.environ.get("MODEL_API_BASE", "https://ark.cn-beijing.volces.com/api/v3")
    model_api_key_env: str = os.environ.get("MODEL_API_KEY_ENV", "ARK_API_KEY")
    model_name: str = os.environ.get("MODEL_NAME", "ep-your-chat-model")
    image_provider_endpoint: str = os.environ.get(
        "IMAGE_PROVIDER_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    )
    image_provider_model: str = os.environ.get("IMAGE_PROVIDER_MODEL", "doubao-seedream-5-0-260128")
    video_provider_endpoint: str = os.environ.get(
        "VIDEO_PROVIDER_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    )
    video_provider_status_endpoint_base: str = os.environ.get(
        "VIDEO_PROVIDER_STATUS_ENDPOINT_BASE",
        "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks",
    )
    video_provider_model: str = os.environ.get("VIDEO_PROVIDER_MODEL", "doubao-seedance-2-0-fast-260128")
    tts_provider_endpoint: str = os.environ.get(
        "TTS_PROVIDER_ENDPOINT", "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    )
    tts_provider_app_id: str = os.environ.get("TTS_PROVIDER_APP_ID", "")
    tts_provider_access_key: str = os.environ.get("TTS_PROVIDER_ACCESS_KEY", "")
    tts_provider_resource_id: str = os.environ.get("TTS_PROVIDER_RESOURCE_ID", "")
    tts_provider_voice: str = os.environ.get("TTS_PROVIDER_VOICE", "zh_female_cancan_mars_bigtts")
    tts_provider_audio_format: str = os.environ.get("TTS_PROVIDER_AUDIO_FORMAT", "mp3")
    tts_provider_sample_rate: int = int(os.environ.get("TTS_PROVIDER_SAMPLE_RATE", "24000"))
    app_host: str = os.environ.get("APP_HOST", "127.0.0.1")
    app_port: int = int(os.environ.get("APP_PORT", "8010"))
    hyperframes_bin: str = os.environ.get("HYPERFRAMES_BIN", "").strip()
    hyperframes_command_timeout_seconds: int = int(os.environ.get("HYPERFRAMES_COMMAND_TIMEOUT_SECONDS", "180"))
    hyperframes_render_timeout_seconds: int = int(os.environ.get("HYPERFRAMES_RENDER_TIMEOUT_SECONDS", "1800"))
    max_tool_call_steps: int = int(os.environ.get("MAX_TOOL_CALL_STEPS", "8"))
    scripts_dir: Path = APP_ROOT / "scripts"
    media_pipeline_script: Path = resolve_project_path(
        os.environ.get("MEDIA_PIPELINE_SCRIPT", "scripts/build_media_pipeline.py")
    )
    html_builder_script: Path = resolve_project_path(
        os.environ.get("HTML_BUILDER_SCRIPT", "scripts/build_hyperframes_html.py")
    )


def get_settings() -> Settings:
    settings = Settings()
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    return settings
