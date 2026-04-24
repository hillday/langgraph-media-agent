from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import Settings
from .models import PlanResult


def _select_uploaded_images(uploaded_images: list[str], indexes: list[int]) -> list[str]:
    selected: list[str] = []
    seen: set[int] = set()
    for index in indexes:
        if 0 <= index < len(uploaded_images) and index not in seen:
            selected.append(uploaded_images[index])
            seen.add(index)
    return selected


def _next_unused_uploaded_image(uploaded_images: list[str], used_indexes: set[int]) -> tuple[int | None, str | None]:
    for index, image_path in enumerate(uploaded_images):
        if index not in used_indexes:
            return index, image_path
    return None, None


def _target_with_source_extension(target: str, source_image: str) -> str:
    source_suffix = Path(source_image).suffix or ".png"
    return Path(target).with_suffix(source_suffix.lower()).as_posix()


def _normalize_audio_target(target: str, audio_format: str) -> str:
    suffix = f".{audio_format.strip('.').lower() or 'mp3'}"
    return Path(target).with_suffix(suffix).as_posix()


def build_pipeline_payload(
    settings: Settings, plan: PlanResult, uploaded_images: list[str], output_dir_name: str
) -> dict[str, Any]:
    assets: list[dict[str, Any]] = []
    used_local_indexes: set[int] = set()

    for asset in plan.assets:
        asset_payload = asset.model_dump()
        asset_source = asset.asset_source

        if asset.type == "audio":
            asset_payload["asset_source"] = "generated"
            asset_payload["reference_images"] = []
            asset_payload["reference_image_indexes"] = []
            asset_payload["target"] = _normalize_audio_target(asset.target, settings.tts_provider_audio_format)
            assets.append(asset_payload)
            continue

        if asset.type == "video" and asset_source == "local":
            asset_source = "generated_with_reference"

        if asset_source is None:
            if asset.type == "image" and len(used_local_indexes) < len(uploaded_images):
                asset_source = "local"
            elif asset.use_uploaded_images_as_references and uploaded_images:
                asset_source = "generated_with_reference"
            else:
                asset_source = "generated"

        if asset_source == "local" and asset.type == "image":
            local_index = asset.uploaded_image_index
            source_image: str | None = None
            if local_index is not None and 0 <= local_index < len(uploaded_images) and local_index not in used_local_indexes:
                source_image = uploaded_images[local_index]
            else:
                local_index, source_image = _next_unused_uploaded_image(uploaded_images, used_local_indexes)

            if source_image is not None and local_index is not None:
                used_local_indexes.add(local_index)
                asset_payload["asset_source"] = "local"
                asset_payload["uploaded_image_index"] = local_index
                asset_payload["source_image"] = source_image
                asset_payload["target"] = _target_with_source_extension(asset.target, source_image)
                asset_payload["reference_images"] = []
                asset_payload["reference_image_indexes"] = []
            else:
                asset_source = "generated"

        if asset_source == "generated_with_reference":
            default_limit = 1 if asset.type == "image" else 2
            reference_indexes = asset.reference_image_indexes
            if not reference_indexes and asset.uploaded_image_index is not None:
                reference_indexes = [asset.uploaded_image_index]
            reference_images = _select_uploaded_images(uploaded_images, reference_indexes)
            if not reference_images and asset.use_uploaded_images_as_references and uploaded_images:
                reference_images = uploaded_images[:default_limit]
                reference_indexes = list(range(len(reference_images)))

            if reference_images:
                asset_payload["asset_source"] = "generated_with_reference"
                asset_payload["reference_images"] = reference_images
                asset_payload["reference_image_indexes"] = reference_indexes[: len(reference_images)]
            else:
                asset_source = "generated"

        if asset_source == "generated":
            asset_payload["asset_source"] = "generated"
            asset_payload.pop("source_image", None)
            asset_payload["reference_images"] = []
            asset_payload["reference_image_indexes"] = []

        assets.append(asset_payload)

    return {
        "project_name": plan.project_name,
        "request": plan.summary,
        "output_dir": output_dir_name,
        "format": {
            "width": plan.width,
            "height": plan.height,
            "duration": plan.duration,
            "ratio": plan.ratio,
        },
        "providers": {
            "image": {
                "endpoint": settings.image_provider_endpoint,
                "model": settings.image_provider_model,
                "size": "2K",
                "output_format": "png",
                "watermark": False,
            },
            "video": {
                "endpoint": settings.video_provider_endpoint,
                "status_endpoint_base": settings.video_provider_status_endpoint_base,
                "model": settings.video_provider_model,
                "duration": min(plan.duration, 10),
                "ratio": plan.ratio,
                "watermark": False,
                "generate_audio": True,
            },
            "audio": {
                "endpoint": settings.tts_provider_endpoint,
                "app_id": settings.tts_provider_app_id,
                "access_key": settings.tts_provider_access_key,
                "resource_id": settings.tts_provider_resource_id,
                "voice": settings.tts_provider_voice,
                "audio_format": settings.tts_provider_audio_format,
                "sample_rate": settings.tts_provider_sample_rate,
            },
        },
        "assets": assets,
        "scenes": [scene.model_dump() for scene in plan.scenes],
    }


def write_pipeline_file(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_media_pipeline(settings: Settings, pipeline_path: Path, output_dir: Path, full_html: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        [
            sys.executable,
            str(settings.media_pipeline_script),
            "--pipeline",
            str(pipeline_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=settings.app_root,
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )


def run_html_builder(settings: Settings, resolved_pipeline_path: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        [
            sys.executable,
            str(settings.html_builder_script),
            "--pipeline",
            str(resolved_pipeline_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=settings.app_root,
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
