import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")

def get_api_key() -> str:
    # 尝试从指定的环境变量名获取，默认降级到 ARK_API_KEY
    env_name = os.environ.get("MODEL_API_KEY_ENV", "ARK_API_KEY")
    # os.environ.get() may return None; keep the type stable for type checkers and runtime.
    key = (os.environ.get(env_name) or os.environ.get("ARK_API_KEY") or "")
    return key.strip()

def load_pipeline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def ensure_api_key() -> str:
    key = get_api_key()
    if not key:
        raise RuntimeError(f"Missing API Key in environment (checked {os.environ.get('MODEL_API_KEY_ENV', 'ARK_API_KEY')} and ARK_API_KEY).")
    return key


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def raise_for_status_with_hint(response: requests.Response, context: str) -> None:
    if response.status_code in (401, 403):
        details = response.text[:800]
        raise RuntimeError(
            f"{context} failed with {response.status_code}. "
            "Check ARK_API_KEY (missing or invalid).\n"
            f"Response: {details}"
        )
    response.raise_for_status()


def request_with_retry(
    method: str,
    url: str,
    *,
    context: str,
    attempts: int = 4,
    backoff_seconds: float = 2.0,
    retryable_status_codes: tuple[int, ...] = (408, 425, 429, 500, 502, 503, 504),
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code in retryable_status_codes:
                details = response.text[:300].strip()
                raise RuntimeError(f"{context} returned retryable status {response.status_code}: {details}")
            raise_for_status_with_hint(response, context)
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
        except requests.HTTPError:
            raise
        except RuntimeError as exc:
            if "retryable status" not in str(exc):
                raise
            last_error = exc

        if attempt == attempts:
            break

        sleep_seconds = backoff_seconds * attempt
        print(f"[RETRY] {context} attempt {attempt}/{attempts} failed, retrying in {sleep_seconds:.1f}s")
        time.sleep(sleep_seconds)

    raise RuntimeError(f"{context} failed after {attempts} attempts") from last_error


def download_file(url: str, output_path: Path) -> None:
    ensure_parent(output_path)
    response = request_with_retry("GET", url, context="File download", stream=True, timeout=60, attempts=4, backoff_seconds=2.0)
    with output_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_obj.write(chunk)


def normalize_image_ref(value: str, base_dir: Path) -> str:
    file_path = resolve_local_path(value, base_dir)
    if file_path is not None:
        mime_type, _ = mimetypes.guess_type(file_path.name)
        mime_type = mime_type or "image/png"
        encoded = base64.b64encode(file_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    value = value.strip()
    if value.startswith(("http://", "https://", "asset://", "data:")):
        return value
    raise FileNotFoundError(f"Image not found: {value}")


def resolve_local_path(value: str, base_dir: Path) -> Path | None:
    value = value.strip()
    if value.startswith(("http://", "https://", "asset://", "data:")):
        return None

    file_path = Path(value)
    if not file_path.is_absolute():
        file_path = (base_dir / file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Image not found: {value}")
    return file_path


def generate_image(asset: dict[str, Any], provider: dict[str, Any], output_root: Path, base_dir: Path) -> Path:
    target = output_root / asset["target"]
    if target.exists():
        return target

    asset_source = asset.get("asset_source", "generated")
    source_image = asset.get("source_image")
    if asset_source == "local" and source_image:
        source_path = resolve_local_path(source_image, base_dir)
        if source_path is None:
            raise FileNotFoundError(f"Uploaded source image not found: {source_image}")
        ensure_parent(target)
        shutil.copy2(source_path, target)
        return target

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ensure_api_key()}",
    }
    payload = {
        "model": provider["model"],
        "prompt": asset["prompt"],
        "size": provider.get("size", "2K"),
        "output_format": provider.get("output_format", "png"),
        "watermark": provider.get("watermark", False),
    }
    reference_images = asset.get("reference_images", [])
    if asset_source == "generated_with_reference" and reference_images:
        payload["image"] = [normalize_image_ref(image_ref, base_dir) for image_ref in reference_images]
        payload["sequential_image_generation"] = "disabled"

    response = request_with_retry(
        "POST",
        provider["endpoint"],
        context=f"Image generation ({asset['id']})",
        headers=headers,
        json=payload,
        timeout=120,
        attempts=3,
        backoff_seconds=3.0,
    )
    body = response.json()
    data = body.get("data", [])
    if not data or not data[0].get("url"):
        raise RuntimeError(f"Image generation returned no downloadable URL for asset {asset['id']}")

    download_file(data[0]["url"], target)
    return target


def create_video_request(asset: dict[str, Any], provider: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": asset["prompt"]}]

    first_image = asset.get("first_image")
    use_references = asset.get("asset_source") == "generated_with_reference"
    if first_image and not use_references:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": normalize_image_ref(first_image, base_dir)},
            }
        )
    if use_references:
        for image_ref in asset.get("reference_images", []):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": normalize_image_ref(image_ref, base_dir)},
                    "role": "reference_image",
                }
            )

        if asset.get("reference_video"):
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": asset["reference_video"]},
                    "role": "reference_video",
                }
            )

        if asset.get("reference_audio"):
            content.append(
                {
                    "type": "audio_url",
                    "audio_url": {"url": asset["reference_audio"]},
                    "role": "reference_audio",
                }
            )

    return {
        "model": provider["model"],
        "content": content,
        "generate_audio": provider.get("generate_audio", False),
        "ratio": asset.get("ratio", provider.get("ratio", "16:9")),
        "duration": asset.get("duration", provider.get("duration", 5)),
        "watermark": provider.get("watermark", False),
    }


def poll_video_task(provider: dict[str, Any], task_id: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {ensure_api_key()}"}
    status_url = f"{provider['status_endpoint_base'].rstrip('/')}/{task_id}"

    for _ in range(180):
        response = request_with_retry(
            "GET",
            status_url,
            context=f"Video task polling ({task_id})",
            headers=headers,
            timeout=30,
            attempts=4,
            backoff_seconds=2.0,
        )
        body = response.json()
        status = body.get("status")
        if status == "succeeded":
            return body
        if status == "failed":
            raise RuntimeError(f"Video generation failed for task {task_id}: {json.dumps(body)}")
        time.sleep(5)

    raise TimeoutError(f"Video generation timed out for task {task_id}")


def extract_video_url(task_result: dict[str, Any]) -> str:
    content = task_result.get("content", {})
    if isinstance(content, dict) and content.get("video_url"):
        return str(content["video_url"])

    for key in ("outputs", "data"):
        value = task_result.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("video_url"):
                    return str(item["video_url"])

    raise RuntimeError(f"Could not find video URL in task result: {json.dumps(task_result)}")


def generate_video(asset: dict[str, Any], provider: dict[str, Any], output_root: Path, base_dir: Path) -> Path:
    target = output_root / asset["target"]
    if target.exists():
        return target

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ensure_api_key()}",
    }
    payload = create_video_request(asset, provider, base_dir)
    response = request_with_retry(
        "POST",
        provider["endpoint"],
        context=f"Video task creation ({asset['id']})",
        headers=headers,
        json=payload,
        timeout=30,
        attempts=3,
        backoff_seconds=3.0,
    )
    task_info = response.json()
    task_id = task_info.get("id")
    if not task_id:
        raise RuntimeError(f"Video task creation returned no task id: {json.dumps(task_info)}")

    result = poll_video_task(provider, task_id)
    video_url = extract_video_url(result)
    download_file(video_url, target)
    return target


def resolve_single_asset(asset: dict[str, Any], providers: dict[str, Any], output_root: Path, base_dir: Path, idx: int, total: int) -> tuple[str, str]:
    asset_type = asset["type"]
    desc = asset.get("description") or asset.get("id", "")
    print(f"[PROGRESS] Generating asset {idx}/{total}: {desc} ({asset_type})")
    if asset_type == "image":
        generated = generate_image(asset, providers["image"], output_root, base_dir)
    elif asset_type == "video":
        generated = generate_video(asset, providers["video"], output_root, base_dir)
    else:
        raise ValueError(f"Unsupported asset type: {asset_type}")
    print(f"[PROGRESS] Asset {idx}/{total} done: {desc}")
    return asset["id"], generated.relative_to(output_root).as_posix()


def resolve_assets(pipeline: dict[str, Any], output_root: Path, base_dir: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    providers = pipeline["providers"]
    total = len(pipeline["assets"])
    max_workers = min(len(pipeline["assets"]), 4) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(resolve_single_asset, asset, providers, output_root, base_dir, idx, total): asset["id"]
            for idx, asset in enumerate(pipeline["assets"], start=1)
        }
        for future in as_completed(futures):
            asset_id, path = future.result()
            resolved[asset_id] = path
    return resolved


def build_resolved_pipeline(pipeline: dict[str, Any], resolved_assets: dict[str, str]) -> dict[str, Any]:
    resolved_pipeline = json.loads(json.dumps(pipeline))
    for asset in resolved_pipeline.get("assets", []):
        asset_id = asset["id"]
        asset["resolved_path"] = resolved_assets.get(asset_id, "")
        asset["resolved"] = bool(asset.get("resolved_path"))
    return resolved_pipeline


def write_pipeline_outputs(output_root: Path, resolved_pipeline: dict[str, Any]) -> None:
    ensure_parent(output_root / "pipeline.resolved.json")
    (output_root / "pipeline.resolved.json").write_text(
        json.dumps(resolved_pipeline, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    brief_lines = [
        f"# {resolved_pipeline.get('project_name', 'media-project')}",
        "",
        "## Request",
        str(resolved_pipeline.get("request", "")),
        "",
        "## Assets",
    ]
    for asset in resolved_pipeline.get("assets", []):
        brief_lines.append(
            f"- {asset.get('id')}: type={asset.get('type')}, source={asset.get('asset_source', 'generated')}, path={asset.get('resolved_path', '(missing)')}"
        )
    brief_lines.append("")
    brief_lines.append("## Scenes")
    for scene in resolved_pipeline.get("scenes", []):
        brief_lines.append(
            f"- {scene.get('id')}: start={scene.get('start')}s, duration={scene.get('duration')}s, title={scene.get('title', '')}"
        )
    brief_lines.append("")
    (output_root / "creative-brief.md").write_text("\n".join(brief_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: generate local assets + pipeline.resolved.json.")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for the project (assets, resolved json)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_path = Path(args.pipeline).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(pipeline_path)
    base_dir = pipeline_path.parent
    resolved_assets = resolve_assets(pipeline, output_root, base_dir)
    resolved_pipeline = build_resolved_pipeline(pipeline, resolved_assets)
    write_pipeline_outputs(output_root, resolved_pipeline)

    print(f"Generated assets and resolved pipeline at: {output_root}")


if __name__ == "__main__":
    main()
