import argparse
import base64
import json
import mimetypes
import os
import re
import shutil
import subprocess
import time
import wave
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
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    expected_status_codes: tuple[int, ...] = (200,),
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(method, url, **kwargs)
            if response.status_code not in expected_status_codes:
                details = response.text[:800].strip()
                raise RuntimeError(
                    f"{context} returned unexpected status {response.status_code}. "
                    f"Expected {expected_status_codes}. Response: {details}"
                )
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
        except RuntimeError as exc:
            last_error = exc

        if attempt == attempts:
            break

        sleep_seconds = backoff_seconds * attempt
        print(f"[RETRY] {context} attempt {attempt}/{attempts} failed, retrying in {sleep_seconds:.1f}s")
        time.sleep(sleep_seconds)

    raise RuntimeError(f"{context} failed after {attempts} attempts") from last_error


def download_file(url: str, output_path: Path) -> None:
    ensure_parent(output_path)
    response = request_with_retry("GET", url, context="File download", stream=True, timeout=60, attempts=3, backoff_seconds=2.0)
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
        timeout=300,
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
        response = requests.request("GET", status_url, headers=headers, timeout=30)
        raise_for_status_with_hint(response, f"Video task polling ({task_id})")
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
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            response = requests.request(
                "POST",
                provider["endpoint"],
                headers=headers,
                json=payload,
                timeout=30,
            )
            raise_for_status_with_hint(response, f"Video task creation ({asset['id']})")
            task_info = response.json()
            task_id = task_info.get("id")
            if not task_id:
                raise RuntimeError(f"Video task creation returned no task id: {json.dumps(task_info)}")

            result = poll_video_task(provider, task_id)
            video_url = extract_video_url(result)
            download_file(video_url, target)
            return target
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            sleep_seconds = 3.0 * attempt
            print(f"[RETRY] Video generation ({asset['id']}) draw {attempt}/3 failed, retrying in {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Video generation failed after 3 draws for asset {asset['id']}") from last_error


def ensure_tts_credentials(provider: dict[str, Any]) -> None:
    missing = [name for name in ("app_id", "access_key", "resource_id") if not str(provider.get(name, "")).strip()]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing TTS provider configuration: {joined}. Set TTS_PROVIDER_APP_ID / ACCESS_KEY / RESOURCE_ID.")


def get_audio_duration_seconds(path: Path) -> float | None:
    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        value = (probe.stdout or "").strip()
        if value:
            return float(value)
    except Exception:
        pass

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav_file:
                frame_rate = wav_file.getframerate() or 1
                return wav_file.getnframes() / frame_rate
        except Exception:
            return None
    return None


def tts_http_stream(url: str, headers: dict[str, str], params: dict[str, Any], audio_save_path: Path) -> None:
    session = requests.Session()
    response: requests.Response | None = None
    try:
        ensure_parent(audio_save_path)
        response = session.post(url, headers=headers, json=params, stream=True, timeout=300)
        if response.status_code not in (200,):
            details = response.text[:800].strip()
            raise RuntimeError(f"TTS request failed with status {response.status_code}. Response: {details}")

        audio_data = bytearray()
        for chunk in response.iter_lines(decode_unicode=True):
            if not chunk:
                continue
            data = json.loads(chunk)
            if data.get("code", 0) == 0 and data.get("data"):
                audio_data.extend(base64.b64decode(data["data"]))
                continue
            if data.get("code", 0) == 0 and data.get("sentence"):
                continue
            if data.get("code", 0) == 20000000:
                break
            if data.get("code", 0) > 0:
                raise RuntimeError(f"TTS error response: {data}")

        if not audio_data:
            raise RuntimeError("TTS generation returned empty audio data.")

        audio_save_path.write_bytes(audio_data)
        try:
            os.chmod(audio_save_path, 0o644)
        except Exception:
            pass
    finally:
        if response is not None:
            response.close()
        session.close()


def generate_audio(asset: dict[str, Any], provider: dict[str, Any], output_root: Path) -> tuple[Path, float | None]:
    ensure_tts_credentials(provider)
    target = output_root / asset["target"]
    text = str(asset.get("text") or asset.get("prompt") or "").strip()
    if not text:
        raise RuntimeError(f"Audio asset {asset['id']} is missing narration text.")

    if not target.exists():
        headers = {
            "X-Api-App-Id": str(provider["app_id"]),
            "X-Api-Access-Key": str(provider["access_key"]),
            "X-Api-Resource-Id": str(provider["resource_id"]),
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }
        payload = {
            "user": {"uid": f"media-agent-{asset['id']}"},
            "req_params": {
                "text": text,
                "speaker": provider.get("voice", "zh_female_cancan_mars_bigtts"),
                "audio_params": {
                    "format": provider.get("audio_format", "mp3"),
                    "sample_rate": int(provider.get("sample_rate", 24000)),
                    "enable_timestamp": True,
                },
                "additions": json.dumps(
                    {
                        "explicit_language": "zh",
                        "disable_markdown_filter": True,
                        "enable_timestamp": True,
                    },
                    ensure_ascii=False,
                ),
            },
        }
        tts_http_stream(str(provider["endpoint"]), headers, payload, target)

    duration = get_audio_duration_seconds(target)
    target_duration = float(asset.get("target_duration") or asset.get("duration") or 0)
    if duration is not None and target_duration > 0 and abs(duration - target_duration) > 0.12:
        print(
            f"[PROGRESS] Audio duration drift for {asset['id']}: target={target_duration:.3f}s actual={duration:.3f}s; "
            "will retime scenes using actual TTS duration."
        )
    return target, duration


def resolve_single_asset(
    asset: dict[str, Any], providers: dict[str, Any], output_root: Path, base_dir: Path, idx: int, total: int
) -> tuple[str, dict[str, Any]]:
    asset_type = asset["type"]
    desc = asset.get("description") or asset.get("id", "")
    print(f"[PROGRESS] Generating asset {idx}/{total}: {desc} ({asset_type})")
    duration: float | None = None
    if asset_type == "image":
        generated = generate_image(asset, providers["image"], output_root, base_dir)
    elif asset_type == "video":
        generated = generate_video(asset, providers["video"], output_root, base_dir)
    elif asset_type == "audio":
        generated, duration = generate_audio(asset, providers["audio"], output_root)
    else:
        raise ValueError(f"Unsupported asset type: {asset_type}")
    print(f"[PROGRESS] Asset {idx}/{total} done: {desc}")
    return asset["id"], {"resolved_path": generated.relative_to(output_root).as_posix(), "duration": duration}


def resolve_assets(pipeline: dict[str, Any], output_root: Path, base_dir: Path) -> dict[str, dict[str, Any]]:
    resolved: dict[str, dict[str, Any]] = {}
    providers = pipeline["providers"]
    total = len(pipeline["assets"])
    max_workers = min(len(pipeline["assets"]), 4) or 1
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(resolve_single_asset, asset, providers, output_root, base_dir, idx, total): asset["id"]
            for idx, asset in enumerate(pipeline["assets"], start=1)
        }
        for future in as_completed(futures):
            asset_id, payload = future.result()
            resolved[asset_id] = payload
    return resolved


def retime_scenes_from_audio(resolved_pipeline: dict[str, Any]) -> None:
    assets_by_id = {asset["id"]: asset for asset in resolved_pipeline.get("assets", [])}
    scenes = resolved_pipeline.get("scenes", [])
    if not scenes:
        return

    cumulative_shift = 0.0
    max_end = 0.0
    for scene in scenes:
        original_start = float(scene.get("start", 0.0))
        original_duration = float(scene.get("duration", 0.0))
        new_duration = original_duration

        audio_asset_id = scene.get("audio_asset_id")
        if audio_asset_id:
            audio_asset = assets_by_id.get(audio_asset_id)
            actual_duration = audio_asset.get("resolved_duration") if audio_asset else None
            if actual_duration is not None:
                new_duration = round(float(actual_duration), 3)
                scene["resolved_audio_duration"] = new_duration

        new_start = round(original_start + cumulative_shift, 3)
        scene["start"] = new_start
        scene["duration"] = new_duration

        delta = new_duration - original_duration
        cumulative_shift += delta
        max_end = max(max_end, new_start + new_duration)

    fmt = resolved_pipeline.setdefault("format", {})
    fmt["duration"] = round(max_end, 3)


def build_resolved_pipeline(pipeline: dict[str, Any], resolved_assets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    resolved_pipeline = json.loads(json.dumps(pipeline))
    asset_index = {asset["id"]: asset for asset in resolved_pipeline.get("assets", [])}
    for asset in resolved_pipeline.get("assets", []):
        asset_id = asset["id"]
        resolved = resolved_assets.get(asset_id, {})
        asset["resolved_path"] = resolved.get("resolved_path", "")
        asset["resolved"] = bool(asset.get("resolved_path"))
        if resolved.get("duration") is not None:
            asset["resolved_duration"] = round(float(resolved["duration"]), 3)

    for scene in resolved_pipeline.get("scenes", []):
        asset_id = scene.get("asset_id")
        audio_asset_id = scene.get("audio_asset_id")
        visual_asset = asset_index.get(asset_id) if asset_id else None
        if visual_asset and visual_asset.get("type") == "video":
            scene["audio_asset_id"] = None
            scene["voiceover_text"] = ""
            audio_asset_id = None
        audio_asset = asset_index.get(audio_asset_id) if audio_asset_id else None
        scene["resolved_asset_path"] = visual_asset.get("resolved_path", "") if visual_asset else ""
        scene["resolved_audio_path"] = audio_asset.get("resolved_path", "") if audio_asset else ""
        if audio_asset and audio_asset.get("resolved_duration") is not None:
            scene["resolved_audio_duration"] = audio_asset["resolved_duration"]
        if audio_asset and not scene.get("voiceover_text"):
            scene["voiceover_text"] = audio_asset.get("text", "")
    retime_scenes_from_audio(resolved_pipeline)
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
            f"- {asset.get('id')}: type={asset.get('type')}, source={asset.get('asset_source', 'generated')}, path={asset.get('resolved_path', '(missing)')}, duration={asset.get('resolved_duration', asset.get('target_duration', ''))}"
        )
    brief_lines.append("")
    brief_lines.append("## Scenes")
    for scene in resolved_pipeline.get("scenes", []):
        brief_lines.append(
            f"- {scene.get('id')}: start={scene.get('start')}s, duration={scene.get('duration')}s, title={scene.get('title', '')}, asset={scene.get('asset_id', '')}, audio={scene.get('audio_asset_id', '')}"
        )
        if scene.get("voiceover_text"):
            brief_lines.append(f"  voiceover: {scene.get('voiceover_text')}")
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
