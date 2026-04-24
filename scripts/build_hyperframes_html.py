import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return normalized or f"composition-{int(time.time())}"


def load_pipeline(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def find_asset(pipeline: dict[str, Any], asset_id: str) -> dict[str, Any] | None:
    for asset in pipeline.get("assets", []):
        if asset.get("id") == asset_id:
            return asset
    return None


def resolved_path(pipeline: dict[str, Any], asset_id: str) -> str:
    asset = find_asset(pipeline, asset_id)
    if not asset:
        return ""
    return str(asset.get("resolved_path", ""))


def media_src(path: str) -> str:
    if not path:
        return ""
    return path if path.startswith("./") else f"./{path.lstrip('./')}"


def build_html(pipeline: dict[str, Any]) -> str:
    fmt = pipeline["format"]
    composition_id = slugify(pipeline.get("project_name", "media-project"))
    scenes = pipeline.get("scenes", [])

    def scene_markup(scene: dict[str, Any], idx: int) -> str:
        start = scene["start"]
        duration = scene["duration"]
        visual_path = media_src(resolved_path(pipeline, scene.get("asset_id", "")))
        visual_asset = find_asset(pipeline, scene.get("asset_id", ""))
        audio_path = ""
        if not (visual_asset and visual_asset.get("type") == "video"):
            audio_path = media_src(resolved_path(pipeline, scene.get("audio_asset_id", "")))
        visual_html = ""
        if visual_path.endswith((".mp4", ".webm", ".mov")):
            visual_html = f"""<video id="visual-{idx}" class="clip visual" src="{visual_path}" muted playsinline loop data-start="{start}" data-duration="{duration}" data-track-index="{idx*10}"></video>"""
        elif visual_path:
            visual_html = f"""<img id="visual-{idx}" class="clip visual" src="{visual_path}" data-start="{start}" data-duration="{duration}" data-track-index="{idx*10}" />"""

        audio_html = ""
        if audio_path:
            audio_html = (
                f"""<audio id="audio-{idx}" class="clip narration" src="{audio_path}" """
                f"""data-start="{start}" data-duration="{duration}" data-track-index="{idx*10+1}" data-volume="1"></audio>"""
            )

        kicker = scene.get("kicker", "")
        body = scene.get("body") or scene.get("subtitle", "")
        points = scene.get("points", [])
        points_html = ""
        if points:
            items = "\n".join(f"<li>{p}</li>" for p in points)
            points_html = f"<ul class=\"points\">{items}</ul>"

        return f"""
      <section id="{scene['id']}" class="scene clip" data-start="{start}" data-duration="{duration}" data-track-index="{idx*10+2}">
        {visual_html}
        {audio_html}
        <div class="overlay"></div>
        <div class="copy">
          <div class="kicker">{kicker}</div>
          <h1>{scene.get("title","")}</h1>
          <p>{body}</p>
          {points_html}
        </div>
      </section>
"""

    scenes_html = "\n".join(scene_markup(scene, i) for i, scene in enumerate(scenes, start=1))

    timeline_lines: list[str] = []
    for scene in scenes:
        start = float(scene["start"])
        scene_id = scene["id"]
        timeline_lines.append(
            f'tl.from("#{scene_id} .copy", {{ opacity: 0, y: 30, duration: 0.8, ease: "power3.out" }}, {start + 0.2});'
        )

    timeline = "\n        ".join(timeline_lines)

    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width={fmt['width']}, height={fmt['height']}" />
    <title>{pipeline.get('project_name','media-project')}</title>
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>
      * {{ box-sizing: border-box; }}
      html, body {{
        margin: 0;
        width: {fmt['width']}px;
        height: {fmt['height']}px;
        overflow: hidden;
        background: #060914;
        color: #f4f7ff;
        font-family: Inter, system-ui, Arial, sans-serif;
      }}
      #root {{ position: relative; width: {fmt['width']}px; height: {fmt['height']}px; }}
      .scene {{ position: absolute; inset: 0; }}
      .visual {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; opacity:0.45; }}
      .overlay {{ position:absolute; inset:0; background: linear-gradient(90deg, rgba(5,9,20,0.78), rgba(5,9,20,0.2)); }}
      .copy {{
        position:absolute; left:112px; top:120px; width:760px;
        padding:42px 46px; border-radius:28px;
        background: rgba(8,14,28,0.58); border:1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(14px);
      }}
      .kicker {{
        display:inline-flex; padding:10px 16px; border-radius:999px;
        border:1px solid rgba(255,255,255,0.14); background: rgba(255,255,255,0.06);
        font-size:22px; margin-bottom:20px;
      }}
      h1 {{ margin:0 0 14px; font-size:88px; line-height:0.94; letter-spacing:-2px; }}
      p {{ margin:0; font-size:31px; line-height:1.28; color: rgba(244,247,255,0.9); }}
      .points {{ margin:22px 0 0; padding-left:24px; font-size:22px; line-height:1.4; color: rgba(244,247,255,0.82); }}
    </style>
  </head>
  <body>
    <div
      id="root"
      data-composition-id="{composition_id}"
      data-start="0"
      data-duration="{fmt['duration']}"
      data-width="{fmt['width']}"
      data-height="{fmt['height']}"
    >
{scenes_html}
      <script id="generation-spec" type="application/json">{json.dumps(pipeline, ensure_ascii=True)}</script>
      <script>
        window.__timelines = window.__timelines || {{}};
        const tl = gsap.timeline({{ paused: true }});
        {timeline}
        window.__timelines["{composition_id}"] = tl;
      </script>
    </div>
  </body>
</html>
"""


def write_project_files(pipeline: dict[str, Any], output_root: Path, html: str) -> None:
    ensure_parent(output_root / "index.html")
    (output_root / "index.html").write_text(html, encoding="utf-8")
    meta = {
        "id": slugify(pipeline.get("project_name", "media-project")),
        "name": pipeline.get("project_name", "media-project"),
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: build HyperFrames index.html from pipeline.resolved.json.")
    parser.add_argument("--pipeline", required=True, help="Path to pipeline.resolved.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for the HyperFrames project")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline_path = Path(args.pipeline).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    pipeline = load_pipeline(pipeline_path)
    html = build_html(pipeline)
    write_project_files(pipeline, output_root, html)
    print(f"Wrote index.html + meta.json to: {output_root}")


if __name__ == "__main__":
    main()
