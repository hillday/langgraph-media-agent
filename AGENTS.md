# LangGraph Media Agent

Web UI agent for prompt-to-video generation with HyperFrames.

## Core Flow

1. planner
2. executor
3. verifier
4. html author
5. validate
6. preview / render

## Key Files

- `app/graph.py` — LangGraph state machine
- `app/server.py` — FastAPI endpoints and Web UI entry
- `app/skill_registry.py` — dynamic skill loading from repo skill folders
- `app/file_tools.py` — model-callable `list_dir` / `read_file` / `write_file` / `patch_file` tools for authoring
- `app/pipeline_tools.py` — wrappers around `demo-minimal` media scripts
- `app/hyperframes_runner.py` — preview and render command helpers

## Rules

- Keep media generation separate from final HyperFrames rendering
- Final HTML must use local asset paths only
- Always preserve dynamic skill loading
- Keep file tools sandboxed to project write access and controlled read access
- Do not replace the web UI with a desktop client
- After changing the HTML authoring path, rerun lint/validate on generated projects
