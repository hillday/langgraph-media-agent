from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from .config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    api_key = os.environ.get(settings.model_api_key_env, "")
    if not api_key:
        raise RuntimeError(
            f"Missing model API key in env var `{settings.model_api_key_env}`. "
            "Update .env or export the key before starting the agent."
        )

    return ChatOpenAI(
        model=settings.model_name,
        api_key=api_key,
        base_url=settings.model_api_base,
        temperature=0.2,
    )
