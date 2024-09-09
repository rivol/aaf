import os
from typing import Optional

from .openai import OpenAIRunner
from .types import ModelInfo


class OllamaRunner(OpenAIRunner):
    MODELS = [
        ModelInfo(
            name="llama3.1:8b",
            aliases=["llama3.1", "llama3", "llama"],
        ),
    ]

    def __init__(self, *, base_url: Optional[str] = None, **kwargs):
        if base_url is None:
            base_url = os.environ.get("OLLAMA_BASE_URL")
        if base_url is None:
            base_url = f"http://localhost:11434/v1"

        super().__init__(base_url=base_url, **kwargs)
