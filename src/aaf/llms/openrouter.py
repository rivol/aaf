from os import getenv

from .openai import OpenAIRunner
from .types import ModelCost, ModelInfo


class OpenRouterRunner(OpenAIRunner):
    MODELS = [
        # OpenAI o1 models
        ModelInfo(
            name="openai/o1-mini-2024-09-12",
            aliases=["o1-mini"],
            cost=ModelCost(prompt_per_1m=3, completion_per_1m=12),
        ),
        ModelInfo(
            name="openai/o1-preview-2024-09-12",
            aliases=["o1", "o1-preview"],
            cost=ModelCost(prompt_per_1m=15, completion_per_1m=60),
        ),
        # Llama 3.1 models
        ModelInfo(
            name="meta-llama/llama-3.1-8b-instruct",
            aliases=["llama-3.1-8b", "llama-8b"],
            cost=ModelCost(prompt_per_1m=0.07, completion_per_1m=0.07),  # approx cost
        ),
        ModelInfo(
            name="meta-llama/llama-3.1-70b-instruct",
            aliases=["llama-3.1-70b", "llama-70b"],
            cost=ModelCost(prompt_per_1m=0.4, completion_per_1m=0.4),  # approx cost
        ),
        ModelInfo(
            name="meta-llama/llama-3.1-405b-instruct",
            aliases=["llama-3.1-405b", "llama-405b"],
            cost=ModelCost(prompt_per_1m=0.4, completion_per_1m=0.4),  # approx cost
        ),
    ]

    def __init__(self, **kwargs):
        api_key = getenv("OPENROUTER_API_KEY")
        super().__init__(base_url="https://openrouter.ai/api/v1", api_key=api_key)
