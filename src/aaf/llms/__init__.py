from .anthropic import AnthropicRunner
from .base import ModelRunner
from .ollama import OllamaRunner
from .openai import OpenAIRunner
from .openrouter import OpenRouterRunner

PROVIDERS = [OpenAIRunner, AnthropicRunner, OllamaRunner, OpenRouterRunner]


def get_llm_provider_and_model(model_name: str) -> tuple[type["ModelRunner"], str]:
    return ModelRunner.get_provider_and_model(PROVIDERS, model_name)
