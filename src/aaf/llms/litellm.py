import os
from typing import Callable

# Use local model cost map for LiteLLM, avoiding fairly large download at startup
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "1"
import litellm
from litellm import acompletion

from ..tools_support.schema import jsonschema_for_function
from .base import APIConnectionError, ModelResponseStream, ModelRunner, RateLimitError
from .openai import OpenAIResponseAdapter
from .types import ChatRequest, ModelCost, ModelInfo, StopReason, ToolCallResult


class LiteLLMRunner(ModelRunner):
    """LiteLLM provider implementation that supports multiple LLM providers through a unified API."""

    MODELS = [
        ModelInfo(
            name="gemini/gemini-2.0-flash",
            aliases=["gemini-2.0-flash", "gemini-flash", "flash"],
            cost=ModelCost(prompt_per_1m=0.10, completion_per_1m=0.40),
        ),
        ModelInfo(
            name="gemini/gemini-2.5-pro-preview-03-25",
            aliases=["gemini-2.5-pro", "gemini-pro"],
            # Note that prices are for <= 200k tokens
            cost=ModelCost(prompt_per_1m=1.25, completion_per_1m=10.00),
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__()
        # We don't need to initialize a client as we'll use acompletion directly

    async def run(self, model: str, request: ChatRequest, **kwargs) -> ModelResponseStream:
        self.log_run_request(model, request, kwargs)

        # Ensure the model is valid and get canonical name
        model_info = self.get_model_info(model)

        # Extract the actual model name (after litellm/ prefix if present)
        if model_info.name.startswith("litellm/"):
            actual_model = model_info.name.split("/", 1)[1]
        else:
            actual_model = model_info.name

        # Set defaults if not provided
        kwargs.setdefault("max_tokens", request.max_tokens)

        try:
            # Use litellm's acompletion function directly
            stream = await acompletion(
                model=actual_model,
                messages=request.messages,
                stream=True,
                stream_options={"include_usage": True},
                tools=self.get_tools_schema(request.tools) if request.tools else None,
                **kwargs,
            )

            return ModelResponseStream(OpenAIResponseAdapter(stream))

        except litellm.RateLimitError as error:
            # Handle rate limit error
            metadata = getattr(error, "metadata", {})
            retry_after = getattr(error, "retry_after", 10)
            raise RateLimitError(retry_in_secs=retry_after, metadata=metadata) from error

        except litellm.ServiceUnavailableError as error:
            # Handle service unavailable
            raise APIConnectionError(message=str(error)) from error

        except litellm.BadRequestError as error:
            # Handle bad request
            raise APIConnectionError(message=str(error)) from error

        except Exception as error:
            # Generic error handling
            raise APIConnectionError(message=str(error)) from error

    # Note that the following functions are also in OpenAIRunner - should merge...

    def get_tools_schema(self, tools: list[Callable]) -> list[dict]:
        """Convert tools to the format expected by LiteLLM (OpenAI format)."""
        return [{"type": "function", "function": jsonschema_for_function(func)} for func in tools]

    def get_tool_result_messages(self, results: list[ToolCallResult]) -> list[dict]:
        """Create tool result messages in OpenAI format."""
        return [
            {
                "role": "tool",
                "content": result.content if not result.is_error else "Error",
                "tool_call_id": result.tool_call.id,
            }
            for result in results
        ]

    def get_assistant_messages(self, stream: ModelResponseStream) -> list[dict]:
        """Create assistant messages from stream in OpenAI format."""
        if stream.stop_reason == StopReason.TOOL_USE:
            # Add assistant message with tool calls
            return [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": tool_call.name, "arguments": tool_call.arguments},
                        }
                        for tool_call in stream.get_tool_calls()
                    ],
                }
            ]
        else:
            # Add assistant message with completion
            return [{"role": "assistant", "content": stream.text}]
