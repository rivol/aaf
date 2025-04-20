from typing import AsyncIterator, Callable, Optional

import openai
from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage as OpenAICompletionUsage

from ..logging import log
from ..tools_support.schema import jsonschema_for_function
from .base import APIConnectionError, ModelResponseStream, ModelRunner, ResponseAdapterBase
from .types import (
    ChatRequest,
    CompletionUsage,
    ModelCost,
    ModelInfo,
    ResponseChunk,
    ResponseChunkToolCall,
    ResponseStopReasonChunk,
    ResponseTextChunk,
    ResponseUsageChunk,
    StopReason,
    ToolCall,
    ToolCallResult,
)


class OpenAIResponseAdapter(ResponseAdapterBase):
    def __init__(self, stream: AsyncStream[ChatCompletionChunk]):
        self.stream = stream

        self.raw_chunks = []

    async def __aiter__(self) -> AsyncIterator[ResponseChunk]:
        log.debug(f"{self.__class__.__name__}.__aiter__()", id=id(self))
        async for chunk in self.stream:
            log.debug(f"{self.__class__.__name__}.__aiter__(): got raw chunk", chunk=chunk)
            self.raw_chunks.append(chunk)

            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield ResponseTextChunk(content=content)

            if hasattr(chunk, "usage") and chunk.usage is not None:
                yield ResponseUsageChunk(delta=self.parse_usage(chunk.usage))

            if chunk.choices and chunk.choices[0].finish_reason:
                # Emit any and all tool calls
                for tool_call in self.get_tool_calls():
                    yield ResponseChunkToolCall(tool_call=tool_call)

                reason = self.parse_stop_reason(chunk.choices[0].finish_reason)
                yield ResponseStopReasonChunk(reason=reason)

        log.debug(f"{self.__class__.__name__}.__aiter__(): done", id=id(self))

    def parse_usage(self, usage: OpenAICompletionUsage) -> CompletionUsage:
        return CompletionUsage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens)

    def parse_stop_reason(self, openai_reason: Optional[str]) -> Optional[StopReason]:
        if openai_reason == "stop":
            return StopReason.END_TURN
        if openai_reason == "tool_calls":
            return StopReason.TOOL_USE
        if openai_reason == "length":
            return StopReason.LENGTH

        return None

    def get_tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []

        # Iterate through each chunk
        chunk: ChatCompletionChunk
        for chunk in self.raw_chunks:
            if not chunk.choices:
                continue

            assert len(chunk.choices) == 1
            choice = chunk.choices[0]  # Assumes one choice for simplicity

            # Check if the choice has tool calls
            if not choice.delta.tool_calls:
                continue
            for tool_call in choice.delta.tool_calls:
                idx = tool_call.index
                if idx >= len(calls):
                    calls.append(ToolCall(id="", name="", arguments=""))

                if tool_call.function:
                    # Should id & name be appended or replaced?
                    if tool_call.id:
                        calls[idx].id = tool_call.id
                    if tool_call.function.name:
                        calls[idx].name = tool_call.function.name
                    if tool_call.function.arguments:
                        calls[idx].arguments += tool_call.function.arguments

        return calls


class OpenAIRunner(ModelRunner):
    MODELS = [
        ModelInfo(
            name="gpt-4o-2024-11-20",
            aliases=["gpt-4o", "4o"],
            cost=ModelCost(prompt_per_1m=2.50, completion_per_1m=10),
        ),
        ModelInfo(
            name="gpt-4o-2024-08-06",
            cost=ModelCost(prompt_per_1m=2.50, completion_per_1m=10),
        ),
        ModelInfo(
            name="gpt-4o-mini-2024-07-18",
            aliases=["gpt-4o-mini"],
            cost=ModelCost(prompt_per_1m=0.15, completion_per_1m=0.60),
        ),
        ModelInfo(
            name="chatgpt-4o-latest",
            aliases=["chatgpt", "chat"],
            cost=ModelCost(prompt_per_1m=5, completion_per_1m=15),
        ),
        ModelInfo(
            name="gpt-4.1-2025-04-14",
            aliases=["gpt-4.1", "gpt4", "gpt"],
            cost=ModelCost(prompt_per_1m=2, completion_per_1m=8),
        ),
        ModelInfo(
            name="gpt-4.1-mini-2025-04-14",
            aliases=["gpt-4.1-mini", "gpt-mini"],
            cost=ModelCost(prompt_per_1m=0.4, completion_per_1m=1.6),
        ),
        ModelInfo(
            name="gpt-4.1-nano-2025-04-14",
            aliases=["gpt-4.1-nano", "gpt-nano"],
            cost=ModelCost(prompt_per_1m=0.1, completion_per_1m=0.4),
        ),
        ModelInfo(
            name="o3-2025-04-16",
            aliases=["o3"],
            cost=ModelCost(prompt_per_1m=10, completion_per_1m=40),
        ),
        ModelInfo(
            name="o4-mini-2025-04-16",
            aliases=["o4-mini"],
            cost=ModelCost(prompt_per_1m=1.1, completion_per_1m=4.4),
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__()
        self.client = AsyncOpenAI(**kwargs)

    async def run(self, model: str, request: ChatRequest, **kwargs) -> ModelResponseStream:
        self.log_run_request(model, request, kwargs)

        # Ensure the model is valid
        self.get_model_info(model)

        # Set defaults if not provided
        if request.max_tokens is not None:
            kwargs.setdefault("max_tokens", request.max_tokens)

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=request.messages,
                stream=True,
                stream_options={"include_usage": True},
                tools=self.get_tools_schema(request.tools) if request.tools else NOT_GIVEN,
                **kwargs,
            )
        except openai.APIConnectionError as error:
            raise APIConnectionError(message=error.message) from error

        return ModelResponseStream(OpenAIResponseAdapter(stream))

    def get_tools_schema(self, tools: list[Callable]) -> list[dict]:
        return [{"type": "function", "function": jsonschema_for_function(func)} for func in tools]

    def get_tool_result_messages(self, results: list[ToolCallResult]) -> list[dict]:
        """OpenAI uses one message per tool call result."""

        return [
            {
                "role": "tool",
                "content": result.content if not result.is_error else "Error",
                "tool_call_id": result.tool_call.id,
                "name": result.tool_call.name,
            }
            for result in results
        ]

    def get_assistant_messages(self, stream: ModelResponseStream) -> list[dict]:
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
