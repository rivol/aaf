from typing import AsyncIterator, Callable, Optional

import anthropic
from anthropic import NOT_GIVEN, AsyncAnthropic, AsyncStream
from anthropic.types import ContentBlock, Message, RawMessageStreamEvent, TextBlock, ToolUseBlock, Usage

from ..logging import log
from ..tools_support.schema import jsonschema_for_function
from .base import APIConnectionError, ModelResponseStream, ModelRunner, RateLimitError, ResponseAdapterBase
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


class AnthropicResponseAdapter(ResponseAdapterBase):
    def __init__(self, stream: AsyncStream[RawMessageStreamEvent]):
        super().__init__()

        self.stream = stream

        self.full_message: Optional[Message] = None
        self.raw_chunks = []

    async def __aiter__(self) -> AsyncIterator[ResponseChunk]:
        log.debug("AnthropicResponseAdapter.__aiter__()", id=id(self))
        if self.full_message is not None:
            log.error("AnthropicResponseAdapter.__aiter__(): already processed", id=id(self))
            return

        block: Optional[ContentBlock] = None
        tool_partial_json = ""

        async for chunk in self.stream:
            log.debug("AnthropicResponseAdapter.__aiter__(): got raw chunk", chunk=chunk)
            self.raw_chunks.append(chunk)

            if chunk.type == "message_start":
                assert self.full_message is None
                self.full_message = chunk.message
                yield ResponseUsageChunk(delta=self.parse_usage(chunk.message.usage))

            elif chunk.type == "message_stop":
                assert self.full_message is not None
                assert block is None

            elif chunk.type == "content_block_start":
                assert block is None
                block = chunk.content_block
                tool_partial_json = ""

            elif chunk.type == "content_block_stop":
                assert block is not None
                assert len(self.full_message.content) == chunk.index

                if isinstance(block, ToolUseBlock):
                    block.input = tool_partial_json
                    yield ResponseChunkToolCall(tool_call=ToolCall(id=block.id, name=block.name, arguments=block.input))

                self.full_message.content.append(block)
                block = None

            elif chunk.type == "content_block_delta":
                assert block is not None
                if chunk.delta.type == "text_delta":
                    assert isinstance(block, TextBlock)
                    assert chunk.delta.text
                    block.text += chunk.delta.text
                    yield ResponseTextChunk(content=chunk.delta.text)
                elif chunk.delta.type == "input_json_delta":
                    assert isinstance(block, ToolUseBlock)
                    tool_partial_json += chunk.delta.partial_json
                else:
                    raise ValueError(f"Unexpected block delta type: {chunk.delta.type}")

            elif chunk.type == "message_delta":
                if chunk.usage:
                    self.full_message.usage.output_tokens += chunk.usage.output_tokens
                    yield ResponseUsageChunk(delta=self.parse_usage(chunk.usage))
                if chunk.delta.stop_reason:
                    self.full_message.stop_reason = chunk.delta.stop_reason
                    reason = self.parse_stop_reason(chunk.delta.stop_reason)
                    yield ResponseStopReasonChunk(reason=reason)
                if chunk.delta.stop_sequence:
                    self.full_message.stop_sequence = chunk.delta.stop_sequence

            else:
                raise ValueError(f"Unexpected chunk type: {chunk.type}")

        log.debug("AnthropicResponseAdapter.__aiter__(): done", id=id(self))

    def parse_usage(self, usage: Usage) -> CompletionUsage:
        return CompletionUsage(prompt_tokens=getattr(usage, "input_tokens", 0), completion_tokens=usage.output_tokens)

    def parse_stop_reason(self, ant_reason: Optional[str]) -> Optional[StopReason]:
        if ant_reason == "end_turn":
            return StopReason.END_TURN
        if ant_reason == "tool_use":
            return StopReason.TOOL_USE
        if ant_reason == "max_tokens":
            return StopReason.LENGTH

        return None


class AnthropicRunner(ModelRunner):
    MODELS = [
        ModelInfo(
            name="claude-3-7-sonnet-20250219",
            aliases=["sonnet", "sonnet-3.7"],
            cost=ModelCost(prompt_per_1m=3, completion_per_1m=15),
        ),
        ModelInfo(
            name="claude-3-5-sonnet-20241022",
            aliases=["sonnet-3.5"],
            cost=ModelCost(prompt_per_1m=3, completion_per_1m=15),
        ),
        ModelInfo(
            name="claude-3-5-sonnet-20240620",
            cost=ModelCost(prompt_per_1m=3, completion_per_1m=15),
        ),
    ]

    def __init__(self):
        super().__init__()
        self.client = AsyncAnthropic()

    def _transform_image_formats(self, messages: list[dict]) -> list[dict]:
        """Transform OpenAI-format image inputs to Anthropic-format.

        OpenAI format:
            "content": [
                {"type": "text", "text": "what do you see?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,..."
                    }
                }
            ]

        Anthropic format:
            "content": [
                {"type": "text", "text": "what do you see?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "..."  # base64 data without prefix
                    }
                }
            ]
        """

        def transform_base64_image(url: str) -> dict:
            """Extract media_type and data from a base64 data URL."""
            # Parse the URL to extract media type and base64 data
            # Format: data:image/jpeg;base64,ABC123...
            parts = url.split(";base64,", 1)
            if len(parts) == 2:
                media_type = parts[0].split(":", 1)[1]  # Get "image/jpeg" from "data:image/jpeg"
                base64_data = parts[1]

                return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64_data}}
            else:
                # Fallback for improperly formatted data URLs
                log.warning("Invalid data URL format for image, cannot extract media_type")
                return transform_url_image(url)

        def transform_url_image(url: str) -> dict:
            """Transform a regular URL to Anthropic image format."""
            return {"type": "image", "source": {"type": "url", "url": url}}

        def transform_image_item(content_item: dict) -> dict:
            """Transform an OpenAI image item to Anthropic format."""
            url = content_item["image_url"]["url"]

            # Handle base64 data URLs vs regular URLs
            if url.startswith("data:"):
                return transform_base64_image(url)
            else:
                return transform_url_image(url)

        # Main transformation logic
        result = []
        for message in messages:
            # If content is not a list, no transformation needed
            if not isinstance(message.get("content"), list):
                result.append(message)
                continue

            # Create a new message with transformed content
            new_message = {"role": message["role"]}
            new_content = []

            for item in message["content"]:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    new_content.append(transform_image_item(item))
                else:
                    # Keep non-image content as is
                    new_content.append(item)

            new_message["content"] = new_content
            result.append(new_message)

        return result

    async def run(self, model: str, request: ChatRequest, **kwargs) -> ModelResponseStream:
        self.log_run_request(model, request, kwargs)

        # Ensure the model is valid
        self.get_model_info(model)

        system_messages = [message for message in request.messages if message["role"] == "system"]
        assert len(system_messages) <= 1
        non_system_messages = [message for message in request.messages if message["role"] != "system"]

        # Transform OpenAI-format image inputs to Anthropic-format
        transformed_messages = self._transform_image_formats(non_system_messages)

        kwargs.setdefault("max_tokens", 4096)
        try:
            stream = await self.client.messages.create(
                model=model,
                messages=transformed_messages,
                system=system_messages[0]["content"] if system_messages else NOT_GIVEN,
                tools=self.get_tools_schema(request.tools) if request.tools else NOT_GIVEN,
                tool_choice={"type": "auto"} if request.tools else NOT_GIVEN,
                stream=True,
                **kwargs,
            )

        except anthropic.APIConnectionError as error:
            raise APIConnectionError(message=error.message) from error

        except anthropic.RateLimitError as error:
            metadata = {
                header: value
                for header, value in error.response.headers.items()
                if header.startswith("anthropic-ratelimit-")
            }
            # Flat 10-second retry for now
            raise RateLimitError(retry_in_secs=10, metadata=metadata) from error

        return ModelResponseStream(AnthropicResponseAdapter(stream))

    def get_tools_schema(self, tools: list[Callable]) -> list[dict]:
        return [jsonschema_for_function(func, parameters_key="input_schema") for func in tools]

    def get_tool_result_messages(self, results: list[ToolCallResult]) -> list[dict]:
        """Anthropic needs all tool results to be sent in a single message."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result.tool_call.id,
                        "is_error": result.is_error,
                        "content": result.content,
                    }
                    for result in results
                ],
            }
        ]

    def get_assistant_messages(self, stream: ModelResponseStream) -> list[dict]:
        if stream.stop_reason == StopReason.TOOL_USE:
            # Add assistant message with tool calls
            return [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.arguments_dict,
                        }
                        for tool_call in stream.get_tool_calls()
                    ],
                }
            ]
        else:
            # Add assistant message with completion
            return [{"role": "assistant", "content": stream.text}]
