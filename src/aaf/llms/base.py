import asyncio
from typing import AsyncIterator, Optional

from .. import logging
from ..logging import log
from ..utils import truncate_text
from .types import (
    ChatRequest,
    CompletionUsage,
    ModelInfo,
    ResponseChunk,
    ResponseChunkCompleteText,
    ResponseChunkToolCall,
    ResponseControlChunkStreamBegin,
    ResponseControlChunkStreamEnd,
    ResponseStopReasonChunk,
    ResponseStream,
    ResponseTextChunk,
    ResponseUsageChunk,
    StopReason,
    ToolCall,
    ToolCallResult,
)


class RateLimitError(Exception):
    def __init__(self, retry_in_secs: float, metadata: dict):
        super().__init__()

        self.retry_in_secs = retry_in_secs
        self.metadata = metadata


class APIConnectionError(Exception):
    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r})"


class ResponseAdapterBase:
    """Base class for adapters which translate model-specific response stream into a common format.

    It should take model-specific response and yield following ResponseChunk instances:
    - ResponseTextChunk (additive)
    - ResponseUsageChunk (additive)
    - ResponseChunkToolCall
    - ResponseStopReasonChunk

    Chunks should be emitted as soon as they are available, in the original order.
    All Text and ToolCall chunks should be emitted before the StopReason chunk.
    """

    async def __aiter__(self) -> AsyncIterator[ResponseChunk]:
        raise NotImplementedError()


class ModelResponseStream(ResponseStream):
    """Stateful response stream from a model.

    It works with adapter to get basic response stream from model, and adds some higher-level stateful logic.

    For example, it keeps track of full response text, tool calls, stop reason, and token usage.
    It also adds some control-flow chunks to the response stream, e.g for detecting beginning or end of the stream.

    It supports pausing and then continuing processing of the stream, e.g across multiple for loops.
    """

    def __init__(self, adapter: ResponseAdapterBase):
        super().__init__()

        self._text = ""
        self._usage = CompletionUsage()
        self._stop_reason = None
        self._is_finished = False
        self._tool_calls = []
        self._exception = None

        self.processed_queue = asyncio.Queue()
        self.processed_chunks: list[ResponseChunk] = []

        self.run_adapter_task = asyncio.create_task(self.background_process(adapter, self.processed_queue))

    @property
    def text(self) -> str:
        return self._text

    @property
    def usage(self) -> CompletionUsage:
        return self._usage

    @property
    def stop_reason(self) -> Optional[StopReason]:
        return self._stop_reason

    async def __aiter__(self) -> AsyncIterator[ResponseChunk]:
        if self._is_finished:
            return

        while True:
            chunk = await self.processed_queue.get()
            if chunk is None:
                self._is_finished = True
                break

            self.processed_chunks.append(chunk)

            if isinstance(chunk, Exception):
                self._exception = chunk

            yield chunk

    async def background_process(self, adapter: ResponseAdapterBase, queue: asyncio.Queue) -> None:
        """The main method for iterating over the response stream.

        It's implemented as wrapper around iterate_raw_stream(). The latter is responsible for emitting 'raw' chunks
        from the stream.
        This method adds some generic, higher-level logic for flow control and state management.
        """

        if self._is_finished:
            return

        log.debug("ModelResponseStream.background_process()", stream_id=id(self))
        await queue.put(ResponseControlChunkStreamBegin())

        async for chunk in adapter:
            log.debug("ModelResponseStream.background_process(): got chunk", stream_id=id(self), chunk=chunk)
            if isinstance(chunk, ResponseTextChunk):
                self._text += chunk.content
            if isinstance(chunk, ResponseUsageChunk):
                self._usage += chunk.delta
            if isinstance(chunk, ResponseChunkToolCall):
                self._tool_calls.append(chunk.tool_call)

            if isinstance(chunk, ResponseStopReasonChunk):
                # End of stream
                self._stop_reason = chunk.reason

                if self._text:
                    await queue.put(ResponseChunkCompleteText(content=self._text))

            await queue.put(chunk)
            log.debug("ModelResponseStream.background_process(): added to queue", stream_id=id(self), chunk=chunk)

        await queue.put(ResponseControlChunkStreamEnd())
        await queue.put(None)
        log.debug("ModelResponseStream.background_process(): DONE", stream_id=id(self))

    async def finish(self, ignore_exceptions: bool = False) -> None:
        log.debug("ModelResponseStream.finish(): finishing...", stream_id=id(self))
        await self.run_adapter_task

        # Consume the stream
        async for _ in self:
            pass

        if self._exception is not None and not ignore_exceptions:
            log.error("ModelResponseStream.finish(): exception", stream_id=id(self), exception=self._exception)
            raise self._exception

        log.debug("ModelResponseStream.finish(): done", stream_id=id(self))

    def get_tool_calls(self) -> list[ToolCall]:
        return self._tool_calls


class ModelRunner:
    """Invokes model with given messages and parameters."""

    # Lists all supported models. Each element is (model_name, aliases).
    MODELS: list[ModelInfo]

    def __init__(self):
        pass

    async def run(self, model: str, request: ChatRequest, **kwargs) -> ModelResponseStream:
        raise NotImplementedError()

    def cost_in_usd(self, model: str, usage: CompletionUsage) -> float:
        cost_info = self.get_model_info(model).cost
        return (
            usage.prompt_tokens * cost_info.prompt_per_1m / 1_000_000
            + usage.completion_tokens * cost_info.completion_per_1m / 1_000_000
        )

    @classmethod
    def get_provider_and_model(
        cls, providers: list[type["ModelRunner"]], model_name: str
    ) -> tuple[type["ModelRunner"], str]:
        """Resolves the user-specified model name into a provider and a canonical model name."""

        for provider in providers:
            for info in provider.MODELS:
                if model_name in [info.name] + info.aliases:
                    return provider, info.name

        raise ValueError(f"Model {model_name} is not supported by any of the providers: {providers}")

    @classmethod
    def get_model_info(cls, model: str) -> ModelInfo:
        for model_info in cls.MODELS:
            if model in [model_info.name] + model_info.aliases:
                return model_info
        else:
            raise ValueError(f"Model {model} is not supported by {cls.__name__}")

    def create_tool_result_message(self, tool_call: ToolCall, result: str) -> dict:
        raise NotImplementedError()

    def create_tool_error_message(self, tool_call: ToolCall) -> dict:
        raise NotImplementedError()

    def get_tool_result_messages(self, results: list[ToolCallResult]) -> list[dict]:
        raise NotImplementedError()

    def get_assistant_messages(self, stream: ModelResponseStream) -> list[dict]:
        raise NotImplementedError()

    def log_run_request(self, model: str, request: ChatRequest, kwargs: dict) -> None:
        event = f"{self.__class__.__name__}.run()"
        if logging.min_level <= logging.DEBUG:
            log.debug(event, model=model, request=request, kwargs=kwargs)
        else:
            # Create a version of messages with content truncated, to make the log message more manageable
            brief_request = request.model_copy(
                update={"messages": self.truncate_messages_for_logging(request.messages)}
            )
            log.info(event, model=model, request=brief_request, kwargs=kwargs)

    def truncate_messages_for_logging(self, messages: list[dict]) -> list[dict]:
        truncated = []
        for full_message in messages:
            message = full_message.copy()
            if isinstance(message["content"], str):
                message["content"] = truncate_text(message["content"], 25)
            if message.get("tool_calls"):
                message["tool_calls"] = f"<{len(message['tool_calls'])} tool call requests>"
            truncated.append(message)

        return truncated
