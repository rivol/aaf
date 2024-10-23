import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Callable, Optional

from .forwarding import ResponseQueue, VirtualModelAdapter
from .llms import ModelRunner, get_llm_provider_and_model
from .llms.base import APIConnectionError, ModelResponseStream, RateLimitError
from .llms.types import (
    ChatRequest,
    CompletionUsage,
    CostAndUsage,
    ResponseChunkToolCall,
    ResponseChunkToolCallFailed,
    ResponseChunkToolCallFinished,
    ResponseChunkToolCallStarted,
    ResponseControlChunkRateLimited,
    ResponseDebugChunk,
    ResponseStopReasonChunk,
    ResponseStream,
    ToolCall,
    ToolCallResult,
)
from .logging import log
from .utils import truncate_text


class StatsTracking:
    """Helper base class that provides cost&usage and elapsed time tracking."""

    def __init__(self):
        super().__init__()

        self._started_at = datetime.now()

    def cost_and_usage(self) -> CostAndUsage:
        raise NotImplementedError()

    def elapsed_secs(self, now: Optional[datetime] = None) -> float:
        if now is None:
            now = datetime.now()

        return (now - self._started_at).total_seconds()


class Thread(StatsTracking):
    """Runs and represents a conversation with a model.

    Thread has methods to run either one step of a conversation (i.e. it gets one chat completion from the model),
    or it can run a loop which handles tool calls and multiple model completions.

    It also tracks cost and token usage.

    For actually running the model, it invokes the ModelRunner, asking it to run specified model
    with the given messages & parameters.
    """

    def __init__(
        self,
        model: str,
        request: ChatRequest,
        name: Optional[str] = None,
        *,
        runner: Optional[ModelRunner] = None,
        system: Optional[str] = None,
        tools: Optional[list[Callable]] = None,
    ):
        super().__init__()

        if runner is None:
            runner_cls, self.model = get_llm_provider_and_model(model)
            self.model_runner = runner_cls()
        else:
            self.model = model
            self.model_runner = runner

        self.name = name or "thread"
        self.request = request
        self.tools = {func.__name__: func for func in (tools or [])}

        self.messages = request.messages.copy()
        if system:
            self.messages.insert(0, {"role": "system", "content": system})

        self.runs_count = 0
        self.per_run_costs = []

    @asynccontextmanager
    async def run(
        self, *, name: Optional[str] = None, ignore_exceptions: bool = False, **kwargs
    ) -> AsyncIterator[ResponseStream]:
        log.info("Thread.run()")

        response = ResponseQueue()
        process_task = asyncio.create_task(self._run_step_background(response, name=name, **kwargs))
        stream = ModelResponseStream(VirtualModelAdapter(process_task, response))

        log.debug("Thread.run(): yield stream")
        yield stream

        await stream.finish(ignore_exceptions)
        log.info("Thread.run(): DONE")

    async def _run_step_background(self, response: ResponseQueue, *, name: Optional[str] = None, **kwargs) -> None:
        log.info("Thread._run_step_background()")

        if name is None:
            name = f"run_{self.runs_count}"

        request = self.request.model_copy(update={"messages": self.messages, "tools": list(self.tools.values())})

        # The loop handles retries for rate limits
        max_retries = 5
        for step in range(max_retries + 1):
            try:
                stream = await self.model_runner.run(self.model, request, **kwargs)
                break

            except RateLimitError as error:
                log.warning("Rate limit exceeded", step=step, error=error)
                if step >= max_retries:
                    await response.put(error)
                    await response.mark_finished()
                    return

                log.info("Rate limit recovery - waiting", step=step, seconds=error.retry_in_secs)
                await response.put(
                    ResponseControlChunkRateLimited(delay_secs=error.retry_in_secs, metadata=error.metadata)
                )
                await asyncio.sleep(error.retry_in_secs)

            except APIConnectionError as error:
                log.warning("API connection error", step=step, error=error)
                await asyncio.sleep(0.5)

            except Exception as error:
                # This will cause the user code to freeze, since the `yield` will not be executed.
                #  There are ways to fix it, but they're complicated, so for now we just print out the alert for the user.
                log.error(
                    f"ERROR: opening stream failed. Code will freeze!", model=self.model, runner=self.model_runner
                )
                log.critical(f"ERROR: {error}")
                raise

        else:
            log.error("Thread._run_step_background(): reached maximum number of iterations")
            await response.mark_finished()
            return

        try:
            async for chunk in stream:
                await response.put(chunk)

            # Ensure we've processed the full stream, even if the user didn't
            await stream.finish()
            await response.mark_finished()

            for msg in self.model_runner.get_assistant_messages(stream):
                self.add_message(**msg)

        finally:
            self.per_run_costs.append(self.create_cost_and_usage(name, stream.usage))
            self.runs_count += 1

    @asynccontextmanager
    async def run_loop(
        self, *, max_iterations: int = 10, ignore_exceptions: bool = False, **kwargs
    ) -> AsyncIterator[ResponseStream]:
        log.info("Thread.run_loop()", max_iterations=max_iterations)

        response = ResponseQueue()
        process_task = asyncio.create_task(self._run_loop_background(response, max_iterations=max_iterations, **kwargs))
        stream = ModelResponseStream(VirtualModelAdapter(process_task, response))

        log.debug("Thread.run_loop(): yield stream")
        yield stream

        await stream.finish(ignore_exceptions)
        log.info("Thread.run_loop(): DONE")

    async def _run_loop_background(self, response: ResponseQueue, max_iterations: int, **kwargs) -> None:
        log.debug("Thread.run_loop_background()", max_iterations=max_iterations)
        for i in range(max_iterations):
            tool_use_requests: list[ToolCall] = []
            async with self.run(ignore_exceptions=True, **kwargs) as step_stream:
                log.debug("Thread._run_loop_background(): stream", stream_id=id(step_stream))
                async for chunk in step_stream:
                    log.debug("Thread.run_loop_background(): got chunk", chunk=chunk)
                    if isinstance(chunk, Exception):
                        log.error("Thread.run_loop_background(): got exception", exception=chunk)
                        await response.put(chunk)
                        await response.mark_finished()
                        return

                    elif isinstance(chunk, ResponseChunkToolCall):
                        tool_use_requests.append(chunk.tool_call)
                    elif isinstance(chunk, ResponseStopReasonChunk):
                        log.info("Thread.run_loop_background(): got stop reason", reason=chunk.reason)
                    else:
                        # log.debug("Thread.run_loop_background(): putting chunk", chunk=chunk)
                        await response.put(chunk)
                log.debug("Thread._run_loop_background(): stream done", stream_id=id(step_stream))

            log.debug("Thread.run_loop_background(): loop finished")
            if not tool_use_requests:
                break

            # execute tool use requests
            await response.put(ResponseDebugChunk(content="Executing tool use requests"))
            results = await self.execute_tool_calls(tool_use_requests, response)
            for message in self.model_runner.get_tool_result_messages(results):
                self.add_message(**message)

        await response.mark_finished()

    def create_cost_and_usage(self, name: str, usage: CompletionUsage) -> CostAndUsage:
        return CostAndUsage(
            name=name,
            completion_tokens=usage.completion_tokens,
            prompt_tokens=usage.prompt_tokens,
            cost=self.model_runner.cost_in_usd(self.model, usage),
        )

    def cost_and_usage(self) -> CostAndUsage:
        return CostAndUsage.aggregate(self.name, self.per_run_costs)

    def add_message(self, role: str, content: str, **kwargs) -> None:
        self.messages.append({"role": role, "content": content, **kwargs})

    async def execute_tool_calls(self, tool_calls: list[ToolCall], queue: ResponseQueue) -> list[ToolCallResult]:
        """Executes the tool calls generated by the model.

        Message containing the requested tool calls is added to the stream.
        Another message is added to the stream for each tool call result.
        """

        # self.event_handler.debug(f"Executing {len(tool_calls)} tool calls...")
        results = await asyncio.gather(*[self.execute_tool_call(tool_call, queue) for tool_call in tool_calls])
        return list(results)

    async def execute_tool_call(self, tool_call: ToolCall, queue: ResponseQueue) -> ToolCallResult:
        """Executes a single tool call, which can be async or sync."""

        log.info("Thread.execute_tool_call()", name=tool_call.name, arguments=tool_call.arguments)
        try:
            await queue.put(ResponseChunkToolCallStarted(tool_call=tool_call))

            tool = self.tools[tool_call.name]
            params = tool_call.arguments_dict.copy()

            if asyncio.iscoroutinefunction(tool):
                result = await tool(**params)
            else:
                result = tool(**params)

        except Exception as error:
            log.warning("Thread.execute_tool_call(): error", name=tool_call.name, arguments=tool_call.arguments)
            await queue.put(ResponseChunkToolCallFailed(tool_call=tool_call, error=repr(error)))
            return ToolCallResult(tool_call=tool_call, is_error=True, content="Error")
        else:
            log.info(
                "Thread.execute_tool_call(): result",
                name=tool_call.name,
                arguments=tool_call.arguments,
                result=truncate_text(result, 500) if isinstance(result, str) else result,
            )
            await queue.put(ResponseChunkToolCallFinished(tool_call=tool_call, result=result))
            return ToolCallResult(tool_call=tool_call, content=json.dumps(result))


class Session(StatsTracking):
    """Session keeps configuration and tracks costs.

    It's usually the main entrypoint for creating threads, and thus starting conversations.
    """

    def __init__(self, name: str = "root"):
        super().__init__()

        self.name = name
        self.config: dict = {}

        self.threads = []

    def create_thread(
        self,
        model: str,
        request: Optional[ChatRequest] = None,
        name: Optional[str] = None,
        *,
        runner: Optional[ModelRunner] = None,
        system: Optional[str] = None,
        tools: Optional[list[Callable]] = None,
    ) -> Thread:
        if request is None:
            request = ChatRequest()

        thread = Thread(model=model, request=request, name=name, runner=runner, system=system, tools=tools)
        self.threads.append(thread)
        return thread

    def cost_and_usage(self) -> CostAndUsage:
        return CostAndUsage.aggregate(self.name, [thread.cost_and_usage() for thread in self.threads])
