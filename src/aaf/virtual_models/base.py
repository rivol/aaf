import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from ..forwarding import ResponseQueue, VirtualModelAdapter
from ..llms import ModelRunner
from ..llms.base import ModelResponseStream
from ..llms.types import ChatRequest, CompletionUsage
from ..logging import log
from ..threads import Session, Thread


class VirtualModelBase(ModelRunner):
    """Base class for virtual models.

    Virtual models usually add custom logic on top of the standard LLMs, and usually implement more complex flows,
    e.g. calling multiple models in sequence.

    The main method to implement is process(), which should write response chunks to the provided queue.
    """

    id: str
    name: str | None = None

    @property
    def display_name(self) -> str:
        """Returns the display name for the model - either configured name or class name."""
        return self.name or self.__class__.__name__

    async def run(self, model: str, request: ChatRequest, **kwargs) -> ModelResponseStream:
        queue = ResponseQueue()
        process_task = asyncio.create_task(self._process_wrapper(request, queue))
        return ModelResponseStream(VirtualModelAdapter(process_task, queue))

    async def _process_wrapper(self, request: ChatRequest, queue: ResponseQueue) -> None:
        try:
            await self.process(request, queue)
            log.debug(f"{self.__class__.__name__}._process_wrapper(): process finished")
        except Exception as error:
            log.critical(f"{self.__class__.__name__}.process() failed", error=error)
            await queue.add(f"Error: {repr(error)}")
        finally:
            await queue.put(None)

    async def process(self, chat: ChatRequest, queue: ResponseQueue) -> None:
        raise NotImplementedError()

    def get_assistant_messages(self, stream: ModelResponseStream) -> list[dict]:
        return [{"role": "assistant", "content": stream.text}]

    def cost_in_usd(self, model: str, usage: CompletionUsage) -> float:
        # TODO: not entirely correct, since we use sub-models, but can't track them ATM
        return 0

    async def process_continuation(
        self, session: Session, chat: ChatRequest, queue: ResponseQueue, model_name: str, **kwargs
    ) -> None:
        log.info("process_continuation()", model=self.id, messages_count=len(chat.messages))
        thread = session.create_thread(model_name, name="continuation")
        for message in chat.messages:
            thread.add_message(**message)

        async with thread.run(**kwargs) as stream:
            await stream.text_chunks().redirect(queue)

    @asynccontextmanager
    async def run_progressbar(self, queue: ResponseQueue, delay: float = 1.0, *, message: Optional[str] = None) -> None:
        """Runs a progress bar in the background, yielding dots to the queue.

        The progress bar is stopped when the context manager exits.
        """

        # If message is provided, output it first in italics
        if message:
            await queue.add(f"_{message}_ ")

        task = asyncio.create_task(self._run_progressbar_task(queue, delay))
        try:
            yield
        finally:
            task.cancel()
            await queue.add("\n")

    async def _run_progressbar_task(self, queue: ResponseQueue, delay: float) -> None:
        while True:
            await queue.add(".\u200b")
            await asyncio.sleep(delay)

    async def output_addendum(self, queue: ResponseQueue, session_or_thread: Session | Thread) -> None:
        await queue.add("\n\n---\n")
        await queue.add(f"Elapsed: {session_or_thread.elapsed_secs():.3f} secs\n")
        await queue.add("Costs: " + session_or_thread.cost_and_usage().pretty() + "\n")
