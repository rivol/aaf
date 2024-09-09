import asyncio
from asyncio import Queue
from typing import AsyncIterator

from .llms.base import ResponseAdapterBase
from .llms.types import (
    ResponseChunk,
    ResponseDebugChunk,
    ResponseStopReasonChunk,
    ResponseTextChunk,
    ResponseVerboseChunk,
    StopReason,
)


class ResponseQueue(Queue):
    """Queue for sending response chunks.

    This is similar to ResponseStream, but focuses on the writing part (whereas ResponseStream focuses on reading).
    It's a subclass of asyncio.Queue, with some convenience methods for adding different types of chunks.
    """

    async def add(self, content: str) -> None:
        await self.put(ResponseTextChunk(content=content))

    async def add_debug(self, content: str) -> None:
        await self.put(ResponseDebugChunk(content=content))

    async def add_verbose(self, content: str) -> None:
        await self.put(ResponseVerboseChunk(content=content))

    async def mark_finished(self) -> None:
        await self.put(None)


class VirtualModelAdapter(ResponseAdapterBase):
    """Response stream adapter for virtual models.

    It reads chunks from the provided queue and forwards them to the stream.
    The queue must end with a None value to indicate the end of the stream.
    """

    def __init__(self, process_task: asyncio.Task[None], queue: ResponseQueue):
        super().__init__()

        self.process_task = process_task
        self.queue = queue

        self._stop_reason = None

    async def __aiter__(self) -> AsyncIterator[ResponseChunk]:
        """Forward chunks from the queue to the stream.

        Takes care to avoid yielding the stop reason twice.
        """

        try:
            while True:
                chunk = await self.queue.get()
                if chunk is None:  # End of stream marker
                    if self._stop_reason is None:
                        yield ResponseStopReasonChunk(reason=StopReason.END_TURN)

                    break

                if isinstance(chunk, ResponseStopReasonChunk):
                    self._stop_reason = chunk.reason

                yield chunk

        finally:
            # Ensure the process task is completed
            await self.process_task
