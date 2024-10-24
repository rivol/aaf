import asyncio
from typing import Callable

from ..forwarding import ResponseQueue
from ..llms.types import ChatRequest
from ..threads import Session
from ..tools import demo
from .base import VirtualModelBase


class MinimalVirtualModel(VirtualModelBase):
    id: str = "rivo/minimal-proxy"
    name: str = "Minimal Demo Model"

    async def process(self, chat: ChatRequest, queue: ResponseQueue) -> None:
        await asyncio.sleep(0.1)
        await queue.add("hello ")
        await asyncio.sleep(0.1)
        await queue.add("there!")
        await asyncio.sleep(1)
        await queue.add(" I am ")
        await asyncio.sleep(0.1)
        await queue.add(self.id)
        await asyncio.sleep(0.1)
        await queue.add("!")
        await queue.add("\n")
        await queue.add(
            """<details open>
            <summary>Hello</summary>
            World!
            </details>"""
        )


class SimpleVirtualModel(VirtualModelBase):
    id: str = "rivo/simple-proxy"

    async def process(self, chat: ChatRequest, queue: ResponseQueue) -> None:
        thread = Session().create_thread("gpt-4o", system="Pretend to be a pirate")
        for message in chat.messages:
            thread.add_message(**message)

        async with thread.run() as stream:
            await stream.text_chunks().redirect(queue)

        await self.output_addendum(queue, thread)


class DemoToolUsageModel(VirtualModelBase):
    id: str = "rivo/demo-tools"

    async def process(self, chat: ChatRequest, queue: ResponseQueue) -> None:
        tools: list[Callable] = [demo.get_weather_at, demo.get_location_coordinates]
        thread = Session().create_thread("gpt-4o", tools=tools)

        for message in chat.messages:
            thread.add_message(**message)

        async with thread.run_loop() as stream:
            await stream.text_chunks().redirect(queue)

        await self.output_addendum(queue, thread)
