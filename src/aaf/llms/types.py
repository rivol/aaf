import json
from asyncio import Queue
from enum import Enum
from typing import Any, AsyncIterator, Callable, Optional

from pydantic import BaseModel

from ..logging import log


class CompletionUsage(BaseModel):
    completion_tokens: int = 0
    """Number of tokens in the generated completion."""

    prompt_tokens: int = 0
    """Number of tokens in the prompt."""

    @property
    def total_tokens(self):
        """Total number of tokens used in the request (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "CompletionUsage") -> "CompletionUsage":
        return CompletionUsage(
            completion_tokens=self.completion_tokens + other.completion_tokens,
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
        )


class CostAndUsage(CompletionUsage):
    cost: float = 0
    name: str = "_unknown_"

    children: list["CostAndUsage"] = []

    @classmethod
    def aggregate(cls, name: str, children: list["CostAndUsage"]) -> "CostAndUsage":
        total = cls(name=name, children=children)
        for child in children:
            total.completion_tokens += child.completion_tokens
            total.prompt_tokens += child.prompt_tokens
            total.cost += child.cost

        return total

    def pretty_root(self) -> str:
        return f"{self.name}: {self.cost:.4f} USD  ({self.prompt_tokens} + {self.completion_tokens} tokens)"

    def pretty(self, *, level: int = 0) -> str:
        result = self.pretty_root()
        if level:
            result = f"{(level - 1) * '    '}- {result}"

        for child in self.children:
            result += "\n" + child.pretty(level=level + 1)
        return result

    def as_log(self) -> str:
        return f"{self.cost:.4f} USD; {self.prompt_tokens} + {self.completion_tokens} tokens"


class ToolCall(BaseModel):
    id: str

    name: str
    """The name of the function to call."""

    arguments: str
    """
    The arguments to call the function with, as generated by the model in JSON
    format. Note that the model does not always generate valid JSON, and may
    hallucinate parameters not defined by your function schema. Validate the
    arguments in your code before calling your function.
    """

    @property
    def arguments_dict(self) -> dict[str, Any]:
        return json.loads(self.arguments)

    def pretty(self) -> str:
        return f"{self.name}({', '.join(f'{k}={repr(v)}' for k, v in self.arguments_dict.items())})"


class ToolCallResult(BaseModel):
    tool_call: ToolCall
    is_error: bool = False
    content: Optional[str] = None


# TODO: use StrEnum when we upgrade to Python 3.11
class StopReason(Enum):
    END_TURN = "end_turn"
    """The conversation has ended."""

    TOOL_USE = "tool_use"
    """A tool was used."""

    LENGTH = "length"
    """The maximum length was reached."""


class ResponseChunk(BaseModel):
    pass


class ResponseBaseTextChunk(ResponseChunk):
    content: str


class ResponseTextChunk(ResponseBaseTextChunk):
    pass


class ResponseVerboseChunk(ResponseBaseTextChunk):
    pass


class ResponseDebugChunk(ResponseBaseTextChunk):
    pass


class ResponseChunkToolCall(ResponseChunk):
    tool_call: ToolCall


class ResponseChunkToolCallStarted(ResponseChunk):
    tool_call: ToolCall


class ResponseChunkToolCallFinished(ResponseChunk):
    tool_call: ToolCall
    result: Any


class ResponseChunkToolCallFailed(ResponseChunk):
    tool_call: ToolCall
    error: str


class ResponseUsageChunk(ResponseChunk):
    delta: CompletionUsage


class ResponseStopReasonChunk(ResponseChunk):
    reason: StopReason


class ResponseChunkCompleteText(ResponseChunk):
    content: str


class ResponseControlChunk(BaseModel):
    pass


class ResponseControlChunkStreamBegin(BaseModel):
    pass


class ResponseControlChunkStreamEnd(BaseModel):
    pass


class ResponseControlChunkRateLimited(BaseModel):
    delay_secs: float
    metadata: dict


class ResponseChunksStream:

    def __init__(self, stream: "ResponseStream", filter: Callable[[ResponseChunk], bool]):
        self.stream = stream
        self.filter = filter

    async def __aiter__(self) -> AsyncIterator[ResponseChunk]:
        async for chunk in self.stream:
            if self.filter(chunk):
                yield chunk

    async def redirect(self, queue: Queue) -> None:
        async for chunk in self:
            await queue.put(chunk)


class ResponseStream:
    """Represents a stream of response chunks from model.

    Stream contains either "physical" response (single model completion, from an API call) or a "logical" response
    (chain of responses, e.g. with tool calls).

    It also has a few helper methods.
    """

    async def __aiter__(self) -> AsyncIterator[ResponseChunk]:
        raise NotImplementedError()

    def all_chunks(self) -> ResponseChunksStream:
        return ResponseChunksStream(self, lambda chunk: True)

    def text_chunks(self) -> ResponseChunksStream:
        return ResponseChunksStream(self, lambda chunk: isinstance(chunk, ResponseTextChunk))

    async def redirect(self, queue: Queue) -> None:
        await self.all_chunks().redirect(queue)

    async def finish(self) -> None:
        async for _ in self:
            pass

    async def put(self, chunk: ResponseChunk) -> None:
        raise NotImplementedError()

    @property
    def text(self) -> str:
        raise NotImplementedError()

    @property
    def usage(self) -> CompletionUsage:
        raise NotImplementedError()


class ModelCost(BaseModel):
    prompt_per_1m: float = 0
    completion_per_1m: float = 0


class ModelInfo(BaseModel):
    name: str
    aliases: list[str] = []
    cost: ModelCost = ModelCost()


class ChatRequest(BaseModel):
    messages: list[dict] = []
    stream: bool = True
    max_tokens: Optional[int] = None
    tools: list[Callable] = []
    tool_choice: str | dict = "auto"