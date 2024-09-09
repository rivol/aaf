import rich
from rich.text import Text

from .event_handlers import EventHandlerBase
from .llms.types import ResponseStream, ResponseTextChunk, ToolCall
from .utils import truncate_text


class SimpleStreamingUserInterface(EventHandlerBase):
    """Simple UI that shows streamed responses and some status info."""

    stream_responses = True

    def info(self, message: str):
        rich.print(f"[bold]{message}[/bold]")

    def debug(self, message: str):
        print(f"{message}")

    def assistant_message_stream_start(self, stream: ResponseStream):
        pass

    def assistant_message_stream_chunk(self, stream: ResponseStream, chunk: ResponseTextChunk):
        if self.stream_responses:
            text = Text(chunk.content, style="bold yellow")
            rich.print(text, end="")

    def assistant_message_stream_end(self, stream: ResponseStream):
        if self.stream_responses:
            rich.print()
        else:
            text = Text(stream.text, style="bold yellow")
            rich.print(text)

    def tool_call_start(self, tool_call: ToolCall):
        rich.print(f"Starting tool call: [bold]{tool_call.pretty()}[/bold]")

    def tool_call_result(self, tool_call: ToolCall, result: str):
        print(f"    result: {truncate_text(str(result), 200)}")

    def tool_call_error(self, tool_call: ToolCall, error: str):
        rich.print(f"    [bold red]error: {error}[/bold red]")
