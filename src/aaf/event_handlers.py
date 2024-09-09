from .llms.types import (
    ResponseChunkToolCallFailed,
    ResponseChunkToolCallFinished,
    ResponseChunkToolCallStarted,
    ResponseControlChunkStreamBegin,
    ResponseControlChunkStreamEnd,
    ResponseStream,
    ResponseTextChunk,
    ToolCall,
)


class EventHandlerBase:
    """Base class for event handlers that process messages and events from the assistant and tools.

    This can be used to implement User Interface classes, custom logic, etc.
    """

    def info(self, message: str):
        """Print an informational message."""
        pass

    def debug(self, message: str):
        """Print a debug message."""
        pass

    def assistant_message_stream_start(self, stream: ResponseStream):
        """Called when the assistant starts sending a message stream."""
        pass

    def assistant_message_stream_chunk(self, stream: ResponseStream, chunk: ResponseTextChunk):
        """Called for each chunk of the assistant's response."""
        pass

    def assistant_message_stream_end(self, stream: ResponseStream):
        """Called when the assistant finishes sending a message stream."""
        pass

    def tool_call_start(self, tool_call: ToolCall):
        """Called when a tool call is about to be executed."""
        pass

    def tool_call_result(self, tool_call: ToolCall, result: str):
        """Called when a tool call has been executed successfully."""
        pass

    def tool_call_error(self, tool_call: ToolCall, error: str):
        """Called when a tool call has failed."""
        pass

    def loop_end(self, cost: float):
        pass


async def use_event_handler(stream: ResponseStream, handler: EventHandlerBase):
    """Uses the event handler for the stream's messages.

    This is basically an adapter that directs the stream's messages to the event handler.
    """

    async for chunk in stream.all_chunks():
        if isinstance(chunk, ResponseControlChunkStreamBegin):
            handler.assistant_message_stream_start(stream)
        elif isinstance(chunk, ResponseControlChunkStreamEnd):
            handler.assistant_message_stream_end(stream)
        elif isinstance(chunk, ResponseTextChunk):
            handler.assistant_message_stream_chunk(stream, chunk)
        elif isinstance(chunk, ResponseChunkToolCallStarted):
            handler.tool_call_start(chunk.tool_call)
        elif isinstance(chunk, ResponseChunkToolCallFinished):
            handler.tool_call_result(chunk.tool_call, chunk.result)
        elif isinstance(chunk, ResponseChunkToolCallFailed):
            handler.tool_call_error(chunk.tool_call, chunk.error)
