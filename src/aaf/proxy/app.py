import time
from typing import Annotated, AsyncIterable

from fastapi import Depends, FastAPI, HTTPException, Request
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice as FullResponseChoice
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from sse_starlette import EventSourceResponse

from ..forwarding import ResponseQueue
from ..llms.types import ChatRequest
from ..logging import log
from ..threads import Session
from ..virtual_models.base import VirtualModelBase
from ..virtual_models.demos import DemoToolUsageModel, MinimalVirtualModel, SimpleVirtualModel
from ..virtual_models.multiphase import MultiphaseChatGPTModel, MultiphaseGPTModel, MultiphaseModel
from ..virtual_models.router import RouterGPTModel, RouterModel
from ..virtual_models.two_phase import TwoPhaseChatGPTModel, TwoPhaseGPTModel, TwoPhaseModel
from ..virtual_models.types import ModelCard, ModelList


class APIChatRequest(ChatRequest):
    model: str


async def process_wrapper(model: VirtualModelBase, chat: ChatRequest, queue: ResponseQueue) -> None:
    try:
        await model.process(chat, queue)
    finally:
        await queue.put(None)


async def chat_completion_streamed(model: VirtualModelBase, chat: ChatRequest) -> AsyncIterable[ChatCompletionChunk]:
    user_messages = [message for message in chat.messages if message["role"] == "user"]
    log.info(
        "app.chat_completion_streamed: starting",
        model=model.id,
        messages_count=len(chat.messages),
        user_messages_count=len(user_messages),
    )

    chat_id = "static"
    chunk_template = ChatCompletionChunk(
        id=chat_id,
        choices=[],
        created=int(time.time()),
        model=model.id,
        object="chat.completion.chunk",
    )

    yield chunk_template.model_copy(
        update={
            "choices": [Choice(delta=ChoiceDelta(content="", role="assistant"), index=0)],
        }
    )

    thread = Session().create_thread(model.id, chat, runner=model)
    async with thread.run_loop() as stream:
        async for chunk in stream.text_chunks():
            yield chunk_template.model_copy(
                update={
                    "choices": [Choice(delta=ChoiceDelta(content=chunk.content), index=0)],
                }
            )

    yield chunk_template.model_copy(
        update={
            "choices": [Choice(delta=ChoiceDelta(), finish_reason="stop", index=0)],
        }
    )
    yield chunk_template.model_copy(
        update={
            "choices": [],
            "usage": CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        }
    )

    log.info("app.chat_completion_streamed: done", model=model.id, cost=thread.cost_and_usage().pretty_root())


async def generate_chat_title(chat: ChatRequest, model_name: str = "gpt-4o") -> ChatCompletion:
    thread = Session().create_thread(model_name, ChatRequest(messages=chat.messages))
    async with thread.run() as stream:
        await stream.finish()

    title = stream.text
    cost_and_usage = thread.cost_and_usage()
    log.info("generate_chat_title()", title=title, usage=cost_and_usage.pretty_root())

    return ChatCompletion(
        id="static",
        choices=[
            FullResponseChoice(
                finish_reason="stop", index=0, message=ChatCompletionMessage(content=title, role="assistant")
            )
        ],
        created=int(time.time()),
        model=model_name,
        object="chat.completion",
        usage=CompletionUsage(
            prompt_tokens=cost_and_usage.prompt_tokens,
            completion_tokens=cost_and_usage.completion_tokens,
            total_tokens=cost_and_usage.total_tokens,
        ),
    )


def app_models(request: Request) -> list[VirtualModelBase]:
    """Dependency that provides access to configured models from app state."""
    return request.app.state.models


def get_models_list(models: Annotated[list[VirtualModelBase], Depends(app_models)]) -> ModelList:
    """List available models."""
    return ModelList(data=[ModelCard(id=model.id, name=model.display_name, owned_by="Rivo") for model in models])


async def post_chat_completions(
    request: Request, chat: APIChatRequest, models: Annotated[list[VirtualModelBase], Depends(app_models)]
) -> EventSourceResponse | ChatCompletion:
    """Handle chat completion requests."""
    print(f"request: {request.method} {request.url.path}")

    model = next((m for m in models if m.id == chat.model), None)
    if not model:
        raise HTTPException(status_code=404, detail="Unknown model")

    chat = ChatRequest(**chat.model_dump(exclude={"model"}))
    if not chat.stream:
        # Most likely WebUI request to generate chat title
        if "RESPOND ONLY WITH THE TITLE TEXT." in chat.messages[0]["content"]:
            return await generate_chat_title(chat)

        raise HTTPException(status_code=400, detail="Only streaming is supported")

    async def wrapper():
        async for chunk in chat_completion_streamed(model, chat):
            yield dict(data=chunk.model_dump_json())

    return EventSourceResponse(wrapper())


def setup_app(models: list[VirtualModelBase]) -> FastAPI:
    """Create and configure a FastAPI application with the given models."""
    app = FastAPI()

    # Store models in app state
    app.state.models = models

    # Add routes
    app.add_api_route("/v1/models", get_models_list, methods=["GET"])
    app.add_api_route(
        "/v1/chat/completions",
        post_chat_completions,
        methods=["POST"],
        response_model=None,
    )

    return app


# Create default instance with demo models
default_models = [
    MinimalVirtualModel(),
    SimpleVirtualModel(),
    MultiphaseModel(),
    MultiphaseChatGPTModel(),
    MultiphaseGPTModel(),
    TwoPhaseModel(),
    TwoPhaseChatGPTModel(),
    TwoPhaseGPTModel(),
    RouterModel(),
    RouterGPTModel(),
    DemoToolUsageModel(),
]

app = setup_app(default_models)
