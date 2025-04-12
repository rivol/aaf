import asyncio
from typing import Annotated, Callable

import rich
import typer

from aaf import logging
from aaf.event_handlers import use_event_handler
from aaf.llms import AnthropicRunner
from aaf.threads import Session
from aaf.tools import demo
from aaf.ui import SimpleStreamingUserInterface

DEFAULT_MODEL = "sonnet"

app = typer.Typer()
selected_model = DEFAULT_MODEL


@app.callback()
def main(
    model: Annotated[str, typer.Option("--model", "-m")] = DEFAULT_MODEL,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    debug: Annotated[bool, typer.Option("--debug", "-d")] = False,
):
    global selected_model

    logging.set_level_from_flags(quiet=quiet, debug=debug)
    selected_model = model


async def run_question(question: str):
    thread = Session().create_thread(model=selected_model)
    thread.add_message("user", question)

    async with thread.run() as stream:
        async for chunk in stream.text_chunks():
            print(chunk.content, end="", flush=True)
        print()

    print(thread.cost_and_usage().pretty())


@app.command()
def question(question: str = "What's the capital of France?"):
    asyncio.run(run_question(question))


async def run_question_loop(question: str):
    thread = Session().create_thread(model=selected_model)
    thread.add_message("user", question)

    async with thread.run_loop() as stream:
        async for chunk in stream.text_chunks():
            print(chunk.content, end="", flush=True)
        print()

    print(thread.cost_and_usage().pretty())


@app.command()
def question_loop(question: str = "What's the capital of France?"):
    asyncio.run(run_question_loop(question))


async def run_pirate_multistep():
    system = (
        "Talk like a pirate. Answer briefly, without reflecting the question. Use tools only if explicitly instructed"
    )
    ui = SimpleStreamingUserInterface()
    thread = Session().create_thread(selected_model, system=system, tools=[demo.text_complexity])

    thread.add_message("user", "What is the meaning of life, in one short sentence?")
    async with thread.run(name="first_question") as stream:
        await use_event_handler(stream, ui)

    thread.add_message("user", "Print the previous answer in ALL CAPS")
    async with thread.run_loop() as stream:
        await use_event_handler(stream, ui)

    thread.add_message("user", "Calculate the text complexity of the previous answer. Use tools provided")
    async with thread.run_loop() as stream:
        await use_event_handler(stream, ui)

    print(thread.cost_and_usage().pretty())


@app.command()
def pirate_multistep():
    asyncio.run(run_pirate_multistep())


async def run_weather():
    tools: list[Callable] = [demo.get_weather_at, demo.get_location_coordinates]

    ui = SimpleStreamingUserInterface()
    thread = Session().create_thread(selected_model, tools=tools)

    thread.add_message("user", "What's the current weather in Tallinn, New York City, and Paris? ")
    async with thread.run_loop() as stream:
        await use_event_handler(stream, ui)

    thread.add_message("user", "Briefly (no more than 3 lines) aggregate results across locations.")
    async with thread.run_loop() as stream:
        await use_event_handler(stream, ui)

    ui.info(f"Final cost: USD {thread.cost_and_usage().cost:.4f}")


@app.command()
def weather():
    asyncio.run(run_weather())


async def run_rate_limits():
    model = "claude-3-5-sonnet-20240620"
    messages = [
        {
            "role": "user",
            "content": "hi",
        },
    ]

    runner = AnthropicRunner()
    response = await runner.client.messages.with_raw_response.create(
        model=model,
        messages=messages,
        max_tokens=1,
    )

    for k, v in response.headers.items():
        if k.startswith("anthropic-ratelimit-"):
            print(f"{k}: {v}")


@app.command()
def rate_limits():
    asyncio.run(run_rate_limits())


@app.command()
def all():
    global selected_model

    commands = [
        question,
        question_loop,
        weather,
        pirate_multistep,
    ]
    models = [
        "gpt-4o-mini",
        "sonnet",
    ]

    for command in commands:
        for model in models:
            print()
            print()
            rich.print(f"[bright_yellow]RUNNING: {command.__name__} with {model}[/bright_yellow]")

            selected_model = model
            command()


async def run_litellm_test():
    """Test the LiteLLM provider with a simple question."""
    print("\nTesting LiteLLM provider...")

    # Demo with the LiteLLM prefix format
    thread = Session().create_thread(model="litellm/gpt-3.5-turbo", system="You are a helpful assistant.")
    thread.add_message("user", "What's the capital of France?")

    print("\nUsing model: litellm/gpt-3.5-turbo")
    print("Response: ", end="")
    async with thread.run() as stream:
        async for chunk in stream.text_chunks():
            print(chunk.content, end="", flush=True)
        print()

    print(thread.cost_and_usage().pretty())


@app.command()
def litellm_test():
    """Test the LiteLLM provider."""
    asyncio.run(run_litellm_test())


if __name__ == "__main__":
    app()
