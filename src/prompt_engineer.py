import asyncio
from typing import Annotated

import typer

from aaf import logging, prompts
from aaf.event_handlers import use_event_handler
from aaf.threads import Session
from aaf.ui import SimpleStreamingUserInterface

DEFAULT_MODEL = "sonnet"


async def prompt_engineer(question: str, model_name: str):
    ui = SimpleStreamingUserInterface()
    thread = Session().create_thread(model_name)

    thread.add_message("system", prompts.PROMPT_ENGINEER)
    thread.add_message("user", question)
    async with thread.run_loop() as stream:
        await use_event_handler(stream, ui)

    ui.info(f"Final cost: USD {thread.cost_and_usage().cost:.4f}")


def cli(
    question: str,
    model: Annotated[str, typer.Option("--model", "-m")] = DEFAULT_MODEL,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    debug: Annotated[bool, typer.Option("--debug", "-d")] = False,
):
    logging.set_level_from_flags(quiet=quiet, debug=debug)

    asyncio.run(prompt_engineer(question, model_name=model))


if __name__ == "__main__":
    typer.run(cli)
