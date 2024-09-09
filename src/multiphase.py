import asyncio
from typing import Annotated

import typer

from aaf import logging
from aaf.threads import Session
from aaf.virtual_models.multiphase import MultiphaseModel

DEFAULT_MODEL = MultiphaseModel.external_model_name


async def multiphase(model_name: str, question: str) -> None:
    runner = MultiphaseModel(model_name)
    # runner = SimpleVirtualModel()
    thread = Session().create_thread(model=runner.id, runner=runner)
    thread.add_message("user", question)

    async with thread.run() as stream:
        async for chunk in stream.text_chunks():
            print(chunk.content, end="", flush=True)
        print()

    print(thread.cost_and_usage().pretty())


async def main(model: str, question: str):
    await multiphase(model_name=model, question=question)


def main_sync(
    question: str,
    model: Annotated[str, typer.Option("--model", "-m")] = DEFAULT_MODEL,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    debug: Annotated[bool, typer.Option("--debug", "-d")] = False,
):
    logging.set_level_from_flags(quiet=quiet, debug=debug)

    asyncio.run(main(model=model, question=question))


if __name__ == "__main__":
    typer.run(main_sync)
