import asyncio
from datetime import datetime
from typing import Annotated, Optional

import typer

from aaf import logging
from aaf.chat import ChatSession
from aaf.threads import Session

DEFAULT_MODEL = "gpt-mini"
DEFAULT_TEMPERATURE = 1.0
SYSTEM_PROMPT = """
You are a helpful assistant.

{output_format}

Current timestamp is {timestamp}
"""
SYSTEM_PROMPT_REFLECTION = """
When answering user's requests, first start with an internal monologue to gather your thoughts
and plan the answer using CoT (chain-of-thought) techniques.
Then, reflect on your thoughts, critically reviewing and validating them.
If you notice and contradictions, mistakes, or false information, correct them.
Finally, give the user-visible response.

Use the following response format:
<example_response>
<thinking>[internal thoughts for answering user's request]</thinking>
<reflections>[review and validation of your thought process]</reflections>
<response>[final user-visible response]</response>
</example_response>
"""


async def amain(
    model_name: str,
    question: Optional[str] = None,
    interactive: bool = True,
    reflect: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
):
    thread = Session().create_thread(model_name)

    system_prompt = SYSTEM_PROMPT.format(
        timestamp=datetime.now().isoformat(),
        output_format=SYSTEM_PROMPT_REFLECTION if reflect else "",
    )
    thread.add_message("system", system_prompt)

    chat_session = ChatSession(thread)
    print(f"Chatting with {thread.model}\n")
    await chat_session.run_loop(question, interactive=interactive, temperature=temperature)


def main(
    question: Annotated[Optional[str], typer.Argument(help="Initial question to start the chat")] = None,
    model: Annotated[str, typer.Option("--model", "-m")] = DEFAULT_MODEL,
    non_interactive: Annotated[
        bool,
        typer.Option("--non-interactive", "-n", help="Run in non-interactive mode"),
    ] = False,
    reflect: Annotated[
        bool,
        typer.Option("--reflect", "-r", help="Use reflection process when answering"),
    ] = False,
    temperature: Annotated[float, typer.Option("--temperature", "-t")] = DEFAULT_TEMPERATURE,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    debug: Annotated[bool, typer.Option("--debug", "-d")] = False,
):
    logging.set_level_from_flags(quiet=quiet, debug=debug)

    if non_interactive and not question:
        typer.echo("Error: In non-interactive mode, an initial question is required.", err=True)
        raise typer.Exit(code=1)

    asyncio.run(
        amain(
            model_name=model,
            question=question,
            interactive=not non_interactive,
            reflect=reflect,
            temperature=temperature,
        )
    )


if __name__ == "__main__":
    typer.run(main)
