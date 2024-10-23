from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

from aaf.event_handlers import use_event_handler
from aaf.threads import Thread
from aaf.ui import SimpleStreamingUserInterface


class ChatSession:
    """
    Manages an interactive chat session with an AI model.

    This class handles the user input loop, processes questions through the AI model,
    and manages the overall flow of the conversation, including handling user exit commands.

    Attributes:
        thread (Thread): The Thread object managing the conversation with the AI model.
        ui (SimpleStreamingUserInterface): The user interface for displaying output and handling input.
    """

    def __init__(self, thread: Thread, ui: Optional[SimpleStreamingUserInterface] = None):
        self.thread = thread
        self.ui = ui or SimpleStreamingUserInterface()
        self.prompt_session = PromptSession()

    async def run_loop(self, initial_question: Optional[str] = None, *, interactive: bool = True, **kwargs):
        if initial_question:
            await self._process_question(initial_question, **kwargs)

        try:
            while interactive:
                question = await self._get_user_input()
                if question is None:
                    break
                await self._process_question(question, **kwargs)

        finally:
            self.ui.info(f"Final cost: {self.thread.cost_and_usage().pretty_root()}")

    async def _get_user_input(self) -> Optional[str]:
        with patch_stdout():
            try:
                question = await self.prompt_session.prompt_async("You: ")
                if question.lower() in ("exit", "quit"):
                    return None
                return question
            except EOFError:
                # Handle Ctrl+D
                print("\nGoodbye!")
                return None

    async def _process_question(self, question: str, **kwargs):
        self.thread.add_message("user", question)
        async with self.thread.run_loop(**kwargs) as stream:
            await use_event_handler(stream, self.ui)

        self.ui.info(f"Current cost: {self.thread.cost_and_usage().pretty_root()}")
