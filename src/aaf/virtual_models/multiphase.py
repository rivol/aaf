from typing import Optional

from .. import prompts
from ..forwarding import ResponseQueue
from ..llms.types import ChatRequest
from ..logging import log
from ..prompts import MULTIPHASE_FEEDBACK, MULTIPHASE_FINAL
from ..threads import Session
from ..utils import extract_xml_fragment
from .base import VirtualModelBase


class MultiphaseModel(VirtualModelBase):
    id: str = "rivo/multiphase"

    external_model_name = "sonnet"

    def __init__(self, external_model_name: Optional[str] = None):
        super().__init__()

        if external_model_name is not None:
            self.external_model_name = external_model_name

    async def process(self, chat: ChatRequest, queue: ResponseQueue) -> None:
        session = Session()

        user_messages = [message for message in chat.messages if message["role"] == "user"]
        if len(user_messages) > 1:
            return await self.process_continuation(session, chat, queue, self.external_model_name)

        question = user_messages[0]["content"]

        prompt = await self.generate_prompt(session, queue, question)
        await queue.add_debug(f"## _Multiphase: prompt_\n\n{prompt}\n\n---\n\n")

        draft = await self.phase_one(session, queue, prompt, question)
        await queue.add_debug(f"## _Multiphase: draft answer_\n\n{draft}\n\n---\n\n")

        feedback = await self.phase_two(session, queue, question, draft)
        await queue.add_debug(f"## _Multiphase: feedback_\n\n{feedback}\n\n---\n\n")

        answer = await self.phase_three(session, queue, prompt, question, draft, feedback)

        await self.output_addendum(queue, session)

    async def generate_prompt(self, session: Session, queue: ResponseQueue, question: str) -> str:
        thread = session.create_thread(self.external_model_name, name="prompt")
        thread.add_message("system", prompts.PROMPT_ENGINEER)
        thread.add_message("user", question)

        async with self.run_progressbar(queue, message="Writing prompt"):
            async with thread.run() as stream:
                await stream.finish()

        full_response = thread.messages[-1]["content"]
        log.debug("Multiphase.generate_prompt(): Full response", response=full_response)

        result = extract_xml_fragment(full_response, "system_prompt")
        log.info(
            "MultiphaseModel.generate_prompt() done", result_len=len(result), cost=thread.cost_and_usage().as_log()
        )
        return result

    async def phase_one(self, session: Session, queue: ResponseQueue, prompt: str, question: str) -> str:
        """Asks the user's question and returns initial answer."""

        temperature = 0.7

        thread = session.create_thread(self.external_model_name, name="draft")
        thread.add_message("system", prompt)
        thread.add_message("user", question)

        async with self.run_progressbar(queue, message="Drafting initial answer"):
            async with thread.run(temperature=temperature) as stream:
                await stream.finish()

        result = thread.messages[-1]["content"]
        log.info("MultiphaseModel.phase_one() done", result_len=len(result), cost=thread.cost_and_usage().as_log())
        return result

    async def phase_two(self, session: Session, queue: ResponseQueue, question: str, initial_answer: str) -> str:
        """Writes feedback for initial answer."""

        temperature = 1.0

        thread = session.create_thread(self.external_model_name, name="feedback")
        thread.add_message("system", MULTIPHASE_FEEDBACK)
        thread.add_message(
            "user",
            f"""
            <question>\n{question}\n</question>\n
            <answer>\n{initial_answer}\n</answer>\n
        """,
        )
        async with self.run_progressbar(queue, message="Looking for ways to improve"):
            async with thread.run(temperature=temperature) as stream:
                await stream.finish()

        result = thread.messages[-1]["content"]
        log.info("MultiphaseModel.phase_two() done", result_len=len(result), cost=thread.cost_and_usage().as_log())
        return result

    async def phase_three(
        self, session: Session, queue: ResponseQueue, prompt: str, question: str, initial_answer: str, feedback: str
    ) -> str:
        """Composes final answer based on initial answer and feedback."""

        temperature = 0.7

        thread = session.create_thread(self.external_model_name, name="answer")
        thread.add_message("system", prompt + MULTIPHASE_FINAL)
        thread.add_message(
            "user",
            f"""
            <question>\n{question}\n</question>\n
            <answer>\n{initial_answer}\n</answer>\n
            <feedback>\n{feedback}\n</feedback>\n
            """,
        )

        async with thread.run(temperature=temperature) as stream:
            # Display progress bar until the first "-----" is encountered, marking the start of the final answer
            end_of_thinking_seen = False
            async with self.run_progressbar(queue, message="Composing final answer"):
                text = ""
                async for chunk in stream.text_chunks():
                    text += chunk.content

                    if "</thinking>" in text:
                        end_of_thinking_seen = True
                        break

            # Display the final answer
            await queue.add("\n")
            async for chunk in stream.text_chunks():
                await queue.put(chunk)

            # Failsafe, in case the model forgot the </thinking> tag. Just show the full response
            if not end_of_thinking_seen:
                log.info("Multiphase: End of thinking not seen, showing full response")
                await queue.add(text)

        result = thread.messages[-1]["content"]
        log.info("MultiphaseModel.phase_three() done", result_len=len(result), cost=thread.cost_and_usage().as_log())
        return result


class MultiphaseGPTModel(MultiphaseModel):
    id: str = "rivo/multiphase-gpt"

    external_model_name = "gpt-4o"


class MultiphaseChatGPTModel(MultiphaseModel):
    id: str = "rivo/multiphase-chatgpt"

    external_model_name = "chatgpt-4o-latest"
