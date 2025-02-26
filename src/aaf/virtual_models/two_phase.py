from .. import prompts
from ..forwarding import ResponseQueue
from ..llms.types import ChatRequest
from ..logging import log
from ..threads import Session
from ..utils import extract_xml_fragment
from .base import VirtualModelBase


class TwoPhaseModel(VirtualModelBase):
    id: str = "rivo/two-phase"

    external_model_name = "sonnet"

    async def process(self, chat: ChatRequest, queue: ResponseQueue) -> None:
        session = Session()
        user_messages = [message for message in chat.messages if message["role"] == "user"]
        if len(user_messages) > 1:
            return await self.process_continuation(session, chat, queue, self.external_model_name)

        question = user_messages[0]["content"]

        prompt = await self.generate_prompt(session, queue, question)
        await queue.add_debug(f"## _TwoPhase: prompt_\n\n{prompt}\n\n---\n\n")

        await self.generate_answer(session, queue, question, prompt)

        await self.output_addendum(queue, session)

    async def generate_prompt(self, session: Session, queue: ResponseQueue, question: str) -> str:
        thread = session.create_thread(self.external_model_name, name="prompt")
        thread.add_message("system", prompts.PROMPT_ENGINEER)
        thread.add_message("user", question)

        async with self.run_progressbar(queue, message="Writing prompt"):
            async with thread.run() as stream:
                await stream.finish()

        full_response = thread.messages[-1]["content"]
        log.debug("TwoPhaseModel.generate_prompt(): Full response", response=full_response)

        result = extract_xml_fragment(full_response, "system_prompt")
        log.info("TwoPhaseModel.generate_prompt() done", result_len=len(result), cost=thread.cost_and_usage().as_log())
        return result

    async def generate_answer(self, session: Session, queue: ResponseQueue, question: str, prompt: str) -> None:
        thread = session.create_thread(self.external_model_name, name="answer")
        thread.add_message("system", prompt)
        thread.add_message("user", question)

        async with thread.run() as stream:
            await stream.text_chunks().redirect(queue)

        result = thread.messages[-1]["content"]
        log.info("TwoPhaseModel.generate_answer() done", result_len=len(result), cost=thread.cost_and_usage().as_log())
        return result


class TwoPhaseGPTModel(TwoPhaseModel):
    id: str = "rivo/two-phase-gpt"

    external_model_name = "gpt-4o"


class TwoPhaseChatGPTModel(TwoPhaseModel):
    id: str = "rivo/two-phase-chatgpt"

    external_model_name = "chatgpt-4o-latest"
