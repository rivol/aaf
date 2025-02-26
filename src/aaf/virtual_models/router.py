from ..forwarding import ResponseQueue
from ..llms.types import ChatRequest, ResponseControlChunkRateLimited
from ..logging import log
from ..threads import Session
from ..utils import extract_xml_fragment
from .base import VirtualModelBase

ROUTER_PROMPT = """
You will receive user's request and must categorize it, selecting which model to send it to.
Do NOT attempt to fulfill the request or answer it.

Here are the models available:
- simple: this should be used for brief or simple requests.
- pirate: use this if user mentions wanting the response to be in pirate speak, or if the request is strongly related to pirates.
- two-phase: use this for more complex requests, that need deeper thinking and advanced models.
- multiphase: this is for the most complex, research-like tasks, where the answer should be long and comprehensive.

Begin your response with your internal thought process, enclosed in <thinking>...</thinking> tags.
This must be followed by <model>[model name]</model> where [model name] is one of the models listed above.

Here are some examples:
<example>
<question>What is the capital of France?</question>
<response>
<thinking>This is a simple request, the answer is factual and user did not request elaboration.</thinking>
<model>simple</model>
</response>
</example>

<example>
<question>What are the districts of the capital of France, and which one would you recommend for a visit?</question>
<response>
<thinking>This is a more complex request, requiring detailed information and analysis.</thinking>
<model>two-phase</model>
</response>
</example>

<example>
<question>What are the main causes of climate change and how can they be mitigated?</question>
<response>
<thinking>This is a complex, research-like question that requires a comprehensive answer.</thinking>
<model>multiphase</model>
</response>
</example>
"""


class RouterModel(VirtualModelBase):
    id: str = "rivo/router"

    external_model_name = "sonnet"

    async def process(self, chat: ChatRequest, queue: ResponseQueue) -> None:
        session = Session()

        user_messages = [message for message in chat.messages if message["role"] == "user"]
        if len(user_messages) > 1:
            return await self.process_continuation(session, chat, queue, self.external_model_name)

        question = user_messages[0]["content"]

        model_type = await self.select_model(session, queue, question)
        await queue.add(f"_Router: model: {model_type}_\n\n")

        # TODO: call the selected model

        await self.output_addendum(queue, session)

    async def select_model(self, session: Session, queue: ResponseQueue, question: str) -> str:
        thread = session.create_thread(self.external_model_name, name="model selection")
        thread.add_message("system", ROUTER_PROMPT)
        thread.add_message("user", question)

        async with self.run_progressbar(queue, message="Choosing model"):
            async with thread.run() as stream:
                async for chunk in stream:
                    if isinstance(chunk, ResponseControlChunkRateLimited):
                        await queue.add(" _(rate limited)_ ")
                await stream.finish()

        full_response = thread.messages[-1]["content"]
        log.info(f"Router: model selection response:\n{full_response}")

        return extract_xml_fragment(full_response, "model")


class RouterGPTModel(RouterModel):
    id: str = "rivo/router-gpt"

    external_model_name = "gpt-4o"
