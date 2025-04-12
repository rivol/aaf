# AAF (Agentic AI Framework)

AAF is a versatile and extensible framework for building and managing agentic AI models.
It provides a unified interface for various language model providers and
implements advanced virtual models for complex, agent-like conversational scenarios.

Note that AAF is primarily a personal learning project focused on exploring agentic AI and LLM use,
including complex multi-step interactions.
While it can be useful for actual use cases like autonomous chat agents and multi-stage task completion,
please exercise caution when considering it for anything even remotely important.


## Features

- Support for multiple LLM providers (OpenAI, Anthropic, Ollama, LiteLLM) to act as the foundation for AI agents
- Advanced conversation management with Threads and Sessions for maintaining agent state
- Virtual models for complex, multi-step agent behaviors:
  - TwoPhase: For agents that plan before acting
  - Multiphase: For agents that can break down and tackle complex tasks
  - Router: For meta-agents that can delegate to specialized sub-agents
- Tool integration for function calling capabilities
- Cost and token usage tracking


## Installation

```bash
pip install aaf
```


## Quick Start

```python
from aaf.threads import Session

thread = Session().create_thread("gpt-4o", system="You are a helpful assistant.")
thread.add_message("user", "What is the capital of France?")

async with thread.run() as stream:
    async for chunk in stream.text_chunks():
        print(chunk.content, end="", flush=True)
    print()

print(thread.cost_and_usage().pretty())
```


## Usage

### LLM Providers

AAF supports multiple LLM providers. To use a specific provider, specify the model name when creating a thread:

```python
thread = session.create_thread("gpt-4o")  # OpenAI
thread = session.create_thread("claude-3-5-sonnet-20240620")  # Anthropic
thread = session.create_thread("llama3.1:8b")  # Ollama
```


### Virtual Models

AAF implements several virtual models for advanced use cases:

- TwoPhase: Generates a prompt and then uses it to create a response
- Multiphase: Multi-step process for complex questions, including drafting, feedback, and refinement
- Router: Selects the appropriate model based on the user's request

Using a virtual model is same as with standard models:

```python
from aaf.virtual_models.two_phase import TwoPhaseModel
from aaf.threads import Session

thread = Session().create_thread(model="two-phase", runner=TwoPhaseModel())
thread.add_message("user", "What is the capital of France?")

async with thread.run() as stream:
    async for chunk in stream.text_chunks():
        print(chunk.content, end="", flush=True)
    print()

print(thread.cost_and_usage().pretty())
```


## Project Structure

- `aaf/`: Main package directory
  - `llms/`: LLM provider implementations
  - `virtual_models/`: Virtual model implementations
  - `tools/`: Tool definitions
  - `threads.py`: Thread and Session management
  - `logging.py`: Custom logging implementation
  - `utils.py`: Utility functions


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
