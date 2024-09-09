# AAF

AAF is a powerful and flexible framework for building and managing conversational AI models.
It provides a unified interface for various language model providers and
implements advanced virtual models for complex conversational scenarios.


## Features

- Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
- Advanced conversation management with Threads and Sessions
- Virtual models for complex scenarios (TwoPhase, Multiphase, Router)
- Tool integration for function calling capabilities
- Cost and token usage tracking


## Installation

(Add installation instructions here)


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


## LLM Providers

AAF supports the following LLM providers:
- OpenAI
- Anthropic
- Ollama

To use a specific provider, specify the model name when creating a thread:

```python
thread = session.create_thread("gpt-4o")  # OpenAI
thread = session.create_thread("claude-3-5-sonnet-20240620")  # Anthropic
thread = session.create_thread("llama3:instruct")  # Ollama
```


## Virtual Models

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
  - `threads.py`: Thread and Session management
  - `logging.py`: Custom logging implementation
  - `utils.py`: Utility functions


## License

(Add license information)
