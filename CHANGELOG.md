# 0.3.6 (2025-04-20)

- Add new OpenAI models - `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `o3`, `o4-mini`, plus aliases.
- `gpt`, `gpt4`, and `gpt-mini` aliases now point to GPT-4.1 family.


# 0.3.5 (2025-04-12)

- Add LiteLLM support for accessing various LLM providers.
- `gemini/gemini-2.0-flash` and `gemini/gemini-2.5-pro-preview-03-25` are available via LiteLLM.
- Add support for image inputs in AnthropicRunner by transforming from OpenAI format to Anthropic format.
- VirtualModelBase.process_continuation() now accepts **kwargs, e.g. for custom temperature.


# 0.3.4 (2025-02-26)

- Add Claude 3.7 Sonnet (`claude-3-7-sonnet-20250219`). `sonnet`, `sonnet-3.7` aliases now point to this model.
- Upgrade dependencies, and loosen version requirements.
- Make max_tokens configurable.


# 0.3.3 (2025-01-25)

- Add `gpt-4o-2024-11-20` model. `gpt-4o`, `4o`, `gpt4`, `gpt` aliases now point to this model.


# 0.3.2 (2024-10-25)

- Proxy API app now supports custom virtual models
- Return model name in API responses


# 0.3.1 (2024-10-23)

- Add interactive chat script
- OpenRouter runner, with OpenAI's o1 and Llama 3.1 models
- Support new Sonnet 3.5 model (`claude-3-5-sonnet-20241022`)
- Temperature and other parameters can be passed to ChatSession.run_loop() and Thread.run_loop()
- Improve README; update dependencies; add black and isort


# 0.3.0 (2024-09-09)

- Initial release.
