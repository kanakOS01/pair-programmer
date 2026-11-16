# AGENTS.md

Context for coding agents that work on this repository.

`pp` (Pair Programmer) is a terminal coding agent. A user talks to a model. The model reads and changes the user's project through tools.

Read `CONTEXT.md` for the domain language (conversation history, tool call, tool result, compaction, approval policy, provider). Use those terms exactly in code, comments and commit messages.

## Commands

The project uses `uv` and Python 3.13.

| Task | Command |
| --- | --- |
| Install dependencies | `uv sync` |
| Run pp | `uv run pp` (interactive) or `uv run pp "prompt"` (one turn) |
| Run tests | `uv run pytest -q` |
| Run one test | `uv run pytest tests/test_tools.py::test_name -q` |
| Type check | `uv run mypy` |
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| All pre-commit hooks | `uv run pre-commit run --all-files` |

Before you finish a change, run `ruff check`, `ruff format --check`, `mypy` and `pytest`. All four must pass.

### Tool configuration

- **ruff**: line length 130, target `py313`. Rules `E`, `F`, `I`, `B`. See `[tool.ruff.lint] ignore` in `pyproject.toml` for the ignored rules.
- **mypy**: checks `pp/` only, with the `pydantic.mypy` plugin. Tests are not type-checked.
- **pytest**: `asyncio_mode = "auto"`. Write `async def test_...` with no decorator.
- **pre-commit**: runs ruff, ruff-format, trailing-whitespace, end-of-file-fixer, mypy and pytest on each commit.

## Architecture

```text
pp/interfaces/cli/main.py   CLI (typer) and slash commands. Owns Ctrl-C and tool confirmation.
pp/interfaces/cli/tui.py    Rich TUI: streams text, shows tool calls, diffs and prompts.
        |
        v  AgentEvent stream
pp/agents/coding.py         CodingAgent: the agent loop (LLM -> tool calls -> tool results -> LLM).
        |
pp/core/session.py          Session: owns config, LLM, tool registry, executor, history, hooks, MCP, memory.
        |
        +-- pp/llm/                 LLM adapters. router.py picks the adapter from [model] provider on each request.
        |     openai_responses.py   provider "openai" (Responses API)
        |     open_router.py        provider "openrouter" (Chat Completions API)
        +-- pp/context/history.py   ConversationHistory: entries, token count, compaction, (de)serialization.
        +-- pp/tools/registry.py    ToolRegistry: built-in tools, subagent tools, MCP tools.
        +-- pp/tools/executor.py    ToolExecutor: lookup, validation, approval, confirmation, hooks, execute.
        +-- pp/safety/approval.py   ApprovalChecker: decides from the approval policy and the tool effects.
        +-- pp/hooks/hooks.py       HookSystem: user shell commands at fixed trigger points.
        +-- pp/tools/mcp/           MCP client and manager. MCP tools are wrapped as Tool objects.
        +-- pp/prompts/system.py    System prompt and compaction prompt.
```

Other packages:

- `pp/domain/`: plain data types. `LLMEvent`, `AgentEvent`, `ToolCall`, `ToolResult`, `ToolEffects`, history entries (`UserMessage`, `AssistantTurn`, `ToolResultEntry`, `Summary`). Keep this package free of dependencies on other `pp` packages.
- `pp/config/`: `Config` (pydantic) and `loader.py`.
- `pp/tools/builtin/`: one file per built-in tool.
- `pp/tools/subagent.py`: subagent tools (`subagent_<name>`). A subagent runs its own `Session` with `is_subagent = True`.
- `pp/utils/`: paths, text and token count, process helpers, errors.

### Agent loop

`CodingAgent._loop()` runs at most `config.max_turns` times:

1. If the history is near the context window, compact it.
2. Stream one assistant turn from the LLM.
3. Add the assistant turn to the history. Save the session.
4. If there are no tool calls, stop.
5. Run each tool call through `ToolExecutor.run()`. Add each tool result at once. Save the session.

### Invariants

- Every tool call in the history has exactly one tool result after it. If the user stops a turn, the history closes the open call with an "interrupted" tool result.
- Compaction never separates a tool call from its tool result.
- `ToolExecutor.run()` never raises. Every outcome is a `ToolResult`, also a rejection or a crash.
- Approval decides from `Tool.effects()` (paths and shell command), not from parameter names.
- If a call needs confirmation and no `confirm` function is set, the call is rejected.
- The system prompt is built again for each LLM request. It is not part of the history.
- pp never guesses the provider from the model name.
- Session files are written to a temporary file and then renamed, so a crash cannot corrupt them.

### Ctrl-C

- At the prompt, Ctrl-C raises `KeyboardInterrupt`.
- During a turn, `CLI._run_turn()` cancels only the turn task. pp does not exit.
- Child processes (shell tool, hooks, MCP servers) start with `start_new_session=True`. Thus Ctrl-C does not reach them. Keep this flag on new subprocesses.

## Configuration and data

| Item | Location |
| --- | --- |
| User config | `~/.pair-programmer/config.toml` |
| Project config | `<cwd>/.pair-programmer/config.toml` (merged over the user config) |
| API keys | `.env` files. See `.env.example` and `.pair-programmer/.env.example`. |
| Project instructions for pp | `<cwd>/AGENT.md` (read as `developer_instructions`) |
| Sessions and memory | `~/.pair-programmer/user/` |

Note: pp itself reads `AGENT.md`, not this `AGENTS.md`.

## How to add a built-in tool

1. Create `pp/tools/builtin/<name>.py`.
2. Define a pydantic params model.
3. Subclass `Tool`. Set `name`, `description`, `type` (`ToolType`) and `schema`.
4. Implement `async execute(invocation) -> ToolResult`. Return `ToolResult.error_result(...)` for expected failures.
5. If the tool touches files or runs a command, override `effects()`.
6. Resolve user paths with `resolve_path_safe(invocation.cwd, path)`.
7. Add the class to `get_builtin_tools()` and `__all__` in `pp/tools/builtin/__init__.py`.
8. Add tests in `tests/test_tools.py`.

## How to add an LLM provider

1. Add a value to `Provider` in `pp/config/config.py`.
2. Subclass `BaseLLM` in `pp/llm/`. Implement `generate()` (an async generator of `LLMEvent`) and `close()`.
3. Add the class to `ADAPTERS` in `pp/llm/router.py`.
4. Add tests like `tests/test_openai_responses.py`.

## Tests

- `tests/conftest.py` has an autouse fixture. It sets `HOME` to a temporary directory and sets fake API keys. It also disables tiktoken, so tests need no network.
- Use the `project`, `config` and `invocation` fixtures for tool tests.
- Use `FakeLLM` with `text_turn()` and `tool_turn()` to script LLM replies for agent loop tests. A call with `tools=None` is a compaction call and gets a fixed summary.
- Tests must not call real LLM APIs or the network.

## Style

- Match the surrounding code: type hints on all functions in `pp/`, `from __future__ import annotations` where the file already uses it.
- Keep comments few. Write a comment only to explain why, not what.
- Commit messages use conventional prefixes: `feat:`, `fix:`, `chore:`, `refactor:`.
