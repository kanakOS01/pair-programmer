# Pair Programmer (pp)

Pair Programmer (`pp`) is an interactive, agentic AI coding assistant designed to help developers write, explore, and refactor code directly from the terminal.

Built with modern Python, it leverages large language models (LLMs) to provide an intelligent, context-aware coding experience. It features a beautiful Terminal User Interface (TUI) powered by `rich` and `typer`.

## Features

- **Interactive TUI**: A rich terminal interface with Markdown support, syntax highlighting, and live streaming of LLM responses.
- **Agentic Capabilities**: The assistant is equipped with built-in tools to navigate and modify your project:
  - **File Operations**: `read_file`, `list_dir`, `apply_patch` (smart multi-block code editing).
  - **Search**: `grep`, `glob`.
  - **Context Management**: `memory` (persistent memory) and `todos` (task management).
- **Flexible Configuration**: Support for multiple LLM providers and dynamic model selection.
- **Project Context**: Automatically detects your current working directory and interacts with your codebase.

## Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) (for dependency management)

## Installation

Clone the repository and install the dependencies using `uv`:

```bash
git clone https://github.com/yourusername/pair-programmer.git
cd pair-programmer
uv sync
```

## Usage

You can start the Pair Programmer in interactive mode:

```bash
uv run pp
```

Alternatively, you can run a one-off command:

```bash
uv run pp "Create a new python script that calculates fibonacci numbers"
```

You can specify the working directory:

```bash
uv run pp --cwd /path/to/project
```


## Architecture

The project is structured into several key domains:

- `pp/agents/`: Contains the core agent logic (`CodingAgent`) that interacts with the LLM and orchestrates tool calls.
- `pp/interfaces/cli/`: The command-line interface and rich TUI implementation.
- `pp/tools/`: Built-in tools available to the agent (e.g., file system, memory, search).
- `pp/llm/`: Interfaces for communicating with various LLM providers.
- `pp/domain/`: Core domain models and event types for agent communication.

## Development

This project uses `ruff` for linting and formatting, and `pre-commit` for Git hooks.

```bash
uv run ruff check .
uv run ruff format .
```
