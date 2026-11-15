# Pair Programmer

A terminal coding agent. A user talks to a model, and the model reads and changes the user's project through tools.

## Language

### Conversation

**Conversation history**:
The ordered record of user messages, assistant turns and tool results in one session. It does not include the system prompt.
_Avoid_: context, messages, context manager

**System prompt**:
The instructions that the session builds from config, tools and memory for each LLM request. It is not part of the conversation history.
_Avoid_: system message

**Assistant turn**:
One model reply: its text and the tool calls it asks for.
_Avoid_: assistant message, response

**Tool call**:
A request from the model to run one tool with given arguments. Every tool call in the conversation history has exactly one tool result after it.
_Avoid_: function call

**Tool result**:
The outcome of one tool call, sent back to the model.
_Avoid_: tool output, tool message

**Interrupted tool call**:
A tool call that has no result because the user stopped the turn. The conversation history closes it with an "interrupted" tool result when the next user message arrives.

**Compaction**:
Replacing the oldest part of the conversation history with a summary, so the history fits the model's context window. The cut never separates a tool call from its tool result.
_Avoid_: summarization, truncation

### Tool execution

**Tool execution**:
Running one tool call from start to end: lookup, parameter validation, approval, hooks and the tool itself. It always produces a tool result, also when the call is rejected or fails.
_Avoid_: tool invocation, dispatch

**Tool effects**:
What a tool call will touch, as the tool itself declares it: file paths and a shell command. Approval decides from the effects, not from parameter names.
_Avoid_: affected paths, side effects

**Approval policy**:
The user's setting (on-request, auto-edit, auto, never, yolo) that decides if a tool call runs, needs confirmation, or is rejected.
_Avoid_: permission mode

**Confirmation**:
Asking a person to allow one tool call. If no one can be asked, the tool call is rejected.
_Avoid_: prompt, approval request

**Hook**:
A user-configured command that runs at a fixed point: before or after the agent, before or after a tool call, or on error.

### Models

**Provider**:
The service that runs the model: `openai` (the OpenAI API, through the Responses API) or `openrouter` (OpenRouter, through the Chat Completions API). The user sets it in `[model] provider`; pp never guesses it from the model name.
_Avoid_: backend, vendor
