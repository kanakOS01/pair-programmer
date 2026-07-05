import asyncio

from pp.context.manager import ContextManager
from pp.domain.llm import LLMEventType
from pp.domain.message import Message
from pp.domain.shared import TokenUsage
from pp.llm.base import BaseLLM
from pp.prompts.system import get_compaction_prompt
from pp.utils.text import count_tokens


class CompactionLayer:
    """
    CompactionLayer manages summarizing the conversation history in the background
    to keep the context window size under control.
    """

    def __init__(self, context_manager: ContextManager, llm: BaseLLM) -> None:
        self.context_manager = context_manager
        self.llm = llm
        self.background_task: asyncio.Task[tuple[str, TokenUsage]] | None = None
        self.last_summary: str | None = None
        self.last_usage: TokenUsage | None = None

    def start_summary_in_background(self, keep_last_n: int = 4) -> asyncio.Task[tuple[str, TokenUsage]]:
        """
        Starts the context summarization and compaction process in a background asyncio task.
        If a background compaction is already running, returns the existing task.
        """
        if self.background_task and not self.background_task.done():
            return self.background_task

        self.background_task = asyncio.create_task(self.compact(keep_last_n))
        return self.background_task

    async def compact(self, keep_last_n: int = 4) -> tuple[str, TokenUsage]:
        """
        Summarizes older messages in the ContextManager, replaces them with a single
        summary system message, and returns the summary and token usage.
        """
        keep_last_n = max(0, keep_last_n)

        # Get the snapshot of the messages list
        current_messages = list(self.context_manager._messages)

        if keep_last_n >= len(current_messages):
            # Not enough messages to compact
            return "", TokenUsage()

        # Identify messages to summarize
        messages_to_summarize = current_messages[:-keep_last_n] if keep_last_n > 0 else current_messages
        k = len(messages_to_summarize)
        if k == 0:
            return "", TokenUsage()

        # Format messages to summarize
        history_text = self._format_messages_for_summary(messages_to_summarize)

        summary_messages = [
            {"role": "system", "content": get_compaction_prompt()},
            {
                "role": "user",
                "content": f"Please summarize the following conversation history concisely:\n\n{history_text}",
            },
        ]

        summary_text = ""
        usage = TokenUsage()

        # Call LLM to summarize
        async for event in self.llm.generate(messages=summary_messages, tools=None, stream=False):
            if event.type == LLMEventType.TextDelta and event.text_delta:
                summary_text += event.text_delta.text
            elif event.type == LLMEventType.Done:
                if event.text_delta and not summary_text:
                    summary_text = event.text_delta.text
                if event.usage:
                    usage = event.usage
            elif event.type == LLMEventType.Error:
                raise RuntimeError(f"LLM compaction failed: {event.error}")

        summary_text = summary_text.strip()

        # Create system summary message
        summary_content = f"Summary of previous conversation:\n{summary_text}"
        summary_message = Message(
            role="system",
            content=summary_content,
            token_count=count_tokens(summary_content, self.context_manager._model_name),
        )

        # Atomically / safely replace the first k messages in ContextManager's message list
        # while keeping any new messages appended while the LLM call was running.
        self.context_manager._messages = [summary_message] + self.context_manager._messages[k:]

        # Update last summary and usage stats
        self.last_summary = summary_text
        self.last_usage = usage

        return summary_text, usage

    def _format_messages_for_summary(self, messages: list[Message]) -> str:
        lines = []
        for msg in messages:
            role = msg.role
            content = msg.content or ""
            if msg.tool_calls:
                content += f"\n[Tool Calls: {msg.tool_calls}]"
            if msg.tool_call_id:
                lines.append(f"TOOL (ID: {msg.tool_call_id}): {content}")
            else:
                lines.append(f"{role.upper()}: {content}")
        return "\n\n".join(lines)
