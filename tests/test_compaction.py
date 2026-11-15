from conftest import FakeLLM

from pp.config import Config
from pp.context import CompactionLayer, ContextManager


def tool_call(call_id: str) -> list[dict]:
    return [{"id": call_id, "type": "function", "function": {"name": "x", "arguments": "{}"}}]


def make_context(config: Config) -> ContextManager:
    cm = ContextManager(config=config, user_memory=None, tools=None)
    cm.add_user_message("u1")
    cm.add_assistant_message("", tool_calls=tool_call("t1"))
    cm.add_tool_result_message("t1", "r1")
    cm.add_assistant_message("", tool_calls=tool_call("t2"))
    cm.add_tool_result_message("t2", "r2")
    return cm


async def test_compaction_does_not_split_tool_call_from_result(config: Config):
    cm = make_context(config)

    # keep_last_n=1 would keep only the "tool" message for t2.
    await CompactionLayer(cm, FakeLLM()).compact(keep_last_n=1)

    assert [m.role for m in cm._messages] == ["system", "assistant", "tool"]
    assert cm._messages[1].tool_calls[0]["id"] == cm._messages[2].tool_call_id


async def test_compaction_no_op_when_nothing_to_summarize(config: Config):
    # The only valid cut point is index 0, so there is nothing to summarize.
    cm = ContextManager(config=config, user_memory=None, tools=None)
    cm.add_assistant_message("", tool_calls=tool_call("t1"))
    cm.add_tool_result_message("t1", "r1")

    summary, _ = await CompactionLayer(cm, FakeLLM()).compact(keep_last_n=1)

    assert summary == ""
    assert [m.role for m in cm._messages] == ["assistant", "tool"]


async def test_compaction_keeps_messages_added_during_summary(config: Config):
    cm = make_context(config)

    class SlowLLM(FakeLLM):
        async def generate(self, messages, tools, stream=True):
            cm.add_user_message("new while summarizing")
            async for event in super().generate(messages, tools, stream):
                yield event

    await CompactionLayer(cm, SlowLLM()).compact(keep_last_n=2)

    assert cm._messages[-1].content == "new while summarizing"


def test_needs_compaction_uses_context_window(config: Config):
    cm = ContextManager(config=config, user_memory=None, tools=None)
    cm.add_user_message("word " * 100)

    config.model.context_window = 1_000_000
    assert not cm.needs_compaction()
    config.model.context_window = 10
    assert cm.needs_compaction()


def test_tool_call_arguments_count_as_tokens(config: Config):
    cm = ContextManager(config=config, user_memory=None, tools=None)
    big_args = [{"id": "t", "type": "function", "function": {"name": "x", "arguments": "a" * 4000}}]
    cm.add_assistant_message("", tool_calls=big_args)

    assert cm._messages[0].token_count >= 1000


def test_token_model_follows_config(config: Config):
    cm = ContextManager(config=config, user_memory=None, tools=None)
    config.model_name = "some/other-model"
    assert cm._model_name == "some/other-model"
