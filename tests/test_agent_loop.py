from pathlib import Path

from conftest import FakeLLM, text_turn, tool_turn

from pp.agents import CodingAgent
from pp.config import Config
from pp.domain import AgentEventType, AssistantTurn, Summary, ToolResultEntry, UserMessage


async def run_agent(config: Config, llm: FakeLLM, prompt: str = "go"):
    async with CodingAgent(config) as agent:
        agent.session.llm = llm
        events = [event async for event in agent.run(prompt)]
        return agent, events


async def test_text_and_tool_call_in_same_turn_runs_tool(config: Config, project: Path):
    llm = FakeLLM(
        [
            tool_turn("Let me write it.", "c1", "write_file", {"path": "a.txt", "content": "hi"}),
            text_turn("Done."),
        ]
    )

    agent, events = await run_agent(config, llm)

    assert (project / "a.txt").read_text() == "hi"
    done = [e for e in events if e.type == AgentEventType.Done]
    assert done[0].data["response"] == "Done."


async def test_tool_result_follows_its_tool_call_in_history(config: Config):
    llm = FakeLLM([tool_turn("", "c1", "list_dir", {}), text_turn("ok")])

    agent, _ = await run_agent(config, llm)

    kinds = [type(e) for e in agent.session.history.entries]
    assert kinds == [UserMessage, AssistantTurn, ToolResultEntry, AssistantTurn]
    # The second request must contain the tool result.
    assert any(isinstance(e, ToolResultEntry) and e.call_id == "c1" for e in llm.calls[1])


async def test_unknown_tool_returns_error_to_model(config: Config):
    llm = FakeLLM([tool_turn("", "c1", "no_such_tool", {}), text_turn("ok")])

    agent, events = await run_agent(config, llm)

    tool_done = [e for e in events if e.type == AgentEventType.ToolCallDone]
    assert tool_done[0].data["success"] is False
    assert "not found" in tool_done[0].data["error"]


async def test_auto_compaction_runs_when_context_is_full(config: Config):
    config.model.context_window = 50
    llm = FakeLLM([text_turn("first"), text_turn("second")])

    async with CodingAgent(config) as agent:
        agent.session.llm = llm
        # Fill the history so that it is above the threshold.
        for i in range(10):
            agent.session.history.add_user(f"old message {i} " * 20)
            agent.session.history.add_assistant_turn(f"old answer {i} " * 20)
        _ = [e async for e in agent.run("new")]

        first = agent.session.history.entries[0]
        assert isinstance(first, Summary)
        assert first.text == "SUMMARY"


async def test_session_is_saved_and_loaded(config: Config):
    llm = FakeLLM([text_turn("hello")])
    agent, _ = await run_agent(config, llm)

    async with CodingAgent(config) as other:
        other.session.load(agent.session.session_id)
        entries = other.session.history.entries

    assert entries == agent.session.history.entries
    assert [e.text for e in entries] == ["go", "hello"]


async def test_subagent_tool_calls_run_hooks(config: Config, project: Path, monkeypatch):
    import pp.core.session
    from pp.config.config import HookConfig

    config.hooks_enabled = True
    config.hooks = [HookConfig(name="log", trigger="before tool", script='echo "$PP_TOOL_NAME" >> hooks.log')]

    # The subagent builds its own LLM inside Session. Give it a scripted one.
    child_llm = FakeLLM([tool_turn("", "s1", "list_dir", {}), text_turn("found it")])
    monkeypatch.setattr(pp.core.session, "OpenRouterLLM", lambda cfg: child_llm)

    parent_llm = FakeLLM([tool_turn("", "p1", "subagent_codebase_investigator", {"task": "look around"}), text_turn("done")])
    agent, events = await run_agent(config, parent_llm)

    assert (project / "hooks.log").read_text().splitlines() == ["subagent_codebase_investigator", "list_dir"]
    results = [e.data for e in events if e.type == AgentEventType.ToolCallDone]
    assert results[0]["output"] == "found it"
