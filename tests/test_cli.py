from pathlib import Path

import pytest
from conftest import FakeLLM, text_turn, tool_turn

from pp.agents import CodingAgent
from pp.config import Config
from pp.config.config import ApprovalConfig, ApprovalPolicy
from pp.interfaces.cli.main import CLI


@pytest.fixture
async def cli(config: Config):
    cli = CLI(config)
    async with CodingAgent(config) as agent:
        cli.coding_agent = agent
        yield cli


async def test_process_message_runs_tool_after_text(cli: CLI, project: Path):
    cli.coding_agent.session.llm = FakeLLM(
        [
            tool_turn("Let me write it.", "c1", "write_file", {"path": "a.txt", "content": "hi"}),
            text_turn("Done."),
        ]
    )

    result = await cli._process_message("go")

    assert (project / "a.txt").read_text() == "hi"
    assert result == "Done."


async def test_approval_command_keeps_approval_config(cli: CLI, config: Config):
    cli._handle_approval_command("yolo")

    assert isinstance(config.approval, ApprovalConfig)
    assert config.approval.policy == ApprovalPolicy.YOLO


@pytest.mark.parametrize("key", ["approval", "approval.policy"])
async def test_config_command_sets_approval_policy(cli: CLI, config: Config, key: str):
    assert cli._set_config_attr(key, "on-request")

    assert isinstance(config.approval, ApprovalConfig)
    assert config.approval.policy == ApprovalPolicy.ON_REQUEST


async def test_config_command_rejects_invalid_policy(cli: CLI, config: Config):
    assert not cli._set_config_attr("approval", "sometimes")
    assert config.approval.policy == ApprovalPolicy.AUTO_EDIT


async def test_config_command_keeps_path_type(cli: CLI, config: Config, project: Path):
    assert cli._set_config_attr("cwd", str(project))
    assert isinstance(config.cwd, Path)


async def test_config_command_sets_numbers(cli: CLI, config: Config):
    assert cli._set_config_attr("model.temp", "0.5")
    assert cli._set_config_attr("max_turns", "7")
    assert config.model.temp == 0.5
    assert config.max_turns == 7


async def test_allowed_tools_rebuild_keeps_session_and_mcp_tools(cli: CLI, config: Config):
    session = cli.coding_agent.session
    mcp_tool = session.tool_registry.get("read_file")
    # Pretend that this tool came from an MCP server.
    mcp_tool.name = "mcp_fake_tool"
    session.mcp_tools = [mcp_tool]

    assert cli._set_config_attr("allowed_tools", "list_dir,mcp_fake_tool")

    names = sorted(t.name for t in session.tool_registry.list_tools())
    assert names == ["list_dir", "mcp_fake_tool"]
    assert session.tool_registry.get("list_dir").session is session
