from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List, Tuple

import pytest

from server.agents.execution_agent import batch_manager
from server.agents.execution_agent.runtime import ExecutionResult
from server.agents.interaction_agent import agent as ia_agent
from server.agents.interaction_agent import tools
from server.services.execution import log_store, roster as roster_mod


class _Recorder:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, str]] = []

    async def execute_agent(self, agent_name: str, instructions: str, request_id=None) -> ExecutionResult:
        self.calls.append((agent_name, instructions))
        return ExecutionResult(agent_name=agent_name, success=True, response="done")


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    roster = roster_mod.AgentRoster(tmp_path / "roster.json")
    roster.create("Email to Alice", "Invoice thread with alice@piedpiper.com")
    recorder = _Recorder()
    monkeypatch.setattr(roster_mod, "_agent_roster", roster)
    monkeypatch.setattr(log_store, "_execution_agent_logs", log_store.ExecutionAgentLogStore(tmp_path / "logs"))
    monkeypatch.setattr(tools, "_EXECUTION_BATCH_MANAGER", recorder)
    return roster, recorder


async def _settle() -> None:
    await asyncio.sleep(0)


async def test_unknown_id_is_an_error_and_creates_nothing(env) -> None:
    roster, recorder = env

    result = tools.handle_tool_call("send_message_to_agent", {"agent_id": "a9", "instructions": "hi"})
    await _settle()

    assert not result.success and "a9" in result.payload["error"]
    assert [r.id for r in roster.records()] == ["a1"]
    assert recorder.calls == []


async def test_a_name_is_not_accepted_where_an_id_is_expected(env) -> None:
    roster, recorder = env

    result = tools.handle_tool_call("send_message_to_agent", {"agent_id": "Email to Alice", "instructions": "hi"})
    await _settle()

    assert not result.success
    assert recorder.calls == []


async def test_known_id_dispatches_to_that_agent_by_name(env) -> None:
    _, recorder = env

    result = tools.handle_tool_call("send_message_to_agent", {"agent_id": "a1", "instructions": "say approved"})
    await _settle()

    assert result.success and result.payload["new_agent_created"] is False
    assert recorder.calls == [("Email to Alice", "say approved")]


async def test_create_agent_registers_the_description_and_dispatches(env) -> None:
    roster, recorder = env

    result = tools.handle_tool_call(
        "create_agent", {"name": "Thai Dinner", "description": "Table for 2 in Soho", "instructions": "book it"}
    )
    await _settle()

    assert result.payload["agent_id"] == "a2"
    assert roster.get("a2").description == "Table for 2 in Soho"
    assert recorder.calls == [("Thai Dinner", "book it")]


async def test_create_agent_with_a_taken_name_points_at_the_existing_id(env) -> None:
    roster, recorder = env

    result = tools.handle_tool_call(
        "create_agent", {"name": "email to alice", "description": "dup", "instructions": "hi"}
    )
    await _settle()

    assert not result.success and "a1" in result.payload["error"]
    assert len(roster.records()) == 1
    assert recorder.calls == []


def test_active_agents_render_id_name_and_escaped_description(env) -> None:
    roster, _ = env
    roster.create("Q&A <Prep>", "")

    assert ia_agent._render_active_agents() == (
        '<agent id="a1" name="Email to Alice">Invoice thread with alice@piedpiper.com</agent>\n'
        '<agent id="a2" name="Q&amp;A &lt;Prep&gt;" />'
    )


def test_agent_reports_carry_the_agent_id(env) -> None:
    payload = batch_manager.ExecutionBatchManager()._format_batch_payload(
        [ExecutionResult(agent_name="Email to Alice", success=True, response="Sent.")]
    )

    assert payload == "[SUCCESS] Email to Alice (a1): Sent."
