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


async def test_create_agent_returns_similar_agents_instead_of_creating(env) -> None:
    roster, recorder = env

    result = tools.handle_tool_call(
        "create_agent", {"name": "Alice Invoice Reply", "description": "Reply to alice", "instructions": "say approved"}
    )
    await _settle()

    assert not result.success
    assert [a["id"] for a in result.payload["similar_agents"]] == ["a1"]
    assert len(roster.records()) == 1 and recorder.calls == []


async def test_confirm_new_creates_despite_similar_agents(env) -> None:
    roster, recorder = env

    result = tools.handle_tool_call(
        "create_agent",
        {"name": "Alice Birthday", "description": "Gift for alice", "instructions": "find a gift", "confirm_new": True},
    )
    await _settle()

    assert result.success and result.payload["agent_id"] == "a2"
    assert recorder.calls == [("Alice Birthday", "find a gift")]


def test_search_agents_matches_description_words(env) -> None:
    result = tools.handle_tool_call("search_agents", {"query": "the invoice thread"})

    assert [a["id"] for a in result.payload["agents"]] == ["a1"]
    assert tools.handle_tool_call("search_agents", {"query": "tokyo flight"}).payload["agents"] == []


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


@pytest.fixture
def two_hotels(env):
    roster, recorder = env
    roster.create("Hotel in Paris", "Hotel booking in Paris.")
    roster.create("Hotel in Paris for Mom", "Hotel booking in Paris for Mom.")
    return roster, recorder


def _send(agent_id: str, turn: tools.TurnContext):
    return tools.handle_tool_call("send_message_to_agent", {"agent_id": agent_id, "instructions": "check"}, turn)


async def test_a_pick_another_agent_fits_as_well_is_refused_with_both(two_hotels) -> None:
    roster, recorder = two_hotels
    turn = tools.TurnContext("does the paris hotel have late checkout?", transcript="")

    result = _send("a2", turn)
    await _settle()

    assert not result.success
    assert [c["name"] for c in result.payload["candidates"]] == ["Hotel in Paris", "Hotel in Paris for Mom"]
    assert recorder.calls == []
    assert roster.get("a2").last_used_at == roster.get("a3").last_used_at  # not touched either


async def test_after_a_refusal_the_model_may_send_to_any_candidate(two_hotels) -> None:
    _, recorder = two_hotels
    turn = tools.TurnContext("does the paris hotel have late checkout?", transcript="")
    _send("a2", turn)

    first, second = _send("a2", turn), _send("a3", turn)
    await _settle()

    assert first.success and second.success
    assert [name for name, _ in recorder.calls] == ["Hotel in Paris", "Hotel in Paris for Mom"]


async def test_a_word_only_one_agent_has_makes_the_pick_clear(two_hotels) -> None:
    _, recorder = two_hotels
    turn = tools.TurnContext("does mom's paris hotel have late checkout?", transcript="")

    result = _send("a3", turn)
    await _settle()

    assert result.success and recorder.calls == [("Hotel in Paris for Mom", "check")]


async def test_an_agent_the_conversation_named_is_not_ambiguous(two_hotels) -> None:
    _, recorder = two_hotels
    turn = tools.TurnContext("does it have late checkout?", transcript="Hotel in Paris (a2): Booked the Lutetia.")

    result = _send("a2", turn)
    await _settle()

    assert result.success and recorder.calls == [("Hotel in Paris", "check")]


async def test_sending_to_both_in_one_response_is_a_question_across_them_not_a_guess(two_hotels) -> None:
    _, recorder = two_hotels
    turn = tools.TurnContext("do the paris hotels have late checkout?", transcript="", batch_ids={"a2", "a3"})

    results = [_send("a2", turn), _send("a3", turn)]
    await _settle()

    assert all(r.success for r in results) and len(recorder.calls) == 2


async def test_agent_report_turns_skip_the_check(two_hotels) -> None:
    _, recorder = two_hotels

    result = tools.handle_tool_call("send_message_to_agent", {"agent_id": "a2", "instructions": "check"})
    await _settle()

    assert result.success and len(recorder.calls) == 1
