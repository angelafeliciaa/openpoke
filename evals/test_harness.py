"""The sandbox keeps a replayed case away from real execution agents and real state."""

from __future__ import annotations

from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.services.execution import get_agent_roster

from evals.cases import load_cases
from evals.graders import grade
from evals.harness import _openrouter_body, _with_cache_breakpoints, run_case

REPLAY_MODEL = "gpt-5-mini"


def _case(case_id: str):
    return next(c for c in load_cases("routing") if c.id == case_id)


async def test_replay_never_reaches_the_real_execution_manager(monkeypatch) -> None:
    real_calls = []

    async def real_execute_agent(self, agent_name, instructions, request_id=None):
        real_calls.append(agent_name)

    monkeypatch.setattr(ExecutionBatchManager, "execute_agent", real_execute_agent)
    roster_before = get_agent_roster().records()

    run = await run_case(_case("reuse-03@50"), 1, "replay", REPLAY_MODEL)

    assert run.turns[-1].dispatched == ["Flight to Tokyo"]
    assert real_calls == []
    assert get_agent_roster().records() == roster_before


def test_cache_breakpoints_mark_system_and_first_user_message_only() -> None:
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "roster and turn"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "later"},
    ]
    cached = {"cache_control": {"type": "ephemeral"}}

    assert _with_cache_breakpoints(messages) == [
        {"role": "system", "content": [{"type": "text", "text": "prompt", **cached}]},
        {"role": "user", "content": [{"type": "text", "text": "roster and turn", **cached}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "later"},
    ]


def test_bare_openai_ids_go_through_openrouter_with_low_reasoning() -> None:
    body = _openrouter_body({"model": "gpt-5-mini", "messages": []})

    assert (body["model"], body["reasoning"]) == ("openai/gpt-5-mini", {"effort": "low"})


async def test_missing_recording_fails_instead_of_skipping() -> None:
    run = await run_case(_case("reuse-01@5"), 1, "replay", "no-such-model")

    assert run.missing_recording is not None
    assert run.turns == []
    assert grade(run).reason == "missing_recording"
