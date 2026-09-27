"""The sandbox keeps a replayed case away from real execution agents and real state."""

from __future__ import annotations

from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.services.execution import get_agent_roster

from evals.cases import load_cases
from evals.graders import grade
from evals.harness import run_case

REPLAY_MODEL = "gpt-5-mini"


def _case(case_id: str):
    return next(c for c in load_cases("routing") if c.id == case_id)


async def test_replay_never_reaches_the_real_execution_manager(monkeypatch) -> None:
    real_calls = []

    async def real_execute_agent(self, agent_name, instructions, request_id=None):
        real_calls.append(agent_name)

    monkeypatch.setattr(ExecutionBatchManager, "execute_agent", real_execute_agent)
    roster_before = get_agent_roster().get_agents()

    run = await run_case(_case("reuse-01@50"), 1, "replay", REPLAY_MODEL)

    assert run.turns[-1].dispatched == ["Email to Alice"]
    assert real_calls == []
    assert get_agent_roster().get_agents() == roster_before


async def test_missing_recording_fails_instead_of_skipping() -> None:
    run = await run_case(_case("reuse-01@5"), 1, "replay", "no-such-model")

    assert run.missing_recording is not None
    assert run.turns == []
    assert grade(run).reason == "missing_recording"
