"""Pass/fail grading for a routing run. Plain code, no LLM judge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from evals.harness import CaseRun


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def _any_of(names: Iterable[str], wanted: Iterable[str]) -> bool:
    wanted_norm = {_norm(w) for w in wanted}
    return any(_norm(n) in wanted_norm for n in names)


def grade(run: CaseRun) -> Verdict:
    """Grade the last turn; drift also reads turn 1 to learn which agent it created."""
    if run.missing_recording:
        return Verdict(False, "missing_recording")
    last = run.turns[-1]
    if last.error:
        return Verdict(False, "error")
    if not last.dispatched:
        return Verdict(False, "no_delegation")

    case = run.case
    if case.kind == "reuse":
        hit = _any_of(last.dispatched, [case.target])
        if hit and not last.new_agents:
            return Verdict(True, "reused_target")
        if hit:
            return Verdict(False, "reused_target_but_also_spawned")
        if last.new_agents:
            return Verdict(False, "spawned_duplicate")
        return Verdict(False, "wrong_existing_agent")

    if case.kind == "drift":
        created = run.turns[0].new_agents
        if not created:
            return Verdict(False, "turn1_did_not_create")
        if last.new_agents:
            return Verdict(False, "drifted_to_new_name")
        if _any_of(last.dispatched, created):
            return Verdict(True, "reused_turn1_agent")
        return Verdict(False, "wrong_existing_agent")

    if last.new_agents:
        return Verdict(True, "created_new")
    return Verdict(False, "reused_unrelated_agent")
