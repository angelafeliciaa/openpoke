"""Pass/fail grading for a routing run. Plain code, no LLM judge."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from evals.harness import CaseRun


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


# the user never sees agents; a question that names them leaks the machinery
_MENTIONS_AGENTS = re.compile(r"\bagents?\b", re.IGNORECASE)


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


def _any_of(names: Iterable[str], wanted: Iterable[str]) -> bool:
    wanted_norm = {_norm(w) for w in wanted}
    return any(_norm(n) in wanted_norm for n in names)


def grade(run: CaseRun) -> Verdict:
    """Grade the last turn; drift also reads turn 1 to learn which agent it created.

    A turn that routes nothing and replies with a question is `asked_user`: the right call
    only when the case is ambiguous, and friction for the user everywhere else.
    """
    if run.missing_recording:
        return Verdict(False, "missing_recording")
    last = run.turns[-1]
    if last.error:
        return Verdict(False, "error")
    case = run.case
    if not last.dispatched and not last.new_agents:
        if "?" in last.response and _MENTIONS_AGENTS.search(last.response):
            return Verdict(False, "asked_about_agents")
        if "?" in last.response:
            return Verdict(case.kind == "ambiguous", "asked_user")
        return Verdict(False, "no_delegation")

    if case.kind == "ambiguous":
        if last.new_agents:
            return Verdict(False, "spawned_duplicate")
        # answers "did the renewal go through?" for both; does "tell alice i'm late" to both
        if all(_any_of(last.dispatched, [c]) for c in case.candidates):
            if case.read_only:
                return Verdict(True, "checked_every_candidate")
            return Verdict(False, "messaged_every_candidate")
        if _any_of(last.dispatched, case.candidates):
            return Verdict(False, "guessed_candidate")
        return Verdict(False, "wrong_existing_agent")
    if not last.dispatched:
        return Verdict(False, "no_delegation")

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
