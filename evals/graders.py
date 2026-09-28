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
    """Grade the last user turn, then the relay turn when the case scripted agent reports.

    A turn that routes nothing and replies with a question is `asked_user`: the right call
    only when the case is ambiguous, and friction for the user everywhere else.
    """
    if run.missing_recording:
        return Verdict(False, "missing_recording")
    verdict = _grade_last_turn(run)
    if run.relay is None or not verdict.passed:
        return verdict
    return _grade_relay(run)


def _grade_relay(run: CaseRun) -> Verdict:
    """Every dispatched agent answered; the user must hear each answer's marker."""
    relay = run.relay
    if relay.error:
        return Verdict(False, "error")
    told = relay.response.lower()
    dispatched = set(run.turns[-1].dispatched)
    dropped = [r.agent for r in run.case.reports if r.agent in dispatched and r.must_relay.lower() not in told]
    if dropped:
        return Verdict(False, "dropped_a_report")
    return Verdict(True, "relayed_every_report")


_ITERATION_CAP = "tool iteration limit"


def _grade_last_turn(run: CaseRun) -> Verdict:
    last = run.turns[-1]
    if last.error and _ITERATION_CAP in last.error:
        # the model kept calling tools one per response until the runtime's cap and never ended its turn
        return Verdict(False, "hit_tool_iteration_limit")
    if last.error:
        return Verdict(False, "error")
    case = run.case
    if not last.dispatched and not last.new_agents:
        if "?" in last.response and _MENTIONS_AGENTS.search(last.response):
            return Verdict(False, "asked_about_agents")
        if "?" in last.response:
            return Verdict(case.kind == "ambiguous", "asked_user")
        return Verdict(False, "no_delegation")

    if case.kind == "relay":
        if last.new_agents:
            return Verdict(False, "spawned_duplicate")
        if all(_any_of(last.dispatched, [c]) for c in case.candidates):
            return Verdict(True, "checked_every_candidate")
        if _any_of(last.dispatched, case.candidates):
            return Verdict(False, "checked_one_candidate")
        return Verdict(False, "wrong_existing_agent")

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
