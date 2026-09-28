"""Grader and summary behaviour on hand-built runs, no LLM involved."""

from __future__ import annotations

from typing import List, Optional

import pytest

from evals.cases import Case, Report, RosterEntry
from evals.graders import Verdict, grade
from evals.harness import CaseRun, TurnResult
from evals.report import summarize, to_row

ROSTER = tuple(
    RosterEntry(f"a{i}", name, "") for i, name in enumerate(("Email to Alice", "Email to Alicia", "Flight to Tokyo"), 1)
)


def _case(kind: str, target: Optional[str] = None, turns=("do the thing",), candidates=(), read_only=False,
          reports=()) -> Case:
    return Case(id=f"{kind}-x", suite="t", kind=kind, family=kind, target=target, roster=ROSTER, turns=turns,
                candidates=candidates, read_only=read_only, reports=reports)


def _turn(dispatched: List[str], new: List[str] = (), error: Optional[str] = None, response: str = "") -> TurnResult:
    return TurnResult(dispatched=list(dispatched), new_agents=list(new), calls=[], response=response, error=error)


QUESTION = "alice or alicia?"
REPORTS = (Report("Email to Alice", "She approved the invoice.", "approved"),
           Report("Flight to Tokyo", "Confirmed, seat 14A.", "14A"))
RELAY = _case("relay", candidates=("Email to Alice", "Flight to Tokyo"), reports=REPORTS)
AMBIGUOUS_WITH_REPORTS = _case("ambiguous", candidates=("Email to Alice", "Email to Alicia"), read_only=True,
                               reports=(REPORTS[0], Report("Email to Alicia", "No reply yet.", "No reply")))


def _relayed(case: Case, dispatched: List[str], told: str) -> CaseRun:
    run = _run(case, _turn(dispatched))
    run.relay = _turn([], response=told)
    return run


def _run(case: Case, *turns: TurnResult, missing: Optional[str] = None) -> CaseRun:
    return CaseRun(case=case, trial=1, model="m", turns=list(turns), missing_recording=missing)


REUSE = _case("reuse", target="Email to Alice")
CREATE = _case("create")
DRIFT = _case("drift", turns=("book a table", "make it 3"))
AMBIGUOUS = _case("ambiguous", candidates=("Email to Alice", "Email to Alicia"))
AMBIGUOUS_QUESTION = _case("ambiguous", candidates=("Email to Alice", "Email to Alicia"), read_only=True)


@pytest.mark.parametrize(
    "run, expected",
    [
        (_run(REUSE, _turn(["email  to alice"])), Verdict(True, "reused_target")),
        (_run(REUSE, _turn(["Email to Alice", "Alice Invoice"], new=["Alice Invoice"])),
         Verdict(False, "reused_target_but_also_spawned")),
        (_run(REUSE, _turn(["Alice Invoice"], new=["Alice Invoice"])), Verdict(False, "spawned_duplicate")),
        (_run(REUSE, _turn(["Email to Alicia"])), Verdict(False, "wrong_existing_agent")),
        (_run(REUSE, _turn([])), Verdict(False, "no_delegation")),
        (_run(CREATE, _turn(["Thai Dinner"], new=["Thai Dinner"])), Verdict(True, "created_new")),
        (_run(CREATE, _turn(["Flight to Tokyo"])), Verdict(False, "reused_unrelated_agent")),
        (_run(DRIFT, _turn(["Thai Dinner"], new=["Thai Dinner"]), _turn(["thai dinner"])),
         Verdict(True, "reused_turn1_agent")),
        (_run(DRIFT, _turn(["Thai Dinner"], new=["Thai Dinner"]), _turn(["Dinner for 3"], new=["Dinner for 3"])),
         Verdict(False, "drifted_to_new_name")),
        (_run(DRIFT, _turn(["Flight to Tokyo"]), _turn(["Flight to Tokyo"])), Verdict(False, "turn1_did_not_create")),
        (_run(DRIFT, _turn(["Thai Dinner"], new=["Thai Dinner"]), _turn(["Flight to Tokyo"])),
         Verdict(False, "wrong_existing_agent")),
        (_run(REUSE, _turn(["Email to Alice"], error="boom")), Verdict(False, "error")),
        (_run(REUSE, _turn(["Email to Alice"], error="Reached tool iteration limit without final response")),
         Verdict(False, "hit_tool_iteration_limit")),
        (_run(REUSE, missing="abc.json"), Verdict(False, "missing_recording")),
        (_run(REUSE, _turn([], response=QUESTION)), Verdict(False, "asked_user")),
        (_run(AMBIGUOUS, _turn([], response=QUESTION)), Verdict(True, "asked_user")),
        (_run(AMBIGUOUS, _turn([], response="I see two dentist agents. Which one?")),
         Verdict(False, "asked_about_agents")),
        (_run(AMBIGUOUS, _turn([], response="on it")), Verdict(False, "no_delegation")),
        (_run(AMBIGUOUS, _turn(["Email to Alicia"])), Verdict(False, "guessed_candidate")),
        (_run(AMBIGUOUS, _turn(["Email to Alicia", "Email to Alice"])), Verdict(False, "messaged_every_candidate")),
        (_run(AMBIGUOUS_QUESTION, _turn(["Email to Alicia", "Email to Alice"])), Verdict(True, "checked_every_candidate")),
        (_run(AMBIGUOUS_QUESTION, _turn(["Email to Alicia"])), Verdict(False, "guessed_candidate")),
        (_run(AMBIGUOUS_QUESTION, _turn([], response=QUESTION)), Verdict(True, "asked_user")),
        (_run(AMBIGUOUS, _turn(["Alice Late"], new=["Alice Late"])), Verdict(False, "spawned_duplicate")),
        (_run(AMBIGUOUS, _turn(["Flight to Tokyo"])), Verdict(False, "wrong_existing_agent")),
        (_run(RELAY, _turn(["Email to Alice"])), Verdict(False, "checked_one_candidate")),
        (_run(RELAY, _turn(["Email to Alicia"])), Verdict(False, "wrong_existing_agent")),
        (_run(RELAY, _turn(["Email to Alice", "Flight to Tokyo"])), Verdict(True, "checked_every_candidate")),
        (_relayed(RELAY, ["Email to Alice", "Flight to Tokyo"], "Alice approved it and the flight is confirmed, seat 14A."),
         Verdict(True, "relayed_every_report")),
        (_relayed(RELAY, ["Email to Alice", "Flight to Tokyo"], "Alice approved the invoice. Flight's confirmed too."),
         Verdict(False, "dropped_a_report")),
        (_relayed(AMBIGUOUS_WITH_REPORTS, ["Email to Alice", "Email to Alicia"], "Alice approved; no reply from Alicia yet."),
         Verdict(True, "relayed_every_report")),
        (_relayed(AMBIGUOUS_WITH_REPORTS, ["Email to Alice", "Email to Alicia"], "Alice approved it."),
         Verdict(False, "dropped_a_report")),
    ],
)
def test_grade(run: CaseRun, expected: Verdict) -> None:
    assert grade(run) == expected


def test_relay_is_not_graded_when_the_routing_turn_failed() -> None:
    run = _relayed(RELAY, ["Email to Alice"], "Alice approved it.")

    assert grade(run) == Verdict(False, "checked_one_candidate")


def test_unscored_runs_do_not_count_as_delegations() -> None:
    runs = [
        _run(REUSE, _turn(["Email to Alice"])),
        _run(REUSE, _turn([])),
        _run(REUSE, missing="abc.json"),
    ]
    [summary] = summarize([to_row(r, grade(r)) for r in runs])

    assert summary["unscored"] == 1
    assert summary["delegation_rate"] == 0.5
    assert summary["by_kind"] == {"reuse": 1.0}
    assert summary["pass_all_trials"] == 0.0


def test_asking_counts_as_a_routing_decision_and_in_the_ask_rate() -> None:
    runs = [_run(REUSE, _turn(["Email to Alice"])), _run(REUSE, _turn([], response=QUESTION))]
    [summary] = summarize([to_row(r, grade(r)) for r in runs])

    assert (summary["delegation_rate"], summary["ask_rate"], summary["by_kind"]) == (1.0, 0.5, {"reuse": 0.5})
