"""Routing cases: the typed model and the one loader for `cases/<suite>.jsonl`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

CASES_DIR = Path(__file__).parent
SUITES = ("routing", "hard")

Kind = Literal["reuse", "create", "drift", "ambiguous", "relay"]
HistoryRole = Literal["user", "assistant", "agent"]
_KINDS = ("reuse", "create", "drift", "ambiguous", "relay")
_ROLES = ("user", "assistant", "agent")


@dataclass(frozen=True)
class HistoryEntry:
    role: HistoryRole
    text: str


@dataclass(frozen=True)
class RosterEntry:
    """An agent that exists before the case starts; seeded into the roster exactly as given."""

    id: str
    name: str
    description: str


@dataclass(frozen=True)
class Report:
    """What an execution agent answers when the case dispatches to it, scripted so the relay is gradable.

    `must_relay` is one distinctive token from `says` (a name, a code, a price) that the user's reply
    has to carry; dropping it means the agent's answer never reached the user.
    """

    agent: str
    says: str
    must_relay: str


@dataclass(frozen=True)
class Case:
    """One routing scenario.

    reuse: `target` is in the roster and must be called, with nothing spawned.
    create: nothing in the roster fits, so a new agent must be spawned.
    drift: turn 1 creates an agent and the last turn must come back to it.
    ambiguous: two or more `candidates` fit equally well. For an action the agent must ask the user;
    for a `read_only` question, checking every candidate answers it too.
    relay: one question spans every `candidates` agent (the tokyo flight and the lisbon hotel), so
    all of them must be checked and each one's `reports` answer told back to the user.

    Cases with `reports` continue after the last user turn: every candidate that was dispatched
    answers with its scripted report, and the reply to the user is graded for each `must_relay`.
    """

    id: str
    suite: str
    kind: Kind
    family: str
    target: Optional[str]
    roster: Tuple[RosterEntry, ...]
    turns: Tuple[str, ...]
    history: Tuple[HistoryEntry, ...] = ()
    candidates: Tuple[str, ...] = ()
    read_only: bool = False
    reports: Tuple[Report, ...] = ()
    summarize: bool = False
    """Run OpenPoke's conversation summariser over the seeded history before the first turn, so
    the turn sees a summary plus the last few lines, as a long-running chat would."""

    @property
    def roster_size(self) -> int:
        return len(self.roster)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(r.name for r in self.roster)

    @classmethod
    def parse(cls, raw: dict, suite: str) -> "Case":
        kind = raw["kind"]
        if kind not in _KINDS:
            raise ValueError(f"{raw['id']}: unknown kind {kind!r}")
        if (kind == "reuse") != (raw.get("target") is not None):
            raise ValueError(f"{raw['id']}: reuse cases need a target and only reuse cases have one")
        history = tuple(HistoryEntry(h["role"], h["text"]) for h in raw.get("history") or [])
        for h in history:
            if h.role not in _ROLES:
                raise ValueError(f"{raw['id']}: unknown history role {h.role!r}")
        turns = tuple(raw.get("turns") or [raw["message"]])
        if kind == "drift" and len(turns) < 2:
            raise ValueError(f"{raw['id']}: drift cases need at least two turns")
        candidates = tuple(raw.get("candidates") or ())
        if (kind in ("ambiguous", "relay")) != (len(candidates) >= 2):
            raise ValueError(f"{raw['id']}: ambiguous and relay cases need two or more candidates and only they have them")
        read_only = bool(raw.get("read_only"))
        if read_only and kind != "ambiguous":
            raise ValueError(f"{raw['id']}: only ambiguous cases are marked read_only")
        reports = tuple(Report(r["agent"], r["says"], r["must_relay"]) for r in raw.get("reports") or ())
        if reports and not (kind == "relay" or (kind == "ambiguous" and read_only)):
            raise ValueError(f"{raw['id']}: only relay cases and read-only ambiguous cases carry reports")
        if (kind == "relay" or reports) and {r.agent for r in reports} != set(candidates):
            raise ValueError(f"{raw['id']}: every candidate needs exactly one report")
        for r in reports:
            if r.must_relay.lower() not in r.says.lower():
                raise ValueError(f"{raw['id']}: must_relay {r.must_relay!r} is not in what the agent says")
        case = cls(
            id=raw["id"],
            suite=suite,
            kind=kind,
            family=raw.get("family") or kind,
            target=raw.get("target"),
            roster=tuple(RosterEntry(r["id"], r["name"], r["description"]) for r in raw["roster"]),
            turns=turns,
            history=history,
            candidates=candidates,
            read_only=read_only,
            reports=reports,
            summarize=bool(raw.get("summarize")),
        )
        if case.roster_size != raw["roster_size"]:
            raise ValueError(f"{case.id}: roster has {case.roster_size} names, roster_size says {raw['roster_size']}")
        if not set(candidates) <= set(case.names):
            raise ValueError(f"{case.id}: every candidate must be in the roster")
        if case.target is not None and case.target not in case.names:
            raise ValueError(f"{case.id}: target {case.target!r} is not in the roster")
        if len({r.id for r in case.roster}) != case.roster_size:
            raise ValueError(f"{case.id}: roster ids must be unique")
        return case


def load_cases(suite: str) -> List[Case]:
    path = CASES_DIR / f"{suite}.jsonl"
    return [Case.parse(json.loads(line), suite) for line in path.read_text().splitlines() if line.strip()]
