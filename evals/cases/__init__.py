"""Routing cases: the typed model and the one loader for `cases/<suite>.jsonl`."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

CASES_DIR = Path(__file__).parent
SUITES = ("routing", "hard")

Kind = Literal["reuse", "create", "drift"]
HistoryRole = Literal["user", "assistant", "agent"]
_KINDS = ("reuse", "create", "drift")
_ROLES = ("user", "assistant", "agent")


@dataclass(frozen=True)
class HistoryEntry:
    role: HistoryRole
    text: str


@dataclass(frozen=True)
class Case:
    """One routing scenario.

    reuse: `target` is in the roster and must be called, with nothing spawned.
    create: nothing in the roster fits, so a new agent must be spawned.
    drift: turn 1 creates an agent and the last turn must come back to it.
    """

    id: str
    suite: str
    kind: Kind
    family: str
    target: Optional[str]
    roster: Tuple[str, ...]
    turns: Tuple[str, ...]
    history: Tuple[HistoryEntry, ...] = ()

    @property
    def roster_size(self) -> int:
        return len(self.roster)

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
        case = cls(
            id=raw["id"],
            suite=suite,
            kind=kind,
            family=raw.get("family") or kind,
            target=raw.get("target"),
            roster=tuple(raw["roster"]),
            turns=turns,
            history=history,
        )
        if case.roster_size != raw["roster_size"]:
            raise ValueError(f"{case.id}: roster has {case.roster_size} names, roster_size says {raw['roster_size']}")
        return case


def load_cases(suite: str) -> List[Case]:
    path = CASES_DIR / f"{suite}.jsonl"
    return [Case.parse(json.loads(line), suite) for line in path.read_text().splitlines() if line.strip()]
