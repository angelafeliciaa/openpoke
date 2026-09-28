from __future__ import annotations

from server.agents.interaction_agent.agent import SHOW_ALL_UP_TO, VISIBLE_RECENT, visible_agents
from server.services.execution.roster import AgentRecord
from server.services.execution.search import rank, tokenize

OLD = "2026-01-01T00:00:00"


def _record(i: int, name: str, description: str = "", last_used: str = OLD) -> AgentRecord:
    return AgentRecord(f"a{i}", name, description, OLD, last_used)


def _roster(n: int) -> list[AgentRecord]:
    return [_record(i, f"Reminder {i}") for i in range(1, n + 1)]


def test_tokenize_drops_stopwords_and_single_characters_and_folds_plurals() -> None:
    assert tokenize("Book a table for 2 at the Hotels in Lisbon") == ["book", "table", "hotel", "lisbon"]


def test_rank_puts_the_owner_first_and_skips_non_matches() -> None:
    records = [_record(1, "Flight to Lisbon"), _record(2, "Hotel in Lisbon", "Hotel booking in Lisbon."), _record(3, "Gym")]

    ranked = rank(records, "find me a hotel in lisbon")

    assert [r.id for r, _ in ranked] == ["a2", "a1"]


def test_small_rosters_are_listed_whole() -> None:
    records = _roster(SHOW_ALL_UP_TO)

    assert visible_agents(records, "anything", "") == records


def test_large_rosters_keep_mentioned_recent_and_matching_agents_only() -> None:
    records = _roster(200)
    records[9] = _record(10, "Hotel in Lisbon", "Hotel booking in Lisbon.")
    records[19] = _record(20, "Email to Alice")
    records[29] = _record(30, "Dentist", last_used="2026-09-01T00:00:00")

    shown = {r.id for r in visible_agents(records, "does the lisbon place have parking?", "Email to Alice (a20): Sent.")}

    assert {"a10", "a20", "a30"} <= shown
    assert len(shown) <= 1 + VISIBLE_RECENT + 8 + 1
