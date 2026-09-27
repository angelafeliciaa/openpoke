"""Smoke test: the server package imports and the roster round-trips through a temp file."""

from pathlib import Path

from server.services.execution.roster import AgentRoster


def test_roster_roundtrip(tmp_path: Path) -> None:
    roster = AgentRoster(tmp_path / "roster.json")
    roster.add_agent("email-alice")
    roster.add_agent("email-alice")
    reloaded = AgentRoster(tmp_path / "roster.json")
    assert reloaded.get_agents() == ["email-alice"]
