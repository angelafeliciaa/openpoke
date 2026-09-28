from __future__ import annotations

import json
from pathlib import Path

import pytest

from server.services.execution import log_store
from server.services.execution.roster import AgentRoster


def test_ids_are_assigned_in_order_and_survive_a_reload(tmp_path: Path) -> None:
    roster = AgentRoster(tmp_path / "roster.json")
    alice = roster.create("Email to Alice", "Invoice thread with Alice.")
    tokyo = roster.create("Flight to Tokyo", "Booking a flight to Tokyo.")

    reloaded = AgentRoster(tmp_path / "roster.json")

    assert (alice.id, tokyo.id) == ("a1", "a2")
    assert reloaded.records() == [alice, tokyo]
    assert reloaded.create("Hotel in Lisbon", "").id == "a3"


def test_a_name_that_differs_only_in_case_or_spacing_is_taken(tmp_path: Path) -> None:
    roster = AgentRoster(tmp_path / "roster.json")
    roster.create("Email to Alice", "")

    with pytest.raises(ValueError):
        roster.create("email  to alice", "")
    assert roster.find_by_name("EMAIL TO ALICE").id == "a1"


def test_touch_moves_only_that_agents_last_used(tmp_path: Path) -> None:
    stale = "2020-01-01T00:00:00"
    agents = [{"id": i, "name": i, "description": "", "created_at": stale, "last_used_at": stale} for i in ("a1", "a2")]
    (tmp_path / "roster.json").write_text(json.dumps({"next_id": 3, "agents": agents}))
    roster = AgentRoster(tmp_path / "roster.json")

    roster.touch("a1")

    assert roster.get("a1").last_used_at > stale
    assert roster.get("a2").last_used_at == stale


def test_legacy_name_list_converts_once_using_each_agents_log(tmp_path: Path, monkeypatch) -> None:
    logs = log_store.ExecutionAgentLogStore(tmp_path / "logs")
    monkeypatch.setattr(log_store, "_execution_agent_logs", logs)
    logs.record_request("Email to Alice", "Find   alice's email about the invoice")
    logs.record_agent_response("Email to Alice", "Found it.")
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(["Email to Alice", "Flight to Tokyo"]))

    roster = AgentRoster(path)

    alice, tokyo = roster.records()
    assert (alice.id, alice.name, alice.description) == ("a1", "Email to Alice", "Find alice's email about the invoice")
    assert alice.created_at <= alice.last_used_at
    assert (tokyo.id, tokyo.description) == ("a2", "")
    assert json.loads(path.read_text())["next_id"] == 3
