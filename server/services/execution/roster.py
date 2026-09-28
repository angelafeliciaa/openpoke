"""Execution-agent roster: one record per agent, persisted to roster.json.

The interaction agent refers to agents by `id`, which the roster assigns and never reuses, so a
mistyped or invented reference fails instead of silently creating a new agent. `name` stays the
storage key for execution logs and triggers, so it is frozen once created.
"""

import fcntl
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

from ...logging_config import logger

_DESCRIPTION_CHARS = 160


@dataclass(frozen=True)
class AgentRecord:
    id: str
    name: str
    description: str
    created_at: str
    last_used_at: str


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _norm(name: str) -> str:
    return " ".join(name.lower().split())


class AgentRoster:
    """Records in creation order, reloaded from disk before every read that matters."""

    def __init__(self, roster_path: Path):
        self._roster_path = roster_path
        self._records: list[AgentRecord] = []
        self._next_id = 1
        self.load()

    def load(self) -> None:
        self._records, self._next_id = [], 1
        if not self._roster_path.exists():
            return
        try:
            data = json.loads(self._roster_path.read_text())
        except Exception as exc:
            logger.warning(f"Failed to load roster.json: {exc}")
            return
        if isinstance(data, list):
            self._records = [_from_legacy_name(str(name), i) for i, name in enumerate(data, 1)]
            self._next_id = len(self._records) + 1
            self.save()
            logger.info(f"Converted roster.json to records ({len(self._records)} agents)")
            return
        self._records = [AgentRecord(**r) for r in data.get("agents", [])]
        self._next_id = int(data.get("next_id", len(self._records) + 1))

    def save(self) -> None:
        """Save roster.json with file locking."""
        body = {"next_id": self._next_id, "agents": [asdict(r) for r in self._records]}
        max_retries = 5
        retry_delay = 0.1

        for attempt in range(max_retries):
            try:
                self._roster_path.parent.mkdir(parents=True, exist_ok=True)

                with open(self._roster_path, 'w') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        json.dump(body, f, indent=2)
                        return
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            except BlockingIOError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.warning("Failed to acquire lock on roster.json after retries")
            except Exception as exc:
                logger.warning(f"Failed to save roster.json: {exc}")
                break

    def records(self) -> list[AgentRecord]:
        return list(self._records)

    def get(self, agent_id: str) -> Optional[AgentRecord]:
        return next((r for r in self._records if r.id == agent_id), None)

    def find_by_name(self, name: str) -> Optional[AgentRecord]:
        """Names are compared case- and whitespace-insensitively; log files already collide that way."""
        wanted = _norm(name)
        return next((r for r in self._records if _norm(r.name) == wanted), None)

    def create(self, name: str, description: str) -> AgentRecord:
        if self.find_by_name(name):
            raise ValueError(f"an agent named {name!r} already exists")
        now = _now()
        record = AgentRecord(f"a{self._next_id}", name.strip(), description.strip(), now, now)
        self._next_id += 1
        self._records.append(record)
        self.save()
        return record

    def touch(self, agent_id: str) -> None:
        self._records = [replace(r, last_used_at=_now()) if r.id == agent_id else r for r in self._records]
        self.save()

    def clear(self) -> None:
        self._records, self._next_id = [], 1
        try:
            if self._roster_path.exists():
                self._roster_path.unlink()
            logger.info("Cleared agent roster")
        except Exception as exc:
            logger.warning(f"Failed to clear roster.json: {exc}")


def _from_legacy_name(name: str, index: int) -> AgentRecord:
    """The first request an agent received is the best record of what it was created for."""
    from .log_store import get_execution_agent_logs

    entries = list(get_execution_agent_logs().iter_entries(name))
    first_request = next((payload for tag, _, payload in entries if tag == "agent_request"), "")
    created = next((ts for _, ts, _ in entries if ts), "") or _now()
    last_used = next((ts for _, ts, _ in reversed(entries) if ts), "") or created
    description = " ".join(first_request.split())[:_DESCRIPTION_CHARS]
    return AgentRecord(f"a{index}", name, description, created, last_used)


_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_ROSTER_PATH = _DATA_DIR / "execution_agents" / "roster.json"

_agent_roster = AgentRoster(_ROSTER_PATH)


def get_agent_roster() -> AgentRoster:
    """Get the singleton roster instance."""
    return _agent_roster
