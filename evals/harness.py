"""Run one routing case against the real interaction agent inside a sandbox.

The sandbox points the roster and conversation logs at a temp dir, replaces the
execution batch manager with a recorder (routing is under test, not execution),
and serves LLM calls from `recordings/<model>/<suite>.jsonl` so replays need no network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from unittest.mock import patch

import httpx

from evals.cases import Case
from server.agents.execution_agent.batch_manager import ExecutionBatchManager
from server.agents.execution_agent.runtime import ExecutionResult
from server.agents.interaction_agent import runtime as ia_runtime
from server.agents.interaction_agent import tools as ia_tools
from server.config import get_settings
from server.openrouter_client.client import OpenRouterBaseURL, _build_messages
from server.services.conversation import log as conv_log
from server.services.conversation.summarization import summarizer as summarizer_mod
from server.services.conversation.summarization import working_memory_log as wm_log
from server.services.execution import get_agent_roster
from server.services.execution import log_store as exec_logs
from server.services.execution import roster as roster_mod

Mode = Literal["live", "replay"]

RECORDINGS_DIR = Path(__file__).parent / "recordings"


class Recordings:
    """Every recorded LLM call for one model and suite, one JSON line each, keyed by request hash.

    One file per model per suite keeps a suite's recordings browsable (lines are in run order,
    each carries its case and trial) and keeps a re-record to a diff of lines, not of files.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._by_key: Optional[Dict[str, Dict[str, Any]]] = None

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self._by_key is None:
            self._by_key = {}
            if self.path.exists():
                for line in self.path.read_text().splitlines():
                    if line.strip():
                        record = json.loads(line)
                        self._by_key[record["key"]] = record
        return self._by_key

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._load().get(key)

    def put(self, record: Dict[str, Any]) -> None:
        self._load()[record["key"]] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def keys(self) -> List[str]:
        return list(self._load())

    def records(self) -> List[Dict[str, Any]]:
        return list(self._load().values())

    def prune(self, keep: set) -> int:
        """Rewrite the file with only `keep`, in their existing order. Returns how many were dropped."""
        kept = [r for r in self.records() if r["key"] in keep]
        dropped = len(self._load()) - len(kept)
        if dropped:
            self._by_key = {r["key"]: r for r in kept}
            self.path.write_text("".join(json.dumps(r) + "\n" for r in kept))
        return dropped


_open: Dict[Path, Recordings] = {}


def recordings_for(model: str, suite: str) -> Recordings:
    path = RECORDINGS_DIR / model / f"{suite}.jsonl"
    if path not in _open:
        _open[path] = Recordings(path)
    return _open[path]
MAX_TOKENS = 1024
_TIMESTAMP = re.compile(r' timestamp=\\"[^"\\]*\\"')  # matches inside a json.dumps string
# Returned instead of the post-dispatch LLM call: routing is decided by then, and the
# closing reply would cost a full roster-sized prompt per turn. Empty, so whatever the model
# already said this turn stays its reply, in the result and in the conversation log.
# Cases that script reports need every dispatch, and some models dispatch one agent per
# response, so those turns run to the model's own end instead.
_WRAPUP = {"choices": [{"message": {"role": "assistant", "content": ""}}], "usage": {}}
_DISPATCH_TOOLS = ("send_message_to_agent", "create_agent")
_SEEDED_AT = "2026-09-01T09:00:00"


class _Clock:
    """The roster's clock inside the sandbox: one second per turn, so `last_used_at` and the
    recency order it drives are the same on every run. Two dispatches in one turn tie, as they
    do live within a second; the wall clock made the relay-turn prompt flip when they straddled one."""

    def __init__(self) -> None:
        self.turn = 0

    def tick(self) -> None:
        self.turn += 1

    def now(self) -> str:
        return f"2026-09-01T09:00:{self.turn:02d}"


def default_model() -> str:
    """`EVAL_MODEL`, else the model production uses. Ids with a vendor prefix go to OpenRouter."""
    return os.environ.get("EVAL_MODEL") or get_settings().interaction_agent_model


@dataclass(frozen=True)
class LLMCall:
    prompt_tokens: int
    completion_tokens: int
    cost: float


@dataclass
class TurnResult:
    dispatched: List[str]
    """Agents `send_message_to_agent` or `create_agent` actually delivered to, in order."""
    new_agents: List[str]
    calls: List[LLMCall]
    response: str
    error: Optional[str]


@dataclass
class CaseRun:
    case: Case
    trial: int
    model: str
    turns: List[TurnResult] = field(default_factory=list)
    relay: Optional[TurnResult] = None
    """The turn after every candidate reported back, when the case scripts reports. Its `response`
    is everything the user was told in that turn, not only the last message."""
    missing_recording: Optional[str] = None
    recordings: List[str] = field(default_factory=list)
    """Keys of every recording this run replayed or made."""

    @property
    def first_prompt_tokens(self) -> int:
        calls = self.turns[0].calls if self.turns else []
        return calls[0].prompt_tokens if calls else 0

    @property
    def cost(self) -> float:
        return sum(c.cost for t in self.turns + ([self.relay] if self.relay else []) for c in t.calls)


class _RecordingMissing(RuntimeError):
    pass


class _RecordingBatchManager:
    """Stands in for ExecutionBatchManager: notes who was dispatched, runs nothing."""

    def __init__(self) -> None:
        self.dispatched: List[str] = []

    async def execute_agent(self, agent_name: str, instructions: str, request_id: Optional[str] = None) -> ExecutionResult:
        self.dispatched.append(agent_name)
        return ExecutionResult(agent_name=agent_name, success=True, response="(stubbed)")


def _cost(model: str, usage: Dict[str, Any]) -> float:
    return float(usage.get("cost", 0.0))  # OpenRouter reports the price of every call


def _recording_key(payload: Dict[str, Any], case_id: str, trial: int) -> str:
    raw = json.dumps({"case": case_id, "trial": trial, **payload}, sort_keys=True)
    raw = _TIMESTAMP.sub("", raw)  # conversation-log timestamps must not change the key
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _dispatched_this_turn(messages: List[Dict[str, Any]]) -> bool:
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        if result.get("tool") in _DISPATCH_TOOLS and result.get("status") == "success":
            return True
    return False


def _with_cache_breakpoints(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mark the system prompt and the first user message as cacheable prefixes.

    Anthropic caches only up to explicit breakpoints. The system prompt is shared by
    every call; the first user message holds the roster, which repeats across trials.
    """
    marked: List[Dict[str, Any]] = []
    user_marked = False
    for message in messages:
        content = message.get("content")
        is_prefix = message["role"] == "system" or (message["role"] == "user" and not user_marked)
        if is_prefix and isinstance(content, str) and content:
            user_marked = user_marked or message["role"] == "user"
            message = {**message, "content": [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]}
        marked.append(message)
    return marked


def _openrouter_body(payload: Dict[str, Any]) -> Dict[str, Any]:
    body = {**payload, "max_tokens": MAX_TOKENS, "stream": False}
    if payload["model"].startswith("anthropic/"):
        body["messages"] = _with_cache_breakpoints(payload["messages"])
    return body


async def _live_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{OpenRouterBaseURL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"},
            json=_openrouter_body(payload),
        )
        if response.status_code >= 400:
            raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
        return response.json()


class Sandbox:
    """Temp-dir state, a dispatch recorder, and LLM record/replay for one case trial."""

    def __init__(self, case: Case, trial: int, mode: Mode, model: str) -> None:
        self.case = case
        self.trial = trial
        self.mode = mode
        self.model = model
        self.store = recordings_for(model, case.suite)
        self.missing: Optional[str] = None
        self.used: List[str] = []
        self._manager = _RecordingBatchManager()
        self._clock = _Clock()
        self._wrapup_after_dispatch = not case.reports
        self._calls: List[LLMCall] = []
        self._stack = AsyncExitStack()
        self._tasks_before: set = set()

    async def __aenter__(self) -> "Sandbox":
        tmp = Path(self._stack.enter_context(tempfile.TemporaryDirectory(prefix="openpoke-eval-")))
        settings = get_settings()
        roster_file = tmp / "roster.json"
        roster_file.write_text(json.dumps({
            "next_id": self.case.roster_size + 1,
            "agents": [
                {"id": r.id, "name": r.name, "description": r.description,
                 "created_at": _SEEDED_AT, "last_used_at": _SEEDED_AT}
                for r in self.case.roster
            ],
        }))
        roster = roster_mod.AgentRoster(roster_file)

        self._stack.enter_context(
            patch.object(wm_log, "_working_memory_log", wm_log.WorkingMemoryLog(tmp / "working_memory.log"))
        )
        # ConversationLog binds the working-memory log when constructed, so build it after that patch.
        conversation = conv_log.ConversationLog(tmp / "conversation.log")

        patches = [
            patch.object(conv_log, "_conversation_log", conversation),
            patch.object(roster_mod, "_agent_roster", roster),
            patch.object(exec_logs, "_execution_agent_logs", exec_logs.ExecutionAgentLogStore(tmp / "execution_agents")),
            patch.object(ia_runtime, "request_chat_completion", self._llm),
            patch.object(summarizer_mod, "request_chat_completion", self._llm),
            patch.object(ia_tools, "_EXECUTION_BATCH_MANAGER", self._manager),
            patch.object(roster_mod, "_now", self._clock.now),
            patch.object(settings, "conversation_summary_threshold", 0),
        ]
        if not settings.openrouter_api_key:
            # The runtime refuses to start without one; every LLM call is intercepted anyway.
            patches.append(patch.object(settings, "openrouter_api_key", "unused-by-evals"))
        for p in patches:
            self._stack.enter_context(p)
        self._stack.push_async_callback(self.settle)
        self._tasks_before = asyncio.all_tasks()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self._stack.__aexit__(*exc_info)

    async def settle(self) -> None:
        """Run the dispatch tasks `send_message_to_agent` scheduled.

        They look up the batch manager only when they start, so they must finish
        while the recorder is still patched in, or they reach the real one.
        """
        current = asyncio.current_task()
        while pending := asyncio.all_tasks() - self._tasks_before - {current}:
            await asyncio.gather(*pending, return_exceptions=True)

    def tick(self) -> None:
        self._clock.tick()

    async def summarize(self) -> None:
        """Compress the seeded history the way production does after 100 messages, then leave
        summarisation on so the turn reads the summary plus the tail instead of the full log."""
        settings = get_settings()
        self._stack.enter_context(patch.object(settings, "conversation_summary_threshold", 100))
        try:
            while await summarizer_mod.summarize_conversation():
                pass  # one pass per 100 lines, as production would have done over time
        except _RecordingMissing:
            pass

    def take_turn(self) -> tuple[List[str], List[LLMCall]]:
        dispatched, calls = self._manager.dispatched, self._calls
        self._manager.dispatched, self._calls = [], []
        return dispatched, calls

    async def _llm(self, *, model: str, messages, system=None, api_key=None, tools=None, base_url=None):
        if self._wrapup_after_dispatch and _dispatched_this_turn(messages):
            return _WRAPUP
        payload: Dict[str, Any] = {"model": self.model, "messages": _build_messages(messages, system)}
        if tools:
            payload["tools"] = tools
        key = _recording_key(payload, self.case.id, self.trial)

        recorded = self.store.get(key)
        if recorded is not None:
            data = recorded["response"]
        elif self.mode == "live":
            data = await _live_call(payload)
            self.store.put({"key": key, "case": self.case.id, "trial": self.trial, "request": payload, "response": data})
        else:
            self.missing = key
            raise _RecordingMissing(f"{self.case.id} trial {self.trial}: no recording {key} in {self.store.path}")
        self.used.append(key)

        usage = data.get("usage") or {}
        self._calls.append(
            LLMCall(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                cost=_cost(self.model, usage),
            )
        )
        return data


def _roster_names() -> List[str]:
    return [r.name for r in get_agent_roster().records()]


def _seed_history(case: Case) -> None:
    log = conv_log.get_conversation_log()
    record = {"user": log.record_user_message, "assistant": log.record_reply, "agent": log.record_agent_message}
    for entry in case.history:
        record[entry.role](entry.text)


def _replies_so_far() -> int:
    return sum(1 for tag, _, _ in conv_log.get_conversation_log().iter_entries() if tag == "poke_reply")


def _told_the_user(since: int) -> str:
    """Every reply recorded after `since`, joined. The runtime's own `response` keeps only the last."""
    replies = [text for tag, _, text in conv_log.get_conversation_log().iter_entries() if tag == "poke_reply"]
    return "\n".join(replies[since:])


def scripted_reports(case: Case, dispatched: List[str]) -> str:
    """The batch payload production would build once the dispatched agents finished, in dispatch order."""
    says = {r.agent: r.says for r in case.reports}
    results = [ExecutionResult(agent_name=name, success=True, response=says[name]) for name in dispatched if name in says]
    return ExecutionBatchManager()._format_batch_payload(results)


async def run_case(case: Case, trial: int, mode: Mode, model: str) -> CaseRun:
    """Run every turn of `case` in one sandbox; stops at the first error or missing recording."""
    run = CaseRun(case=case, trial=trial, model=model)
    async with Sandbox(case, trial, mode, model) as sb:
        _seed_history(case)
        if case.summarize:
            await sb.summarize()
            if sb.missing:
                run.missing_recording = sb.missing
                return run
        for message in case.turns:
            sb.tick()
            before = set(_roster_names())
            outcome = await ia_runtime.InteractionAgentRuntime().execute(message)
            await sb.settle()
            if sb.missing:
                run.missing_recording = sb.missing
                break
            dispatched, calls = sb.take_turn()
            run.turns.append(
                TurnResult(
                    dispatched=dispatched,
                    new_agents=[a for a in _roster_names() if a not in before],
                    calls=calls,
                    response=outcome.response,
                    error=outcome.error,
                )
            )
            if outcome.error:
                break
        if run.turns and _every_candidate_checked(case, run.turns[-1]):
            run.relay = await _relay_reports(case, run.turns[-1].dispatched, sb)
            if sb.missing:
                run.missing_recording = sb.missing
        run.recordings = sb.used
    return run


def _every_candidate_checked(case: Case, last: TurnResult) -> bool:
    return bool(case.reports) and not last.error and set(case.candidates) <= set(last.dispatched)


async def _relay_reports(case: Case, dispatched: List[str], sb: Sandbox) -> TurnResult:
    """Feed the agents' scripted answers through the real agent-message path and keep what the user heard."""
    sb.tick()
    replies_before = _replies_so_far()
    before = set(_roster_names())
    outcome = await ia_runtime.InteractionAgentRuntime().handle_agent_message(scripted_reports(case, dispatched))
    await sb.settle()
    dispatched_again, calls = sb.take_turn()
    return TurnResult(
        dispatched=dispatched_again,
        new_agents=[a for a in _roster_names() if a not in before],
        calls=calls,
        response=_told_the_user(replies_before),
        error=outcome.error,
    )
