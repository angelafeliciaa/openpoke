"""Run one routing case against the real interaction agent inside a sandbox.

The sandbox points the roster and conversation logs at a temp dir, replaces the
execution batch manager with a recorder (routing is under test, not execution),
and serves LLM calls from `recordings/<model>/` so replays need no network.
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
from server.agents.execution_agent.runtime import ExecutionResult
from server.agents.interaction_agent import runtime as ia_runtime
from server.agents.interaction_agent import tools as ia_tools
from server.config import get_settings
from server.openrouter_client.client import OpenRouterBaseURL, _build_messages
from server.services.conversation import log as conv_log
from server.services.conversation.summarization import working_memory_log as wm_log
from server.services.execution import get_agent_roster
from server.services.execution import log_store as exec_logs
from server.services.execution import roster as roster_mod

Mode = Literal["live", "replay"]

RECORDINGS_DIR = Path(__file__).parent / "recordings"
MAX_TOKENS = 1024
OPENAI_BASE_URL = "https://api.openai.com/v1"
_TIMESTAMP = re.compile(r' timestamp=\\"[^"\\]*\\"')  # matches inside a json.dumps string
# USD per million tokens (input, cached input, output). OpenAI responses carry no cost field.
_PRICES = {
    "gpt-5-mini": (0.25, 0.025, 2.0),
    "gpt-5-nano": (0.05, 0.005, 0.4),
    "gpt-5": (1.25, 0.125, 10.0),
    "gpt-4.1-mini": (0.40, 0.10, 1.6),
}
# Returned instead of the post-dispatch LLM call: routing is decided by then, and the
# "on it" reply would cost a full roster-sized prompt per turn.
_WRAPUP = {"choices": [{"message": {"role": "assistant", "content": "On it."}}], "usage": {}}


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
    """Agents `send_message_to_agent` actually delivered to, in order."""
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
    missing_recording: Optional[str] = None
    recordings: List[Path] = field(default_factory=list)

    @property
    def first_prompt_tokens(self) -> int:
        calls = self.turns[0].calls if self.turns else []
        return calls[0].prompt_tokens if calls else 0

    @property
    def cost(self) -> float:
        return sum(c.cost for t in self.turns for c in t.calls)


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
    if "cost" in usage:
        return float(usage["cost"])
    if model not in _PRICES:
        return 0.0
    inp, cached_p, out = _PRICES[model]
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
    prompt = int(usage.get("prompt_tokens", 0)) - cached
    return (prompt * inp + cached * cached_p + int(usage.get("completion_tokens", 0)) * out) / 1e6


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
        if result.get("tool") == "send_message_to_agent" and result.get("status") == "success":
            return True
    return False


async def _live_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "/" in payload["model"]:
        url = f"{OpenRouterBaseURL.rstrip('/')}/chat/completions"
        key = os.environ["OPENROUTER_API_KEY"]
        body = {**payload, "max_tokens": MAX_TOKENS, "stream": False}
    else:
        url = f"{OPENAI_BASE_URL}/chat/completions"
        key = os.environ["OPENAI_API_KEY"]
        body = {**payload, "max_completion_tokens": MAX_TOKENS, "reasoning_effort": "low"}
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            url, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, json=body
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
        self.recordings = RECORDINGS_DIR / model
        self.missing: Optional[str] = None
        self.used: List[Path] = []
        self._manager = _RecordingBatchManager()
        self._calls: List[LLMCall] = []
        self._stack = AsyncExitStack()
        self._tasks_before: set = set()

    async def __aenter__(self) -> "Sandbox":
        tmp = Path(self._stack.enter_context(tempfile.TemporaryDirectory(prefix="openpoke-eval-")))
        settings = get_settings()
        roster = roster_mod.AgentRoster(tmp / "roster.json")
        for name in self.case.roster:
            roster.add_agent(name)

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
            patch.object(ia_tools, "_EXECUTION_BATCH_MANAGER", self._manager),
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

    def take_turn(self) -> tuple[List[str], List[LLMCall]]:
        dispatched, calls = self._manager.dispatched, self._calls
        self._manager.dispatched, self._calls = [], []
        return dispatched, calls

    async def _llm(self, *, model: str, messages, system=None, api_key=None, tools=None, base_url=None):
        if _dispatched_this_turn(messages):
            return _WRAPUP
        payload: Dict[str, Any] = {"model": self.model, "messages": _build_messages(messages, system)}
        if tools:
            payload["tools"] = tools
        path = self.recordings / f"{_recording_key(payload, self.case.id, self.trial)}.json"

        if path.exists():
            data = json.loads(path.read_text())["response"]
        elif self.mode == "live":
            data = await _live_call(payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"case": self.case.id, "trial": self.trial, "request": payload, "response": data}
            path.write_text(json.dumps(record, indent=1))
        else:
            self.missing = path.name
            raise _RecordingMissing(f"{self.case.id} trial {self.trial}: no recording at {path}")
        self.used.append(path)

        usage = data.get("usage") or {}
        self._calls.append(
            LLMCall(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                cost=_cost(self.model, usage),
            )
        )
        return data


def _seed_history(case: Case) -> None:
    log = conv_log.get_conversation_log()
    record = {"user": log.record_user_message, "assistant": log.record_reply, "agent": log.record_agent_message}
    for entry in case.history:
        record[entry.role](entry.text)


async def run_case(case: Case, trial: int, mode: Mode, model: str) -> CaseRun:
    """Run every turn of `case` in one sandbox; stops at the first error or missing recording."""
    run = CaseRun(case=case, trial=trial, model=model)
    async with Sandbox(case, trial, mode, model) as sb:
        _seed_history(case)
        for message in case.turns:
            before = set(get_agent_roster().get_agents())
            outcome = await ia_runtime.InteractionAgentRuntime().execute(message)
            await sb.settle()
            if sb.missing:
                run.missing_recording = sb.missing
                break
            dispatched, calls = sb.take_turn()
            run.turns.append(
                TurnResult(
                    dispatched=dispatched,
                    new_agents=[a for a in get_agent_roster().get_agents() if a not in before],
                    calls=calls,
                    response=outcome.response,
                    error=outcome.error,
                )
            )
            if outcome.error:
                break
        run.recordings = sb.used
    return run
