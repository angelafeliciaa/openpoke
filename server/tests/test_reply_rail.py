"""The reply rail: the word check, the judge, and the one-rewrite rule in the runtime."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from server.agents.interaction_agent import reply_rail
from server.agents.interaction_agent import runtime as ia_runtime
from server.agents.interaction_agent import tools
from server.services.conversation import log as conv_log
from server.services.execution import roster as roster_mod


def test_the_word_check_catches_the_leak_we_saw() -> None:
    assert reply_rail.names_the_machinery(
        "I see you have two dentist appointment agents - one for the kids and one for yourself. Which one?"
    )


@pytest.mark.parametrize("text", [
    "Your travel agent emailed back about the Tokyo flight.",
    "The real estate agent can show the flat on Tuesday.",
    "Two dentist appointments are booked. Which one should I move?",
])
def test_everyday_uses_and_clean_replies_pass_the_word_check(text: str) -> None:
    assert reply_rail.names_the_machinery(text) is None


def _judge_saying(*lines: str):
    calls: List[Dict[str, Any]] = []

    async def complete(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "\n".join(lines)}}]}

    return complete, calls


async def test_the_judge_reads_meaning_and_passes_its_reason_back() -> None:
    complete, calls = _judge_saying("LEAK", "it mentions the helper that handles bookings")

    reason = await reply_rail.check("I'll ask my booking helper to move it.", complete, "judge-model")

    assert reason == "it reveals how the assistant works (it mentions the helper that handles bookings)"
    assert calls[0]["model"] == "judge-model" and calls[0]["system"] == reply_rail.JUDGE_SYSTEM


async def test_an_ok_verdict_lets_the_reply_through() -> None:
    complete, _ = _judge_saying("OK")

    assert await reply_rail.check("Moved your dentist appointment to Friday.", complete, "m") is None


async def test_the_word_check_runs_before_the_judge() -> None:
    complete, calls = _judge_saying("OK")

    assert await reply_rail.check("Two agents are on it.", complete, "m")
    assert calls == []


@pytest.fixture
def scripted_runtime(tmp_path, monkeypatch):
    """A runtime whose LLM answers from a script, with logs and roster in a temp dir."""
    monkeypatch.setattr(roster_mod, "_agent_roster", roster_mod.AgentRoster(tmp_path / "roster.json"))
    conversation = conv_log.ConversationLog(tmp_path / "conversation.log")
    monkeypatch.setattr(conv_log, "_conversation_log", conversation)
    settings = ia_runtime.get_settings()
    monkeypatch.setattr(settings, "conversation_summary_threshold", 0)
    monkeypatch.setattr(settings, "openrouter_api_key", "test")

    script: List[Dict[str, Any]] = []
    seen: List[Dict[str, Any]] = []

    def say(text: str) -> Dict[str, Any]:
        call = {"id": f"c{len(script)}", "type": "function",
                "function": {"name": "send_message_to_user", "arguments": json.dumps({"message": text})}}
        return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [call]}}]}

    async def fake_llm(**kwargs):
        seen.append(kwargs)
        if kwargs.get("system") == reply_rail.JUDGE_SYSTEM:
            return {"choices": [{"message": {"content": "OK"}}]}
        return script.pop(0)

    monkeypatch.setattr(ia_runtime, "request_chat_completion", fake_llm)
    return script, say, seen, conversation


async def test_a_leaking_reply_is_sent_back_once_and_the_rewrite_is_what_the_user_sees(scripted_runtime) -> None:
    script, say, seen, conversation = scripted_runtime
    script += [
        say("I see two dentist appointment agents. Which one?"),
        say("You have two dentist appointments, yours and the kids'. Which one should I move?"),
        {"choices": [{"message": {"role": "assistant", "content": ""}}]},
    ]

    result = await ia_runtime.InteractionAgentRuntime().execute("move the dentist appointment to friday")

    assert result.response == "You have two dentist appointments, yours and the kids'. Which one should I move?"
    replies = [text for tag, _, text in conversation.iter_entries() if tag == "poke_reply"]
    assert replies == [result.response]
    refusal = next(m for m in seen[1]["messages"] if m["role"] == "tool")
    assert "Not sent" in refusal["content"] and "agent" in refusal["content"]


async def test_a_second_trip_in_the_same_turn_goes_through(scripted_runtime) -> None:
    script, say, _, conversation = scripted_runtime
    script += [
        say("Two agents are on it."),
        say("Both agents are on it."),
        {"choices": [{"message": {"role": "assistant", "content": ""}}]},
    ]

    result = await ia_runtime.InteractionAgentRuntime().execute("check both")

    assert result.response == "Both agents are on it."
    assert [t for t, _, _ in conversation.iter_entries() if t == "poke_reply"] == ["poke_reply"]
