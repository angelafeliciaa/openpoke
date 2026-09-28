"""Does this reply read as one assistant talking to the user, or does it show the machinery?

Poke is one persona. Behind it, work is delegated to execution agents with ids, tools, and a
prompt, and the user must never learn that. The prompt says so twice, and the model still slips:
"I see you have two dentist appointment agents" (Sonnet 4, once in 14 ambiguous cases, and
more often on smaller models). So the reply is checked before the user sees it, the way
Sierra's supervisors review each response as it is generated and NeMo's self-check output rail
asks a second model whether the reply should be shown.

Two layers, cheapest first:

1. A word check for "agent" or "agents" on their own. It is the leak we have actually seen,
   costs nothing, and cannot disagree with itself. Everyday compounds ("your travel agent") pass.
2. A judge call that reads the reply for meaning: does it mention agents, tools, prompts, ids,
   delegation, or how the assistant works, in any words? Synonyms are what the word check misses.

A trip does not drop the reply. The tool refuses with the reason, and the model rewrites once.
A second trip in the same turn is let through and logged, so a stubborn model cannot loop.
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Dict, Optional

Complete = Callable[..., Awaitable[Dict[str, Any]]]

_AGENT_WORD = re.compile(r"\bagents?\b", re.IGNORECASE)
# Ordinary English uses of the word, about people in the user's life rather than the machinery.
_EVERYDAY = re.compile(
    r"\b(travel|real estate|estate|insurance|booking|ticket(?:ing)?|listing|leasing|literary|sports|talent|"
    r"customs|secret|special|free|double|federal)\s+agents?\b",
    re.IGNORECASE,
)

JUDGE_SYSTEM = """You review a personal assistant's reply before the user sees it.

The assistant is one persona named Poke. Behind it, tasks are delegated to execution agents that have
ids, tools, and instructions, and the assistant follows a system prompt. The user must never learn any
of that.

LEAK means the reply refers to the assistant's own machinery or helpers: agents, sub-agents, assistants,
helpers, workers, teams, people or systems "handling" or "working on" the user's task, ids like a12,
tools, prompts, instructions, or what happens behind the scenes.
  LEAK: "Let me work with the agent handling that negotiation."
  LEAK: "I asked the teams handling both and will summarize their status."
  LEAK: "Do you mean your Tokyo flight (a446) or Sam's (a115)?"

OK means the reply speaks as one assistant about the user's own world: the user's people, employer,
team, accountant, landlord, recruiter, bookings, flights, hotels, threads, and what the assistant will do
or found out. Checking on something, following up, or drafting are normal assistant work, not a leak.
  OK: "Let me check for any reply from your accountant."
  OK: "I'll draft a sick day message for your team."
  OK: "Let me check on the Vercel response for you."
  OK: "You have two dentist appointments, yours and the kids'. Which one should I move?"

Answer with one word on the first line: OK or LEAK. After LEAK, add one short line saying what it
revealed."""


def names_the_machinery(text: str) -> Optional[str]:
    """The word check. Returns why the reply cannot be sent, or None."""
    scrubbed = _EVERYDAY.sub("", text)
    if _AGENT_WORD.search(scrubbed):
        return 'it says "agent" or "agents", which the user must never hear'
    return None


def _first_line(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or [{}]
    content = (choices[0].get("message") or {}).get("content") or ""
    return content.strip().splitlines()[0].strip() if content.strip() else ""


async def reveals_internals(text: str, complete: Complete, model: str) -> Optional[str]:
    """The judge. Returns why the reply cannot be sent, or None. An unreadable verdict passes."""
    response = await complete(model=model, messages=[{"role": "user", "content": text}], system=JUDGE_SYSTEM)
    verdict = _first_line(response)
    if verdict.upper().startswith("LEAK"):
        why = verdict.partition(":")[2].strip() or _second_line(response)
        return f"it reveals how the assistant works ({why})" if why else "it reveals how the assistant works"
    return None


def _second_line(response: Dict[str, Any]) -> str:
    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    return lines[1] if len(lines) > 1 else ""


async def check(text: str, complete: Complete, model: str) -> Optional[str]:
    """Why this reply cannot be shown to the user, or None. Word check first, then the judge."""
    return names_the_machinery(text) or await reveals_internals(text, complete, model)
