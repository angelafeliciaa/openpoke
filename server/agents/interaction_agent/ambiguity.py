"""Is the agent the model picked the only listed agent that fits what the user just said?

The prompt asks the model to check with the user when two agents fit equally. It guesses
anyway on about half of those turns, so the send tool asks the same question in code before it
dispatches: score the user's own words (not the model's rewritten instructions) against the
listed agents, and if another one fits about as well and nothing tells them apart, refuse and
hand both back.

What tells them apart is words. "does the paris hotel have late checkout?" uses only the words
"Hotel in Paris" and "Hotel in Paris for Mom" share, so it is ambiguous. "move the kids dentist
appointment" carries a word only one agent has, and "the tokyo flight and the lisbon hotel"
names both, one question across two agents. Neither of those is a guess.
"""

from __future__ import annotations

from typing import List, Sequence, Set

from ...services.execution import search
from ...services.execution.roster import AgentRecord
from .agent import _last_mention, visible_agents

CLOSE = 0.8
"""Another agent within this fraction of the chosen agent's score fits about as well."""


def _words(record: AgentRecord) -> Set[str]:
    return set(search.tokenize(f"{record.name} {record.description}"))


def close_alternatives(
    records: Sequence[AgentRecord],
    chosen: AgentRecord,
    latest_text: str,
    transcript: str,
    also_sent_to: Set[str] = frozenset(),
) -> List[AgentRecord]:
    """Listed agents that fit the user's message as well as `chosen` with nothing to tell them
    apart. Empty means dispatch. `also_sent_to` holds ids this turn is messaging anyway."""
    if _last_mention(chosen, transcript.lower()) >= 0:
        return []
    listed = visible_agents(records, latest_text, transcript)
    scores = {r.id: s for r, s in search.rank(listed, latest_text)}
    own = scores.get(chosen.id, 0.0)
    if own <= 0:
        return []  # nothing in the message points at the chosen agent; the model chose from context
    said = set(search.tokenize(latest_text))
    chosen_words = _words(chosen)
    alternatives = []
    for other in listed:
        if other.id == chosen.id or other.id in also_sent_to or scores.get(other.id, 0.0) < CLOSE * own:
            continue
        other_words = _words(other)
        names_chosen = said & (chosen_words - other_words)
        names_other = said & (other_words - chosen_words)
        if names_chosen or names_other:
            continue
        alternatives.append(other)
    return alternatives
