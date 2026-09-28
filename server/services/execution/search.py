"""BM25 over agent records: what an agent is called, what it owns, and what it was last asked.

Keyword scoring is deterministic, needs no model call, and handles the common case where the
user names the thing ("the lisbon hotel"). Paraphrases with no shared words are what it misses;
the conversation-mention and recency rules in the visible set cover the usual ones.

Scores are only comparable within one query and roster (they grow with roster size), so callers
use the ranking, never a fixed score cutoff.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, List, Sequence, Tuple

from .roster import AgentRecord

_K1 = 1.2
_B = 0.75
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can could do for from get go going have he her him his how i if im in is it its "
    "me my of on or our out please she so that the their them then there they this to up us was we what when "
    "where which who will with would you your".split()
)


def tokenize(text: str) -> List[str]:
    tokens = []
    for token in _TOKEN.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def rank(
    records: Sequence[AgentRecord],
    query: str,
    extra_text: Callable[[AgentRecord], str] = lambda _: "",
) -> List[Tuple[AgentRecord, float]]:
    """Records with a positive score, best first; ties keep roster order."""
    docs = [tokenize(f"{r.name} {r.name} {r.description} {extra_text(r)}") for r in records]
    terms = set(tokenize(query))
    if not docs or not terms:
        return []
    n = len(docs)
    avg_len = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d) if t in terms)
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in df}

    scored = []
    for index, (record, doc) in enumerate(zip(records, docs)):
        tf = Counter(doc)
        score = sum(
            idf[t] * tf[t] * (_K1 + 1) / (tf[t] + _K1 * (1 - _B + _B * len(doc) / avg_len))
            for t in idf
            if tf[t]
        )
        if score > 0:
            scored.append((-score, index, record))
    scored.sort()
    return [(record, -neg) for neg, _, record in scored]


def top(records: Sequence[AgentRecord], query: str, k: int, extra_text=lambda _: "") -> List[AgentRecord]:
    return [r for r, _ in rank(records, query, extra_text)[:k]]


def recent_requests(limit: int = 3) -> Callable[[AgentRecord], str]:
    """What each agent was last asked, from its execution log."""
    from .log_store import get_execution_agent_logs

    logs = get_execution_agent_logs()

    def text(record: AgentRecord) -> str:
        requests = [payload for tag, _, payload in logs.iter_entries(record.name) if tag == "agent_request"]
        return " ".join(requests[-limit:])

    return text
