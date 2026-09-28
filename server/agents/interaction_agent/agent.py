"""Interaction agent helpers for prompt construction."""

from html import escape
from pathlib import Path
from typing import Dict, List, Sequence

from ...services.execution import get_agent_roster, search
from ...services.execution.roster import AgentRecord

_prompt_path = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _prompt_path.read_text(encoding="utf-8").strip()

SHOW_ALL_UP_TO = 15
VISIBLE_MENTIONED = 10
VISIBLE_RECENT = 5
VISIBLE_MATCHES = 8


# Load and return the pre-defined system prompt from markdown file
def build_system_prompt() -> str:
    """Return the static system prompt for the interaction agent."""
    return SYSTEM_PROMPT


# Build structured message with conversation history, active agents, and current turn
def prepare_message_with_history(
    latest_text: str,
    transcript: str,
    message_type: str = "user",
) -> List[Dict[str, str]]:
    """Compose a message that bundles history, roster, and the latest turn."""
    sections: List[str] = []

    sections.append(_render_conversation_history(transcript))
    sections.append(f"<active_agents>\n{_render_active_agents(latest_text, transcript)}\n</active_agents>")
    sections.append(_render_current_turn(latest_text, message_type))

    content = "\n\n".join(sections)
    return [{"role": "user", "content": content}]


# Format conversation transcript into XML tags for LLM context
def _render_conversation_history(transcript: str) -> str:
    history = transcript.strip()
    if not history:
        history = "None"
    return f"<conversation_history>\n{history}\n</conversation_history>"


def _render_active_agents(latest_text: str = "", transcript: str = "") -> str:
    roster = get_agent_roster()
    roster.load()
    records = roster.records()

    if not records:
        return "None"

    visible = visible_agents(records, latest_text, transcript)
    rendered: List[str] = []
    for record in visible:
        attrs = f'id="{escape(record.id, quote=True)}" name="{escape(record.name, quote=True)}"'
        if record.description:
            rendered.append(f"<agent {attrs}>{escape(record.description, quote=False)}</agent>")
        else:
            rendered.append(f"<agent {attrs} />")
    hidden = len(records) - len(visible)
    if hidden:
        rendered.append(f"({hidden} more agents not listed; search_agents finds them by topic)")

    return "\n".join(rendered)


def visible_agents(records: Sequence[AgentRecord], latest_text: str, transcript: str) -> List[AgentRecord]:
    """The agents the next turn most likely needs, in roster order.

    Every agent the conversation mentions stays listed, so a follow-up never loses its agent
    to the cutoff; then the most recently used, then the best keyword matches for the message.
    Small rosters are listed whole.
    """
    if len(records) <= SHOW_ALL_UP_TO:
        return list(records)
    conversation = f"{transcript}\n{latest_text}".lower()
    mentioned = sorted(
        (r for r in records if _last_mention(r, conversation) >= 0),
        key=lambda r: -_last_mention(r, conversation),
    )[:VISIBLE_MENTIONED]
    by_recency = sorted(enumerate(records), key=lambda ir: (ir[1].last_used_at, ir[0]), reverse=True)
    recent = [r for _, r in by_recency[:VISIBLE_RECENT]]
    matches = search.top(records, latest_text, VISIBLE_MATCHES, search.recent_requests())
    chosen = {r.id for r in (*mentioned, *recent, *matches)}
    return [r for r in records if r.id in chosen]


def _last_mention(record: AgentRecord, conversation: str) -> int:
    return max(conversation.rfind(f"({record.id})"), conversation.rfind(record.name.lower()))


# Wrap the current message in appropriate XML tags based on sender type
def _render_current_turn(latest_text: str, message_type: str) -> str:
    tag = "new_agent_message" if message_type == "agent" else "new_user_message"
    body = latest_text.strip()
    return f"<{tag}>\n{body}\n</{tag}>"
