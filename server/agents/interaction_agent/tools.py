"""Tool definitions for interaction agent."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional

from ...logging_config import logger
from ...services.conversation import get_conversation_log
from ...services.execution import get_agent_roster, get_execution_agent_logs, search
from ...services.execution.roster import AgentRecord
from ..execution_agent.batch_manager import ExecutionBatchManager

SIMILAR_SHOWN = 3
SEARCH_RESULTS = 5


@dataclass
class ToolResult:
    """Standardized payload returned by interaction-agent tools."""

    success: bool
    payload: Any = None
    user_message: Optional[str] = None
    recorded_reply: bool = False

# Tool schemas for OpenRouter
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "send_message_to_agent",
            "description": "Send instructions to an existing execution agent. It keeps its history, so follow-ups on work it did (replying on the same thread, changing a booking, checking for an update) belong with it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The id of an agent listed in <active_agents>, e.g. 'a12'.",
                    },
                    "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                },
                "required": ["agent_id", "instructions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_agent",
            "description": "Start a new execution agent for work that no existing agent owns, and send it its first instructions. If existing agents look similar, nothing is created and they are returned instead; reuse one, or call again with confirm_new if the work is really different.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short name for the work, e.g. 'Vercel Job Offer'. Must differ from every existing agent's name.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One line on what this agent owns, specific enough to tell it apart from similar agents, e.g. 'Start-date negotiation with the Vercel recruiter'.",
                    },
                    "instructions": {"type": "string", "description": "Instructions for the agent to execute."},
                    "confirm_new": {
                        "type": "boolean",
                        "description": "Set only after a previous create_agent call returned similar agents and none of them owns this work.",
                    },
                },
                "required": ["name", "description", "instructions"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_agents",
            "description": "Find existing agents by topic when the one you need is not listed in <active_agents>. Searches every agent's name, description, and recent instructions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Words the agent's work would involve, e.g. 'lisbon hotel booking'."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message_to_user",
            "description": "Deliver a natural-language response directly to the user. Use this for updates, confirmations, or any assistant response the user should see immediately.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Plain-text message that will be shown to the user and recorded in the conversation log.",
                    },
                },
                "required": ["message"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_draft",
            "description": "Record an email draft so the user can review the exact text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email for the draft.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject for the draft.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content (plain text).",
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait silently when a message is already in conversation history to avoid duplicating responses. Adds a <wait> log entry that is not visible to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why waiting (e.g., 'Message already sent', 'Draft already created').",
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]

_EXECUTION_BATCH_MANAGER = ExecutionBatchManager()


def send_message_to_agent(agent_id: str, instructions: str) -> ToolResult:
    """Dispatch to an existing agent; an unknown id is an error, never a new agent."""
    roster = get_agent_roster()
    roster.load()
    record = roster.get(agent_id)
    if record is None:
        return ToolResult(
            success=False,
            payload={"error": f"No agent has id {agent_id!r}. Use an id from <active_agents>, or create_agent for new work."},
        )
    roster.touch(record.id)
    return _dispatch(record, instructions, created=False)


def create_agent(name: str, description: str, instructions: str, confirm_new: bool = False) -> ToolResult:
    """Register a new agent and dispatch its first instructions, unless existing agents look like the same work.

    Search supplies the candidates and the model makes the call: a keyword match alone can't
    tell "Hotel in Lisbon" owning a Lisbon stay from "Reminder 2" matching "table for 2".
    """
    roster = get_agent_roster()
    roster.load()
    existing = roster.find_by_name(name)
    if existing is not None:
        return ToolResult(
            success=False,
            payload={
                "error": f"Agent {existing.id} is already named {existing.name!r}. "
                "Message it with send_message_to_agent, or choose a name for different work."
            },
        )
    if not confirm_new:
        similar = search.top(
            roster.records(), f"{name} {description} {instructions}", SIMILAR_SHOWN, search.recent_requests()
        )
        if similar:
            return ToolResult(
                success=False,
                payload={
                    "error": "Nothing created: these existing agents may already own this work. If one does, "
                    "use send_message_to_agent with its id. If none does, call create_agent again with confirm_new: true.",
                    "similar_agents": [_summary(r) for r in similar],
                },
            )
    return _dispatch(roster.create(name, description), instructions, created=True)


def search_agents(query: str) -> ToolResult:
    """Every agent is searchable, including the ones <active_agents> leaves out."""
    roster = get_agent_roster()
    roster.load()
    found = search.top(roster.records(), query, SEARCH_RESULTS, search.recent_requests())
    if not found:
        return ToolResult(success=True, payload={"agents": [], "note": "No agent matches; create_agent if this is new work."})
    return ToolResult(success=True, payload={"agents": [_summary(r) for r in found]})


def _summary(record: AgentRecord) -> dict:
    return {"id": record.id, "name": record.name, "description": record.description}


def _dispatch(record: AgentRecord, instructions: str, created: bool) -> ToolResult:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.error("No running event loop available for async execution")
        return ToolResult(success=False, payload={"error": "No event loop available"})

    get_execution_agent_logs().record_request(record.name, instructions)
    logger.info(f"{'Created' if created else 'Reused'} agent {record.id}: {record.name}")

    async def _execute_async() -> None:
        try:
            result = await _EXECUTION_BATCH_MANAGER.execute_agent(record.name, instructions)
            status = "SUCCESS" if result.success else "FAILED"
            logger.info(f"Agent '{record.name}' completed: {status}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error(f"Agent '{record.name}' failed: {str(exc)}")

    loop.create_task(_execute_async())

    return ToolResult(
        success=True,
        payload={"status": "submitted", "agent_id": record.id, "agent_name": record.name, "new_agent_created": created},
    )


# Send immediate message to user and record in conversation history
def send_message_to_user(message: str) -> ToolResult:
    """Record a user-visible reply in the conversation log."""
    log = get_conversation_log()
    log.record_reply(message)

    return ToolResult(
        success=True,
        payload={"status": "delivered"},
        user_message=message,
        recorded_reply=True,
    )


# Format and record email draft for user review
def send_draft(
    to: str,
    subject: str,
    body: str,
) -> ToolResult:
    """Record a draft update in the conversation log for the interaction agent."""
    log = get_conversation_log()

    message = f"To: {to}\nSubject: {subject}\n\n{body}"

    log.record_reply(message)
    logger.info(f"Draft recorded for: {to}")

    return ToolResult(
        success=True,
        payload={
            "status": "draft_recorded",
            "to": to,
            "subject": subject,
        },
        recorded_reply=True,
    )


# Record silent wait state to avoid duplicate responses
def wait(reason: str) -> ToolResult:
    """Wait silently and add a wait log entry that is not visible to the user."""
    log = get_conversation_log()
    
    # Record a dedicated wait entry so the UI knows to ignore it
    log.record_wait(reason)
    

    return ToolResult(
        success=True,
        payload={
            "status": "waiting",
            "reason": reason,
        },
        recorded_reply=True,
    )


# Return predefined tool schemas for LLM function calling
def get_tool_schemas():
    """Return OpenAI-compatible tool schemas."""
    return TOOL_SCHEMAS


# Route tool calls to appropriate handlers with argument validation and error handling
def handle_tool_call(name: str, arguments: Any) -> ToolResult:
    """Handle tool calls from interaction agent."""
    try:
        if isinstance(arguments, str):
            args = json.loads(arguments) if arguments.strip() else {}
        elif isinstance(arguments, dict):
            args = arguments
        else:
            return ToolResult(success=False, payload={"error": "Invalid arguments format"})

        if name == "send_message_to_agent":
            return send_message_to_agent(**args)
        if name == "create_agent":
            return create_agent(**args)
        if name == "search_agents":
            return search_agents(**args)
        if name == "send_message_to_user":
            return send_message_to_user(**args)
        if name == "send_draft":
            return send_draft(**args)
        if name == "wait":
            return wait(**args)

        logger.warning("unexpected tool", extra={"tool": name})
        return ToolResult(success=False, payload={"error": f"Unknown tool: {name}"})
    except json.JSONDecodeError:
        return ToolResult(success=False, payload={"error": "Invalid JSON"})
    except TypeError as exc:
        return ToolResult(success=False, payload={"error": f"Missing required arguments: {exc}"})
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("tool call failed", extra={"tool": name, "error": str(exc)})
        return ToolResult(success=False, payload={"error": "Failed to execute"})
