"""Print recorded runs as chat transcripts: what the user said, what the model did, and the verdict.

  python -m evals.show amb-04@5 trap-03@500
  python -m evals.show --family ambiguous

Reads recordings and baselines only; never calls a model.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from typing import Any, Dict, List

from evals.cases import SUITES, Case, load_cases
from evals.harness import recordings_for
from evals.run import baseline_path
from server.agents.interaction_agent.reply_rail import JUDGE_SYSTEM

_ROLE = {"user": "User:", "assistant": "Poke:", "agent": "Agent report:"}
_EXPECTED = {
    "reuse": "send it to the existing agent '{target}'",
    "create": "no existing agent fits, so create a new one",
    "drift": "the last turn goes back to the agent turn 1 created",
}


_buffer: List[tuple] = []
"""(who, text) pairs of the transcript being built; `who` is "" for a plain note."""


def _say(who: str, text: str) -> None:
    _buffer.append((who, text))


def _print_lines(entries: List[tuple]) -> None:
    for who, text in entries:
        lines = textwrap.wrap(text, 90) or [""]
        print(f"  {who:<24}{lines[0]}")
        for line in lines[1:]:
            print(" " * 26 + line)


def _expected(case: Case) -> str:
    fits = ", ".join(case.candidates)
    if case.kind == "relay":
        return f"check every one of: {fits}, then tell the user each answer"
    if case.kind != "ambiguous":
        return _EXPECTED[case.kind].format(target=case.target)
    if case.read_only:
        answers = " and after checking, relay each answer" if case.reports else ""
        return f"ask, or check every one of: {fits}{answers}"
    return f"ask the user which one: {fits}"


def _is_judge(recording: Dict[str, Any]) -> bool:
    return recording["request"]["messages"][0].get("content") == JUDGE_SYSTEM


def _turn_calls(recordings: List[Dict[str, Any]], message: str, tag: str = "new_user_message") -> List[Dict[str, Any]]:
    """The calls of one turn, in order: each one's request carries every earlier step."""
    marker = f"<{tag}>\n{message}"
    calls = [r for r in recordings if not _is_judge(r) and marker in r["request"]["messages"][1]["content"]]
    return sorted(calls, key=lambda r: len(r["request"]["messages"]))


def _rail_verdict(recordings: List[Dict[str, Any]], reply: str) -> str:
    """What the reply rail's judge said about this exact reply, if it was asked."""
    for r in recordings:
        if _is_judge(r) and r["request"]["messages"][1]["content"] == reply:
            content = (r["response"]["choices"][0]["message"].get("content") or "").strip()
            return content.splitlines()[0] if content else "(no verdict)"
    return ""


def _print_call(n: int, call: Dict[str, Any], names: Dict[str, str], recordings: List[Dict[str, Any]] = ()) -> None:
    message = call["response"]["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    args = [json.loads(tc["function"]["arguments"] or "{}") for tc in tool_calls]
    spoken = {a.get("message", "").strip() for a in args}
    if (message.get("content") or "").strip() and message["content"].strip() not in spoken:
        _say(f"[call {n}] Poke:", message["content"])
    for tc, a in zip(tool_calls, args):
        tool = tc["function"]["name"]
        if tool == "send_message_to_user":
            _say(f"[call {n}] Poke:", a["message"])
            verdict = _rail_verdict(recordings, a["message"])
            if verdict:
                _say("  (rail says)", verdict)
        elif tool == "send_message_to_agent":
            _say(f"[call {n}] Poke -> agent:", f"[{names.get(a['agent_id'], a['agent_id'])}] {a['instructions']}")
        elif tool == "create_agent":
            confirmed = " (confirm_new)" if a.get("confirm_new") else ""
            _say(f"[call {n}] Poke creates:", f"'{a['name']}'{confirmed}: {a['instructions']}")
        else:
            _say(f"[call {n}] Poke uses:", f"{tool} {json.dumps(a)}")


def _tool_replies(call: Dict[str, Any], following: Dict[str, Any]) -> List[Dict[str, Any]]:
    new = following["request"]["messages"][len(call["request"]["messages"]):]
    return [json.loads(m["content"]) for m in new if m["role"] == "tool"]


def _print_tool_replies(call: Dict[str, Any], following: Dict[str, Any]) -> None:
    """Tool results that change the model's course: refusals (like similar agents) and search hits."""
    for reply in _tool_replies(call, following):
        if reply["status"] == "error":
            _say("  (tool refuses)", json.dumps(reply.get("error"))[:300])
        elif reply["tool"] == "search_agents":
            _say("  (search finds)", json.dumps(reply.get("result"))[:300])


def _created(call: Dict[str, Any], following: Dict[str, Any] | None) -> List[str]:
    """Names this call created: its create_agent calls the tool didn't refuse.

    The call after a dispatch is never recorded, so the id comes from the roster's counter, not a result.
    """
    tool_calls = call["response"]["choices"][0]["message"].get("tool_calls") or []
    asked = [json.loads(tc["function"]["arguments"] or "{}")["name"] for tc in tool_calls
             if tc["function"]["name"] == "create_agent"]
    replies = _tool_replies(call, following) if following else []
    refused = {r["arguments"].get("name") for r in replies if r["tool"] == "create_agent" and r["status"] == "error"}
    return [name for name in asked if name not in refused]


def transcript(case: Case, model: str, trial: int, verdicts: Dict[str, str]) -> Dict[str, Any]:
    """One case as a chat: header facts plus (who, text) entries, ending with the verdict."""
    _buffer.clear()
    names = {r.id: r.name for r in case.roster}
    header = {
        "id": case.id, "family": case.family, "roster_size": case.roster_size, "correct": _expected(case),
        "agents": [(r.id, r.name, r.description) for r in case.roster if r.name in case.candidates or r.name == case.target],
        "verdict": verdicts.get(f"{case.id}#{trial}", "(not in baseline)"),
    }
    # asking passes only when the case is ambiguous; everywhere else it is friction (see graders.py)
    header["passed"] = header["verdict"] in _PASSING or (header["verdict"] == "asked_user" and case.kind == "ambiguous")
    for entry in case.history:
        _say(_ROLE[entry.role], entry.text)

    recordings = [r for r in recordings_for(model, case.suite).records() if r["case"] == case.id and r["trial"] == trial]
    next_id = case.roster_size + 1
    for message in case.turns:
        _say("", "")
        _say("User:", message)
        calls = _turn_calls(recordings, message)
        if not calls:
            _say("", "(no recording)")
        for n, call in enumerate(calls, 1):
            following = calls[n] if n < len(calls) else None
            _print_call(n, call, names, recordings)
            if following:
                _print_tool_replies(call, following)
            for name in _created(call, following):
                names[f"a{next_id}"] = name
                next_id += 1
    if case.reports:
        # the relay turn's request quotes the batch payload, which starts with the first report's status line
        first_line = f"[SUCCESS] {case.reports[0].agent}"
        calls = _turn_calls(recordings, first_line, tag="new_agent_message")
        if calls:
            _say("", "")
            for r in case.reports:
                _say("Agent report:", f"{r.agent}: {r.says}")
            for n, call in enumerate(calls, 1):
                _print_call(n, call, names, recordings)
    return {**header, "entries": list(_buffer)}


def show(case: Case, model: str, trial: int, verdicts: Dict[str, str]) -> None:
    t = transcript(case, model, trial, verdicts)
    print(f"\n=== {t['id']}  ({t['family']}, {t['roster_size']} agents)")
    print(f"    correct: {t['correct']}")
    for agent_id, name, description in t["agents"]:
        print(f"    {agent_id} '{name}': {description}")
    _print_lines(t["entries"])
    print(f"\n    RESULT: {t['verdict']}")


def markdown(t: Dict[str, Any]) -> str:
    """One case as a Markdown section a reviewer can read on GitHub without running anything."""
    ok = "pass" if t["passed"] else "FAIL"
    out = [f"### `{t['id']}` · {t['family']}, {t['roster_size']} agents · **{ok}: {t['verdict']}**", "",
           f"*Correct:* {t['correct']}", ""]
    for agent_id, name, description in t["agents"]:
        out.append(f"- `{agent_id}` **{name}**: {description}")
    if t["agents"]:
        out.append("")
    for who, text in t["entries"]:
        if who == "":
            out.append("")
        elif who.startswith("  ("):
            out.append(f"> *{who.strip()}* {text}")
        else:
            out.append(f"**{who}** {text}  ")
    out.append("")
    return "\n".join(out)


_PASSING = {"reused_target", "created_new", "reused_turn1_agent", "checked_every_candidate", "relayed_every_report"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_ids", nargs="*")
    ap.add_argument("--family", default="", help="every case of one family, e.g. ambiguous")
    ap.add_argument("--model", default="anthropic/claude-sonnet-4")
    ap.add_argument("--trial", type=int, default=1)
    args = ap.parse_args()
    if not args.case_ids and not args.family:
        ap.error("give case ids or --family")

    cases = [c for suite in SUITES for c in load_cases(suite)]
    chosen = [c for c in cases if c.id in args.case_ids or c.family == args.family]
    verdicts: Dict[str, str] = {}
    for suite in SUITES:
        path = baseline_path(args.model, suite)
        if path.exists():
            verdicts.update(json.loads(path.read_text())["verdicts"])
    for case in chosen:
        show(case, args.model, args.trial, verdicts)


if __name__ == "__main__":
    main()
