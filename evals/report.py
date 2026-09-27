"""Results rows and the per-roster-size summary table.

  python -m evals.report evals/results/<file>.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from evals.graders import Verdict
from evals.harness import CaseRun

Row = Dict[str, Any]
_UNSCORED = ("error", "missing_recording")
_ASKED = ("asked_user", "asked_about_agents")


def to_row(run: CaseRun, verdict: Verdict) -> Row:
    case = run.case
    last = run.turns[-1] if run.turns else None
    return {
        "case_id": case.id,
        "suite": case.suite,
        "kind": case.kind,
        "family": case.family,
        "roster_size": case.roster_size,
        "message": case.turns[0],
        "trial": run.trial,
        "model": run.model,
        "pass": verdict.passed,
        "reason": verdict.reason,
        "error": run.missing_recording or (last.error if last else None),
        "turns": [
            {
                "dispatched": t.dispatched,
                "new_agents": t.new_agents,
                "llm_calls": len(t.calls),
                "cost": round(sum(c.cost for c in t.calls), 5),
            }
            for t in run.turns
        ],
        "first_prompt_tokens": run.first_prompt_tokens,
        "cost": round(run.cost, 5),
        "response": last.response if last else "",
    }


def summarize(rows: List[Row]) -> List[Dict[str, Any]]:
    by_size: Dict[int, List[Row]] = defaultdict(list)
    for r in rows:
        by_size[r["roster_size"]].append(r)

    out = []
    for size in sorted(by_size):
        rs = by_size[size]
        scored = [r for r in rs if r["reason"] not in _UNSCORED]
        # a clarifying question is a routing decision too; only doing nothing is a compliance failure
        delegated = [r for r in scored if r["reason"] != "no_delegation"]
        reuse = [r for r in delegated if r["kind"] == "reuse"]

        trials_by_case: Dict[str, List[bool]] = defaultdict(list)
        for r in rs:
            trials_by_case[r["case_id"]].append(bool(r["pass"]))

        tokens = [r["first_prompt_tokens"] for r in rs if r["first_prompt_tokens"]]
        out.append({
            "roster_size": size,
            "runs": len(rs),
            "unscored": len(rs) - len(scored),
            "delegation_rate": _ratio(len(delegated), len(scored)),
            "ask_rate": _ratio(sum(1 for r in scored if r["reason"] in _ASKED), len(scored)),
            # routing accuracy is conditional on delegating at all; no_delegation is a model-compliance failure
            "by_kind": {k: _pass_rate([r for r in delegated if r["kind"] == k]) for k in sorted({r["kind"] for r in rs})},
            "names_per_request": _names_per_request(rs),
            "dup_rate": _ratio(
                sum(1 for r in reuse if r["reason"] in ("spawned_duplicate", "reused_target_but_also_spawned")), len(reuse)
            ),
            "pass_all_trials": _ratio(sum(all(v) for v in trials_by_case.values()), len(trials_by_case)),
            "mean_prompt_tokens": _ratio(sum(tokens), len(tokens)),
            "cost_usd": sum(r["cost"] for r in rs),
            "failures": _failures(rs),
        })
    return out


def _ratio(num: float, den: int) -> float:
    return num / den if den else float("nan")


def _pass_rate(rows: List[Row]) -> float:
    return _ratio(sum(1 for r in rows if r["pass"]), len(rows))


def _names_per_request(rows: List[Row]) -> float:
    """Mean distinct agent names invented for the same request across trials (create, and drift turn 1)."""
    names: Dict[str, set] = defaultdict(set)
    for r in rows:
        if r["kind"] in ("create", "drift") and r["turns"] and r["turns"][0]["new_agents"]:
            names[r["message"]].update(n.lower() for n in r["turns"][0]["new_agents"])
    return _ratio(sum(len(v) for v in names.values()), len(names))


def _failures(rows: List[Row]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for r in rows:
        if not r["pass"]:
            counts[r["reason"]] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def json_safe(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """NaN (an empty denominator) becomes null so the summary is valid JSON."""

    def clean(v: Any) -> Any:
        if isinstance(v, float):
            return None if v != v else round(v, 4)
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        return v

    return [clean(s) for s in summary]


def _pct(x: float | None) -> str:
    return "-" if x is None or x != x else f"{x:.0%}"


def print_summary(summary: List[Dict[str, Any]]) -> None:
    kinds = sorted({k for s in summary for k in s["by_kind"]})
    head = " ".join(f"{k[:9]:>9}" for k in kinds)
    print(f"{'roster':>6} {'runs':>4} {'unsc':>4} {'deleg':>5} {'ask':>4} {head} {'dup':>5} {'pass^k':>6} {'names/req':>9} "
          f"{'tokens':>7} {'cost':>7}  failures")
    print("        (deleg = routed or asked; per-kind accuracy is among those runs; pass^k is over all runs;\n"
          "         unsc = error or no recording)")
    for s in summary:
        cols = " ".join(f"{_pct(s['by_kind'].get(k)):>9}" for k in kinds)
        npr = s["names_per_request"]
        npr = float("nan") if npr is None else npr
        print(f"{s['roster_size']:>6} {s['runs']:>4} {s['unscored']:>4} {_pct(s['delegation_rate']):>5} "
              f"{_pct(s.get('ask_rate')):>4} {cols} "
              f"{_pct(s['dup_rate']):>5} {_pct(s['pass_all_trials']):>6} {'-' if npr != npr else f'{npr:.1f}':>9} "
              f"{s['mean_prompt_tokens'] or 0:>7.0f} ${s['cost_usd']:>6.3f}  {s['failures']}")


if __name__ == "__main__":
    path = Path(sys.argv[1])
    print_summary(summarize([json.loads(line) for line in path.read_text().splitlines() if line.strip()]))
