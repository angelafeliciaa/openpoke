"""Run routing cases, write results, optionally refresh the committed baseline.

  python -m evals.run --mode live --cases hard --model anthropic/claude-sonnet-4
  python -m evals.run --mode replay --cases routing --update-baseline

Results land in evals/results/<timestamp>-<suite>-<mode>.jsonl; see evals.report.
Baselines live in evals/baselines/<model>/<suite>.json and are what pytest replays against.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from evals.cases import SUITES, Case, load_cases
from evals.graders import Verdict, grade
from evals.harness import CaseRun, Mode, default_model, recordings_for, run_case
from evals.report import json_safe, print_summary, summarize, to_row

RESULTS_DIR = Path(__file__).parent / "results"
BASELINES_DIR = Path(__file__).parent / "baselines"
TRIALS = 2

Graded = Tuple[CaseRun, Verdict]


async def run_suite(
    cases: List[Case], trials: int, mode: Mode, model: str, on_result: Optional[Callable[[Graded], None]] = None
) -> List[Graded]:
    graded: List[Graded] = []
    for case in cases:
        for trial in range(1, trials + 1):
            run = await run_case(case, trial, mode, model)
            result = (run, grade(run))
            graded.append(result)
            if on_result:
                on_result(result)
    return graded


def verdict_map(graded: List[Graded]) -> Dict[str, str]:
    return {f"{run.case.id}#{run.trial}": verdict.reason for run, verdict in graded}


def baseline_path(model: str, suite: str) -> Path:
    return BASELINES_DIR / model / f"{suite}.json"


def unused_recordings(model: str, suite: str, graded: List[Graded]) -> List[str]:
    """Recording keys for this model and suite that no replayed case asked for: stale after a prompt change."""
    used = {k for run, _ in graded for k in run.recordings}
    return sorted(k for k in recordings_for(model, suite).keys() if k not in used)


def write_baseline(model: str, suite: str, trials: int, graded: List[Graded]) -> Path:
    """Verdicts are what pytest checks; the summary is what `evals.compare` diffs across commits."""
    path = baseline_path(model, suite)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "model": model,
        "suite": suite,
        "trials": trials,
        "summary": json_safe(summarize([to_row(run, v) for run, v in graded])),
        "verdicts": dict(sorted(verdict_map(graded).items())),
    }
    path.write_text(json.dumps(body, indent=1) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["live", "replay"], default="replay")
    ap.add_argument("--model", default=default_model(), help="OpenAI id, or vendor/model for OpenRouter")
    ap.add_argument("--cases", choices=SUITES, default="routing")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--sizes", default="", help="comma list, e.g. 5,50,500")
    ap.add_argument("--kinds", default="", help="comma list: reuse,create,drift,ambiguous")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--update-baseline", action="store_true", help="write baselines/<model>/<suite>.json")
    args = ap.parse_args()

    sizes = {int(s) for s in args.sizes.split(",") if s}
    kinds = {k for k in args.kinds.split(",") if k}
    if args.update_baseline and (sizes or kinds or args.limit):
        ap.error("--update-baseline needs the whole suite; drop --sizes/--kinds/--limit")

    cases = [
        c for c in load_cases(args.cases)
        if (not sizes or c.roster_size in sizes) and (not kinds or c.kind in kinds)
    ][: args.limit]

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.cases}-{args.mode}.jsonl"
    rows = []
    with out.open("w") as fh:

        def on_result(result: Graded) -> None:
            run, verdict = result
            row = to_row(run, verdict)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            called = run.turns[-1].dispatched if run.turns else []
            print(f"{run.case.id:<14} t{run.trial} {'PASS' if verdict.passed else 'FAIL':<4} {verdict.reason:<32} "
                  f"tokens={run.first_prompt_tokens:<6} agents={called}")

        graded = asyncio.run(run_suite(cases, args.trials, args.mode, args.model, on_result))

    print(f"\n{args.model}: wrote {len(rows)} rows to {out}\n")
    print_summary(summarize(rows))
    unscored = [f"{run.case.id}#{run.trial}" for run, v in graded if v.reason in ("error", "missing_recording")]
    if args.update_baseline and unscored:
        raise SystemExit(f"\nnot updating the baseline: {len(unscored)} runs errored or had no recording, "
                         f"first: {unscored[:5]}")
    if args.update_baseline:
        print(f"\nbaseline: {write_baseline(args.model, args.cases, args.trials, graded)}")
        used = {k for run, _ in graded for k in run.recordings}
        print(f"pruned {recordings_for(args.model, args.cases).prune(used)} recordings no case uses")


if __name__ == "__main__":
    main()
