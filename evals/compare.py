"""Print each baseline's summary at a git ref next to the working tree's.

  python -m evals.compare stage-0                     # every model and suite
  python -m evals.compare stage-0 --model anthropic/claude-sonnet-4 --suite hard

Baselines are rewritten by `--update-baseline`, so a tag on the commit before a change
is all it takes to show that change's effect.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from evals.report import print_summary
from evals.run import BASELINES_DIR

_REPO = BASELINES_DIR.parent.parent


def _at_ref(ref: str, path: Path) -> Optional[Dict[str, Any]]:
    rel = path.relative_to(_REPO).as_posix()
    shown = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=_REPO, capture_output=True, text=True)
    return json.loads(shown.stdout) if shown.returncode == 0 else None


def _print(label: str, baseline: Optional[Dict[str, Any]]) -> None:
    print(f"\n  {label}")
    if baseline is None:
        print("    (no baseline)")
    elif "summary" not in baseline:
        print("    (baseline predates stored summaries)")
    else:
        print_summary(baseline["summary"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="git ref holding the 'before' baselines, e.g. stage-0")
    ap.add_argument("--model", default=None)
    ap.add_argument("--suite", default=None)
    args = ap.parse_args()

    for path in sorted(BASELINES_DIR.rglob("*.json")):
        current = json.loads(path.read_text())
        if args.model not in (None, current["model"]) or args.suite not in (None, current["suite"]):
            continue
        print(f"\n=== {current['model']} / {current['suite']}")
        _print(args.ref, _at_ref(args.ref, path))
        _print("working tree", current)


if __name__ == "__main__":
    main()
