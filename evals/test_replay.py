"""Replay every committed baseline and require the exact same verdict for every case and trial.

Replays are deterministic, so any difference means the harness, the grader, or the
agent's request changed. A prompt or tool change makes recordings miss; re-record with
`python -m evals.run --mode live --model <model> --cases <suite> --update-baseline`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.cases import load_cases
from evals.run import BASELINES_DIR, run_suite, unused_recordings, verdict_map

BASELINES = sorted(BASELINES_DIR.rglob("*.json"))


@pytest.mark.parametrize("path", BASELINES, ids=[str(p.relative_to(BASELINES_DIR)) for p in BASELINES])
async def test_replay_matches_baseline(path: Path) -> None:
    baseline = json.loads(path.read_text())
    graded = await run_suite(load_cases(baseline["suite"]), baseline["trials"], "replay", baseline["model"])
    actual = verdict_map(graded)
    expected = baseline["verdicts"]

    missing = sorted(k for k, v in actual.items() if v == "missing_recording")
    assert not missing, f"{len(missing)} case-trials have no recording, first: {missing[:5]}"
    changed = {k: f"{expected.get(k)} -> {actual.get(k)}" for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k)}
    assert not changed, f"{len(changed)} verdicts changed: {dict(sorted(changed.items())[:20])}"
    stale = unused_recordings(baseline["model"], baseline["suite"], graded)
    assert not stale, f"{len(stale)} recordings are never replayed; --update-baseline prunes them"


def test_every_suite_has_a_baseline() -> None:
    assert {json.loads(p.read_text())["suite"] for p in BASELINES} == {"routing", "hard"}
