"""Invariants the generators must hold, checked on the committed jsonl."""

from evals.cases import load_cases, SUITES
from evals.cases.generate import same_job


def test_filler_never_shares_a_job_with_the_target_or_candidates():
    """Random filler must not be the target's job under another name. A case grades one answer, so a
    second plausible owner has to be placed on purpose (trap, ambiguous), never drawn at random."""
    offenders = []
    for suite in SUITES:
        for case in load_cases(suite):
            protected = [case.target] if case.target else list(case.candidates)
            if not protected:
                continue
            for entry in case.roster:
                if entry.name in protected or entry.name in _placed_on_purpose(case):
                    continue
                if same_job(entry.name, protected):
                    offenders.append((case.id, entry.name))
    assert offenders == [], offenders


def _placed_on_purpose(case) -> set[str]:
    if case.family != "trap":
        return set()
    from evals.cases.generate_hard import SCENARIOS
    index = int(case.id.split("-")[1].split("@")[0]) - 1
    return set(SCENARIOS[index][3])
