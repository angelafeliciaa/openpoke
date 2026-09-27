# Routing evals

Measures the interaction agent's one decision that causes agent overload: given the roster and a
message, does it reuse the right execution agent, create one when it should, and come back to an
agent it created earlier?

## Run

```bash
.venv/bin/python -m pytest                                          # offline: replays every baseline
.venv/bin/python -m evals.run --mode live --model gpt-5 --cases hard  # paid: records what's missing
.venv/bin/python -m evals.report evals/results/<file>.jsonl
```

`--model` defaults to `EVAL_MODEL`, then to the production `interaction_agent_model`. Every live
call goes through OpenRouter (`OPENROUTER_API_KEY`); bare ids like `gpt-5-mini` are sent as
`openai/gpt-5-mini` with low reasoning effort. Anthropic calls cache the system prompt and roster.
Live mode reuses any recording it already has, so it only pays for the gaps.

## What pytest checks

- `test_replay.py` replays every `baselines/<model>/<suite>.json` and requires the exact same
  verdict for every case and trial. A missing recording fails; it never skips. It also fails
  when recordings exist that no case replays.
- `test_harness.py`: a replayed case never reaches the real execution manager or the real roster,
  and a missing recording is reported as one.
- `test_graders.py`: every grader outcome and the summary's denominators, on hand-built runs.

The pass/fail line is the baseline, not "every case passes". gpt-5-mini skips delegation on over
half its turns, and that is a measurement, not a broken build.

## After changing the prompt, tools, or cases

Recordings are keyed by the full request (system prompt, tool schemas, roster, history, model,
case, trial), so any change there makes replays miss and pytest names what's missing. Re-record
and refresh the baseline in one step, then review the baseline diff; it lists every case whose
verdict moved:

```bash
.venv/bin/python -m evals.run --mode live --model anthropic/claude-sonnet-4 --cases routing --update-baseline
```

`--update-baseline` also deletes that model's recordings for the suite that nothing replays.

## Files

| file | role |
|---|---|
| `cases/__init__.py` | typed `Case` and the one loader for `cases/<suite>.jsonl` |
| `cases/generate.py` | 15 reuse + 15 create scenarios at roster sizes 5, 50, 500 -> `routing.jsonl` |
| `cases/generate_hard.py` | paraphrase, trap, drift families at sizes 5 and 500 -> `hard.jsonl` |
| `harness.py` | sandbox (temp roster and logs, dispatch recorder), LLM record/replay, multi-turn |
| `graders.py` | code-only verdict per run |
| `run.py`, `report.py` | run a suite, write `results/*.jsonl` and baselines, print the table |
| `recordings/<model>/` | one JSON per LLM call: request and response |
| `baselines/<model>/<suite>.json` | the verdict for every case and trial that pytest holds replays to |

## Case kinds

- **reuse**: an agent that already did the work is in the roster. Pass = it is called and nothing new is spawned.
  `paraphrase` follow-ups share no words with the agent name; `trap` rosters add three near-duplicates.
- **create**: nothing matching exists. Pass = a new agent is spawned.
- **drift**: turn 1 creates an agent, turn 2 is a paraphrased follow-up. Pass = turn 2 reuses turn 1's agent.

## Reading the table

- **unsc**: runs that errored or had no recording. They are excluded from every rate except pass^k.
- **deleg**: share of scored runs that called `send_message_to_agent` at all. Below 100% is a
  prompt-compliance problem with the model, not a roster problem, so per-kind accuracy is computed
  among delegated runs only.
- **dup**: reuse runs that spawned a new agent when one existed.
- **pass^k**: cases where every trial passed. Consistency, not average.
- **names/req**: distinct names invented for the same request across trials. Above 1.0 is how rosters grow.
- **tokens**: prompt tokens on the first call, which is where the roster lives.

## Baseline findings (Sept 27, 2026; Sonnet hard set 3 trials, the rest 2)

- Sonnet 4, the production model, routed every delegated routing case correctly at 5, 50 and 500
  agents. What grew was the roster's cost: 3.4k -> 9.2k prompt tokens and 2.2x the price per turn at 500.
- On the hard set Sonnet was also right on every delegated case at both sizes; its only failures
  were 2 turns at 500 where it answered without delegating. The 500-agent half cost 3.9x the 5-agent half.
- Sonnet invented 1.2 to 1.5 distinct names per repeated request. That drift, not mis-routing on
  exact names, is the mechanism that fills the roster.
- gpt-5-mini delegated on only 32 to 47% of turns, and its reuse accuracy fell to 71% at 500 agents.
  Smaller models feel the roster first.
- gpt-5 on the hard set delegated 77 to 79% of the time, reused correctly on every delegated
  paraphrase and trap case, and came back to its own turn-1 agent on 67 to 71% of delegated drift cases.

## Limits

- After the first successful dispatch in a turn, the harness answers the next LLM call with a
  canned "On it." instead of paying for it. A model that dispatches a second agent only after
  seeing the first tool result is not measured.
- Each generator draws every roster from one seeded RNG, so inserting a scenario reshuffles the
  rosters after it and forces a re-record of those cases.
