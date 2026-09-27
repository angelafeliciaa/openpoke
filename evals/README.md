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
- `server/tests/`: the roster, `send_message_to_agent` / `create_agent` / `search_agents`, BM25
  search, and the visible agent list, with no LLM.

The pass/fail line is the baseline, not "every case passes". gpt-5-mini asks the user instead of
routing on about half its turns, and that is a measurement, not a broken build.

## After changing the prompt, tools, or cases

Recordings are keyed by the full request (system prompt, tool schemas, roster, history, model,
case, trial), so any change there makes replays miss and pytest names what's missing. Re-record
and refresh the baseline in one step, then review the baseline diff; it lists every case whose
verdict moved:

```bash
.venv/bin/python -m evals.run --mode live --model anthropic/claude-sonnet-4 --cases routing --update-baseline
```

`--update-baseline` also deletes that model's recordings for the suite that nothing replays, and
stores the summary table in the baseline. To see what a change did, tag the commit before it and
print both tables side by side:

```bash
.venv/bin/python -m evals.compare stage-0 --model anthropic/claude-sonnet-4
```

## Files

| file | role |
|---|---|
| `cases/__init__.py` | typed `Case` and the one loader for `cases/<suite>.jsonl` |
| `cases/generate.py` | 15 reuse + 15 create scenarios at roster sizes 5, 50, 500 -> `routing.jsonl` |
| `cases/generate_hard.py` | paraphrase, trap, drift, ambiguous, sounds_new families at sizes 5 and 500 -> `hard.jsonl` |
| `harness.py` | sandbox (temp roster and logs, dispatch recorder), LLM record/replay, multi-turn |
| `graders.py` | code-only verdict per run |
| `run.py`, `report.py` | run a suite, write `results/*.jsonl` and baselines, print the table |
| `compare.py` | baseline summaries at a git ref next to the working tree's |
| `recordings/<model>/` | one JSON per LLM call: request and response |
| `baselines/<model>/<suite>.json` | the verdict for every case and trial that pytest holds replays to |

## Case kinds

- **reuse**: an agent that already did the work is in the roster. Pass = it is called and nothing new is spawned.
  `paraphrase` follow-ups share no words with the agent name; `trap` rosters add three near-duplicates.
- **create**: nothing matching exists. Pass = a new agent is spawned.
- **drift**: turn 1 creates an agent, turn 2 is a paraphrased follow-up. Pass = turn 2 reuses turn 1's agent.
- **ambiguous**: two agents fit (Email to Alice Park, Email to Alice Wong) and nothing in the
  message or history breaks the tie. Pass = the agent asks the user which one. Messaging one is
  `guessed_candidate`; messaging both is `messaged_every_candidate`, kept apart because it is harmless
  for "did the renewal go through?" and wrong for "tell alice i'm late".
- **sounds_new** (a reuse family): an agent owns the work, but the request reads as brand new and
  there is no history ("look into a place to stay in lisbon" with "Hotel in Lisbon" listed).
  Creating another agent here is how rosters bloat.

A turn that routes nothing and replies with a question is `asked_user`. It passes only on
ambiguous cases; everywhere else a question is friction for the user and fails. A question that
mentions agents is `asked_about_agents` and always fails: the user never sees agents.

## Reading the table

- **unsc**: runs that errored or had no recording. They are excluded from every rate except pass^k.
- **deleg**: share of scored runs that routed to an agent or asked the user. Below 100% means the
  model answered on its own, a prompt-compliance problem rather than a roster problem, so per-kind
  accuracy is computed among these runs only.
- **ask**: share of scored runs that asked the user instead of routing.
- **dup**: reuse runs that spawned a new agent when one existed.
- **pass^k**: cases where every trial passed. Consistency, not average.
- **names/req**: distinct names invented for the same request across trials. Above 1.0 is how rosters grow.
- **tokens**: prompt tokens on the first call, which is where the roster lives.

## Findings

Results for the original code (tag `stage-0`, Sonnet 4, gpt-5 and gpt-5-mini) and for each stage
of the fix are in [`AGENT_OVERLOAD.md`](../AGENT_OVERLOAD.md). The baselines in this tree are for
the current code; the original code's baselines and recordings live under `stage-0`.

## Limits

- After the first successful dispatch in a turn, the harness answers the next LLM call with a
  canned "On it." instead of paying for it. A model that dispatches a second agent only after
  seeing the first tool result is not measured.
- Each generator draws every roster from one seeded RNG, so inserting a scenario reshuffles the
  rosters after it and forces a re-record of those cases.
