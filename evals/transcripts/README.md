# Eval transcripts

Every case as a chat: what the user said, what Poke did, what the tools refused, the verdict.
One file per family per model. Regenerate with `python -m evals.transcripts`.

## anthropic/claude-sonnet-4

| family | pass | file |
|---|---|---|
| reuse | 45/45 | [reuse.md](claude-sonnet-4/reuse.md) |
| create | 45/45 | [create.md](claude-sonnet-4/create.md) |
| paraphrase | 30/30 | [paraphrase.md](claude-sonnet-4/paraphrase.md) |
| trap | 30/30 | [trap.md](claude-sonnet-4/trap.md) |
| drift | 20/20 | [drift.md](claude-sonnet-4/drift.md) |
| ambiguous | 12/14 | [ambiguous.md](claude-sonnet-4/ambiguous.md) |
| sounds_new | 20/20 | [sounds_new.md](claude-sonnet-4/sounds_new.md) |
| relay | 10/10 | [relay.md](claude-sonnet-4/relay.md) |
| long | 20/20 | [long.md](claude-sonnet-4/long.md) |
| stress | 8/10 | [stress.md](claude-sonnet-4/stress.md) |
