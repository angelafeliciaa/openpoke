# Agent overload in OpenPoke

OpenPoke's interaction agent hands every task to a named execution agent and keeps those agents
around, so a follow-up ("any update from vercel?") can go back to the agent that holds the thread.
After weeks of use a user has hundreds of them. This branch changes how the interaction agent finds,
reuses, and creates agents so that a large roster stays cheap and routes correctly, and it adds the
evals that measure whether it does.

## The short version

- **Root cause:** the agent's name was doing three jobs at once. It was the identity the model
  had to retype exactly, the only description of what the agent owned, and the storage key. Every
  prompt also listed every agent.
- **What changed:**
  - Agents get system-assigned ids, and creating an agent is its own tool that requires a
    one-line description.
  - The prompt lists only the ~14 agents the turn likely needs; the rest are reachable through
    search.
  - Creating an agent first checks for similar ones.
  - The model is told to ask only when two agents genuinely fit.
- **Result (Sonnet 4, the production model):**
  - Prompt size at 500 agents fell from 9.2k tokens to 4.2k, the same as at 5 agents.
  - Routing stayed at 100% on every clear-cut family, and the one duplicate the old code created
    is gone.
  - Ambiguous requests improved but are not solved: 7 of 14 pass, against 8 of 42 runs before.
    The 7 that still fail are listed one by one in [To do](#to-do).
- **How we know:** 204 routing cases across 7 case families and roster sizes 5, 50 and 500, run
  against the real interaction agent with recorded LLM responses. Grading is plain code, each
  stage is diffed against a tagged "before", and `python -m evals.show <case>` prints any run as a
  chat transcript.

## What was wrong

The interaction agent routes with one tool, `send_message_to_agent(agent_name, instructions)`.
In the original code (tag `stage-0`, `server/agents/interaction_agent/tools.py`):

1. **An unknown name creates an agent silently.** The tool compared the name to the roster with an
   exact string match and created a new agent on any miss. "Email to Alice" and "email to alice"
   are two agents. The model decides whether to reuse or create by typing a name, and a typo
   decides it for it.
2. **The name is the only description.** `<active_agents>` rendered `<agent name="..."/>` and
   nothing else. With both "Email to Alice Park" and "Email to Alice Wong" listed, "tell alice
   i'm late" can only be answered by a guess.
3. **Every agent is in every prompt.** `_render_active_agents` listed the whole roster on every
   turn. Measured on Sonnet 4 (the production model), the first prompt grows from 3.4k tokens
   with 5 agents to 9.2k with 500, and a 500-agent turn costs 2.2x as much. The model pays to read
   hundreds of agents it will never use, and past about 100 candidates tool-selection accuracy is
   known to drop ([RAG-MCP](https://arxiv.org/abs/2505.03275)).
4. **The name is also the storage key.** Execution logs (`log_store.py`) and triggers
   (`triggers.agent_name`) are keyed by it, so an agent can't be renamed without migrating both.

The mechanism that bloats a roster is 1 plus 2. When the model can't tell from the list that an
agent already owns the work, or types its name slightly differently, a new agent is born. The
"before" runs show Sonnet inventing 1.2 to 1.8 distinct names for the same request across trials.
Each of those names would be a separate agent in real use.

## What we built, and why

Each stage is a separate commit so its effect can be measured on its own.

### Stage 1: ids, a separate `create_agent`, descriptions (`c32437d`)

- **`send_message_to_agent(agent_id, instructions)`.** The roster assigns ids (`a1`, `a2`, ...) and
  never reuses them. An unknown id, or a name where an id belongs, returns an error and creates
  nothing.
  - *Why ids and not strict name matching:* a strict match turns every capitalisation slip into
    an error-and-retry round trip. An id is short, can't be misspelled into a different valid
    agent, and makes "which agent?" a lookup instead of a string comparison.
  - This is the same split LangGraph makes between a `thread_id` and its metadata.
- **`create_agent(name, description, instructions)`.** Creating is now a deliberate act with its
  own tool, and it requires a line saying what the agent owns ("Start-date negotiation with the
  Vercel recruiter"). A taken name returns the existing agent's id.
  - *Why:* reusing and creating used to be the same call, told apart only by whether the name
    happened to match. The description is what lets the model, and later the search, tell
    similar agents apart.
- **The name stays the storage key, frozen at creation.** Logs, triggers and the execution agent
  still key on it, so none of them had to migrate. `roster.json` now stores records (id, name,
  description, created and last-used times). An old name list converts itself once, taking each
  description from the agent's first logged instruction.
- **Agent reports carry the id** (`[SUCCESS] Email to Alice (a4): Sent.`), so a follow-up can be
  routed to the agent that reported.
- **Ask only when it's genuinely ambiguous.** The prompt says to ask a short question, in the
  user's terms and never naming agents, when two agents fit equally and nothing in the
  conversation breaks the tie. It says not to ask when the choice is clear, and never to ask for
  details an agent could find itself.
  - *Why this rule:* guessing between two Alices sends the wrong email. Asking on every turn is
    its own failure. The "before" runs show both: Sonnet guessed on 34 of 42 ambiguous runs, and
    gpt-5-mini asked on over half of all turns.

### Stage 3: a short visible list and `search_agents` (`6f8614a`)

- **`<active_agents>` lists at most ~14 agents:**
  - every agent the conversation mentions, found by its id or name in the transcript;
  - the 5 most recently used;
  - the 8 best keyword matches for the current message.

  It ends with one line: "(487 more agents not listed; search_agents finds them by topic)".
  Rosters of 15 or fewer are listed whole.
  - *Why these three sources:*
    - Follow-ups refer to something in the conversation, so conversation mentions keep them safe.
    - Recency covers "keep going".
    - Matching covers "the lisbon hotel" months later.
  - *Why a count line instead of hiding agents:* nothing is unreachable; the model knows more
    exist. This follows Anthropic's tool-search pattern, which cut tool-definition tokens by 85%
    and raised selection accuracy.
- **`search_agents(query)`** searches every agent's name, description and last three instructions.
- **Search is BM25**, about 60 lines in `server/services/execution/search.py`, with no dependency.
  - *Why not embeddings first:* keyword scoring is deterministic (so it can be tested and
    replayed), costs nothing per message, and needs no index to keep in sync. It misses
    paraphrases with no shared words. The evals decide when that matters; see limits.

### Stage 2: check for similar agents before creating (same commit)

- **`create_agent` searches first.** If any agents match, it creates nothing and returns the top
  3 as `similar_agents`. The model reuses one, or calls again with `confirm_new: true`.
  - *Why the model decides and not a score cutoff:* BM25 scores grow with roster size. A real
    duplicate among 5 agents scored 2 to 6, while a genuinely new request among 500 scored 9 on
    noise ("table for 2" matching "Reminder 2"). No fixed threshold separates them, but the
    ranking was right every time: the true owner ranked first for all 20 duplicate cases.
  - So search supplies candidates and the model judges, as Mem0 does for memory writes.
  - *Why it matters more after stage 3:* once the list is short, the owner of some work may not
    be on it. Without the check, a shorter prompt would trade tokens for duplicates.

## How we tested it

The evals live in `evals/`, and `evals/README.md` has the commands. The design choices:

- **Run the real interaction agent, fake only what isn't under test.** Each case runs
  `InteractionAgentRuntime.execute` inside a sandbox:
  - a temp roster, conversation log and execution logs;
  - a recorder in place of the execution batch manager, since routing is under test, not
    execution.

  The sandbox drains background dispatch tasks before restoring patches. An early version
  leaked real execution-agent calls, and a probe confirmed it (0 real calls inside the sandbox,
  1 after).
- **Record once, replay free.** The first live run saves every LLM response under a hash of the
  full request: system prompt, tools, roster, history, model, case and trial. `pytest` replays
  them offline and requires the exact same verdict for every case and trial.
  - Replay tests *our* code: a prompt, tool or grader change that moves any verdict fails the
    build.
  - Live runs measure *the model*. A prompt change makes recordings miss, and `--update-baseline`
    re-records and rewrites the baseline, whose diff shows exactly which cases moved.
- **Grade with code, not an LLM judge.** A verdict is a pure function of what was dispatched, what
  was created and what the user was told. There are 17 named outcomes (`reused_target`,
  `spawned_duplicate`, `guessed_candidate`, `asked_user`, ...), so a failure says what happened.
- **Read any run as a conversation.** `python -m evals.show amb-06@5` (or `--family ambiguous`)
  prints the scripted history, each model call, what it sent to which agent, what the tools
  refused, and the verdict. It reads recordings only, so it's free.
- **Case families,** each at roster sizes 5 and 500, plus 50 for the basic set:

  | family | what it probes | pass |
  |---|---|---|
  | reuse / create (90) | the obvious cases: the agent is named, or nothing fits | reuse it / create one |
  | paraphrase (30) | follow-up shares no words with the agent; history shows who did the work | reuse it |
  | trap (30) | same, plus 3 near-duplicate names (Email to Alice / Alicia / Lunch with Alice) | reuse the right one |
  | drift (20) | turn 1 creates, turn 2 is a paraphrased follow-up | come back to turn 1's agent |
  | ambiguous (14) | two agents fit and nothing breaks the tie | see below |
  | sounds_new (20) | an agent owns the work but the request reads as new, no history | reuse, don't create |
- **How ambiguous cases pass.** Each one is marked as an action or a question:
  - an action ("tell alice i'm running late", "move the dentist appointment") must be asked
    about, since doing it to both or to the wrong one changes something real;
  - a question ("did the insurance renewal go through?") passes if the model asks *or* checks
    every matching agent and reports on all of them (`checked_every_candidate`). Checking both
    saves the user a round trip.

  A pair only counts as ambiguous if the user could tell the two apart (two Alices, their
  dentist appointment and the kids'). An earlier case with "Stripe Job Offer" and "Stripe Offer
  Negotiation" was removed: those are one job split across two agents, a duplicate, and asking the
  user "the offer or the negotiation?" would make no sense to them.
- **Metrics:**
  - delegation rate;
  - per-kind accuracy among runs that routed or asked;
  - ask rate;
  - duplicate rate;
  - pass^k (every trial passed);
  - distinct names invented per request;
  - prompt tokens;
  - cost.
- **Before and after on the same cases.** The tag `stage-0` holds the original code with every
  case family and its recordings. `python -m evals.compare stage-0` prints each baseline's table
  next to the current one. When new families were added, they were recorded on the old code too,
  in a separate checkout.

## Results

### Before (original code, Sonnet 4, 2 to 3 trials)

| | 5 agents | 50 agents | 500 agents |
|---|---|---|---|
| first prompt, tokens | 3.4k | 3.9k | 9.2k |
| reuse / create / paraphrase / trap / drift accuracy | 100% | 100% | 100% |
| ambiguous, passed | 7/21 | - | 1/21 |
| sounds_new (1 trial) | 10/10 | - | 9/10, one duplicate |

Sonnet routes the obvious cases perfectly at every size. Its problems are cost, which grows with
the roster, and guessing: on ambiguous cases it picked one of the two in 34 of 42 runs. It asked
only once.

### Each stage on gpt-5-mini (hard set, 1 trial per stage; "before" 2 trials)

| | before | stage 1 | stages 2+3 |
|---|---|---|---|
| prompt tokens at 500 agents | 7.4k | 13.5k | **3.2k** |
| paraphrase | 53% | 67% | 63% |
| trap | 48% | 67% | 67% |
| drift | 5% | 15% | 20% |
| ambiguous (asked; graded before the action/question split) | 56% | 69% | 62% |
| sounds_new (reused) | 32% | 25% | 35% |
| duplicates created | 0 | 0 | 0 |

- **Stage 1 improved routing on every family except sounds_new.** Its cost was prompt size:
  descriptions for all 500 agents.
- **Stages 2+3 kept stage 1's accuracy at a quarter of the prompt.** Prompt size no longer depends
  on roster size (3.1k to 3.2k tokens at 5, 50 and 500). Offline, the right agent made the short
  list in every case at every size.
- **gpt-5-mini is a poor router regardless of our code.** It asks a needless question on about
  half its turns ("what's your ZIP?"), which caps every score above. It was the cheap instrument
  for checking each stage works end to end. It also can't show duplicates: a model that asks
  instead of acting never creates one. Sonnet, which almost never asks, is where duplicates would
  appear.

### After (stages 1 to 3, Sonnet 4)

1 trial per case. `python -m evals.compare stage-0 --model anthropic/claude-sonnet-4` prints the
full diff.

| | before, 5 | after, 5 | before, 500 | after, 500 |
|---|---|---|---|---|
| first prompt, tokens | 3.4k | 4.2k | 9.2k | **4.2k** |
| reuse / create / paraphrase / trap / drift | 100% | 100% | 100% | 100% |
| sounds_new (reused) | 10/10 | 10/10 | 9/10 | **10/10** |
| duplicates created | 0 | 0 | 1 | **0** |
| ambiguous, passed | 7/21 (33%) | 4/7 (57%) | 1/21 (5%) | 3/7 (43%) |

- **Overload is solved for prompt size and for duplicates.** The prompt no longer grows with the
  roster: at 500 agents it is less than half its old size. The original code created a duplicate
  ("Auto Insurance Shopping" beside "Car Insurance Renewal") when the request didn't share words
  with the existing agent's name. With the new code, the similarity check caught it.
- **At 5 agents the prompt is about 0.8k tokens bigger** because of descriptions and the new tools.
  That's the fixed price of routing on meaning rather than names.
- **Ambiguity is better but not solved.** Sonnet now asks about the two Alices, and checks both
  insurance renewals and both birthday plans instead of picking one. It still picks one agent
  without asking in 6 of 14 cases, and one question named "agents" to the user. With an earlier,
  blunter ask rule it asked on 5 of 5 but leaked "agents"; the current wording fixed the leak and
  lost some asking. The prompt alone doesn't reliably make Sonnet ask. The exact failing cases and
  the planned code-level fix are in [To do](#to-do).

## Other problems found along the way

- **Asking leaked the machinery.** Sonnet's first clarifying questions said "I see two
  dentist-related agents". The system prompt forbids ever mentioning agents to the user.
  - *Fix:* the ask rule now says to name the options in the user's terms.
  - *Evaluator:* a new verdict, `asked_about_agents`, fails any question that mentions agents.
    It still catches gpt-5-mini doing it on 3 to 4% of runs.
- **Over-asking is a model trait the prompt has to manage.** gpt-5-mini asked on 53 to 67% of
  turns in the original code, while Sonnet almost never asks. Swapping the interaction model
  without an eval like `ask rate` would silently change the product. The prompt now says when not
  to ask; the metric tracks it.
- **The roster only grows.** There's no way to retire an agent; `clear()` deletes everything.
  Stage 3 makes size cheap, but a proper fix would archive agents unused for N days: hidden from
  the list, still searchable, and brought back on use.
  - `last_used_at` is already recorded for this. It's not built.
- **Search reads every agent's log on every message.** `recent_requests` opens each agent's log
  file per query. That's fine at 500 agents, but it wants a small index (or storing the last
  instruction on the record) at larger scale.
- **`roster.json` read-modify-write isn't atomic across processes.** Writes take a file lock,
  but load-then-save doesn't, so two processes creating agents at once can lose one. SQLite,
  already used for triggers, would fix it.
- **No tests existed.** The repo had no test suite. This branch adds 53 tests:
  - server unit tests for the roster, the tools, search and the visible list;
  - grader tests;
  - harness isolation tests;
  - a replay of every baseline.
- **The eval harness put words in the model's mouth.** To save a call, the harness answers the
  model's closing turn after a dispatch itself. It used to answer "On it.", which overwrote what
  the model had actually said ("I'll check on both your insurance renewals") in the results and in
  the conversation log. The second turn of every drift case saw that fake reply.
  - *Fix:* the stand-in reply is now empty, so the model's own words stay. The 11 affected drift
    recordings were re-recorded.
- **Reading runs as conversations found bad test cases.** Grading had passed over them; reading
  the transcripts (`evals.show`) exposed them:
  - A "Stripe Job Offer" / "Stripe Offer Negotiation" pair was graded as ambiguous, but it's one
    job split in two. Removed (see how ambiguous cases pass).
  - "Checked both insurance renewals" was graded a failure. For a question that's the best
    answer, which led to the action/question split.
  - The rest are still open and listed in [To do](#to-do).

## Limits and next steps

- **Paraphrase with no shared words and no history.** Suppose "any word back from them?" refers
  to an old agent that isn't in the conversation. BM25 won't surface it, and the model has to
  think to search.
  - None of our cases isolate this, so it is unmeasured.
  - *Next:* add that case family. If it fails, add embeddings next to BM25 (hybrid retrieval cut
    failed retrievals by 49% in Anthropic's contextual-retrieval study).
- **The similarity check is noisy.** It fired on 30 of 31 creates, because generic words ("find",
  "check", "email") match something. The model confirmed new work correctly and accuracy held,
  but each false alarm is one extra call. A per-roster relevance cutoff, or embeddings, would cut
  it.
- **Small samples.** Stage results and the Sonnet "after" are 1 trial per case, with 14 to 30 runs
  per family, so differences under about 10 points are noise. "Before" numbers use 2 to 3 trials.
- **Synthetic rosters.** Distractor descriptions come from templates ("Email thread with Bea."),
  while scenario agents have hand-written ones. Real rosters would have richer descriptions
  everywhere, which should help search.
- **Execution agents are stubbed.** Routing is what's under test, so dispatched agents don't run
  and never report back. Two things are therefore unmeasured:
  - whether the interaction agent relays every agent's answer after checking several (see To do);
  - drift turn 2 sees no agent report between the turns ("Booked for 7pm"), which makes it
    harder than real use, not easier.
- **The harness skips the model's closing turn after a dispatch.** That turn would only produce a
  reply, so it's skipped to save a full prompt per case. A model that dispatches a second agent
  only after seeing the first tool result is not measured.
- **Some reuse tests at 500 agents have same-company filler next to the target.** The filler
  pool has three agents per company (Job Offer, Interview Prep, Recruiter Follow-up). A 500-agent
  roster takes 499 of 545 names, so they almost always land together. Left as-is; see
  [To do](#to-do).

## To do

### Stop guessing on an action; check every match on a question

The ask rule in the prompt is not enough. On Sonnet 4, 7 of 14 ambiguous cases still fail.
`python -m evals.show --family ambiguous` prints each one. The 7:

**Actions (must ask; doing the wrong one, or both, changes something real)**

| case | user said | the two agents | what Sonnet did |
|---|---|---|---|
| `amb-02@5` | "move the dentist appointment to friday" | Dentist Appointment / Dentist Appointment for Kids | asked, but said "two dentist appointment **agents**" (`asked_about_agents`) |
| `amb-02@500` | same | same | "I'll move **your** dentist appointment to Friday" → yours only |
| `amb-07@500` | "follow up with the professor about the letter" | Email to Professor Chen (*about a recommendation letter*) / Email to Professor Cheng (*no letter in the description*) | "I'll follow up with Professor Chen" → Chen only. At 5 agents it asked, which is the pass. Chen's description mentions a letter and Cheng's does not, so a keyword check also thinks Chen is the clear winner. Still must ask: the user said "the professor", not "Chen". |

`amb-01` ("tell alice im running 10 min late", Alice Wong / Alice Park) already asks at both sizes.

**Questions (ask, or check every match; picking one is a fail)**

| case | user said | the two agents | what Sonnet did |
|---|---|---|---|
| `amb-03@5`, `amb-03@500` | "does the paris hotel have late checkout?" | Hotel in Paris / Hotel in Paris for Mom | checked yours only |
| `amb-06@5` | "is the tokyo flight confirmed?" | Flight to Tokyo / Flight to Tokyo for Sam | checked **Sam's** |
| `amb-06@500` | same | same | checked yours only |

`amb-04` (car vs home insurance) and `amb-05` (mom's gift vs dinner) already check both at both sizes.

**The planned fix is in `send_message_to_agent`, not the prompt.** Before dispatching, score
the user's message (not the model's rewritten instructions) against the other listed agents'
names and descriptions. If another agent's score is close to the chosen one (about 80% of it)
and the conversation does not already name the chosen agent, refuse and return both. Then:

- for a question, the model can call both (already a pass);
- for an action, it has to ask, the same way `create_agent` returns `similar_agents` instead of
  creating.

No new tool argument, so recordings that never fire still replay. Offline against Sonnet's
recorded sends (no API cost):

- **Would catch:** `amb-02@500`, both `amb-03`, both `amb-06`.
- **Would miss:** `amb-07@500`, because "letter" is only in Chen's description.
- **Would not fire** on trap, paraphrase, drift or sounds_new.
- **Would fire once** on `reuse-13@500` ("tell the notion recruiter im still interested"), which
  is a test bug, not a product bug: the filler pool added "Notion Recruiter Follow-up" next to
  "Email to Recruiter at Notion". Same job, two agents. Documented below; don't teach the tool to
  treat that as ambiguity.

`amb-02@5` is a wording leak, not a missed dispatch. The existing `asked_about_agents` grader
already fails it.

### Test bugs to leave or fix later

- **`reuse-13@500` / `para-15@500`:** filler "Notion Recruiter Follow-up" duplicates "Email to
  Recruiter at Notion". The generator should refuse filler that is the same job as the target
  (same person or company *and* the same kind of work).
- **`reuse-02@500` / `reuse-06@500`:** Vercel Job Offer sits next to Vercel Recruiter Follow-up;
  Stripe Interview Prep sits next to Stripe Recruiter Follow-up. These are separate stages of one
  job hunt, so they are closer to real overload than the Notion clone, but the tests grade only
  one answer as correct ("any update from stripe on the onsite?" could reasonably go to the
  recruiter). Left as-is rather than deleted; a later change could accept either reasonable
  owner.

### Relay after checking several agents

When Sonnet checks both insurance renewals it says "I'll check on both" and dispatches both, but
the evals stub the agents, so we never see whether it then tells the user about both answers
(car renewed, home payment failed) or collapses them into one. Drive the real
`handle_agent_message` path with scripted facts per agent and fail `dropped_a_report` if the
reply is missing a fact from either.
