"""Harder routing cases. Four families, roster sizes 5 and 500.

  paraphrase  follow-up shares no keyword with the agent name; history shows which agent did the work
  trap        same, but the roster also holds 3 near-duplicate names (Email to Alice vs Email to Alicia)
  drift       two turns: turn 1 creates an agent, turn 2 is a paraphrased follow-up that must find it
  ambiguous   two agents fit and nothing breaks the tie; the right move is to ask the user which one
  sounds_new  an agent already owns the work but the request reads as brand new, with no history;
              creating another agent here is how rosters bloat
  relay       one question spans two agents; both must be checked and both answers told to the user
  long        the agent's report is 100+ messages back, behind the conversation summary; the
              follow-up must still find it
  stress      same, but three summarisations deep and the follow-up names no topic ("any word
              back yet?"), so the briefing is the only way back to the agent

Run:  python -m evals.cases.generate_hard   -> evals/cases/hard.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from evals.cases.generate import agent_id, build_roster, distractor_pool, roster_entries

OUT = Path(__file__).parent / "hard.jsonl"
SIZES = (5, 500)


def _hist(user: str, agent_name: str, agent_text: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "text": user},
        {"role": "assistant", "text": "On it, give me a sec."},
        {"role": "agent", "text": f"{agent_name}: {agent_text}"},
    ]


# (target, history, follow-up, traps)
SCENARIOS = [
    ("Vercel Job Offer", _hist("ask vercel if they can bump the start date to november", "Vercel Job Offer",
        "Emailed the recruiter asking to push the start date to November. Waiting on a reply."),
     "any word back yet?", ["Vercel Interview Prep", "Vercel Recruiter Follow-up", "Netlify Job Offer"]),
    ("Email to Landlord", _hist("tell the landlord the sink is still leaking", "Email to Landlord",
        "Sent. He says a plumber can come Tuesday."),
     "ask him if it can be the morning instead", ["Water Heater Repair", "Lease Renewal", "Email to Lena"]),
    ("Flight to Tokyo", _hist("find me a flight to tokyo on the 12th", "Flight to Tokyo",
        "Found ANA departing 11:40am for $840. Want me to book it?"),
     "yeah go for it", ["Hotel in Tokyo", "Trip Planning Tokyo", "Flight to Kyoto"]),
    ("Dentist Appointment", _hist("book me a dentist cleaning", "Dentist Appointment",
        "Booked Thursday 3pm at Bright Smile Dental."),
     "actually can we do the following week", ["Dentist Follow-up", "Eye Exam Booking", "Vet Appointment for Milo"]),
    ("Birthday Gift for Mom", _hist("help me find a gift for my mom's 60th", "Birthday Gift for Mom",
        "Three options: a spa day, a Le Creuset set, or a weekend in Napa."),
     "the second one, order it", ["Gift for Dad", "Anniversary Dinner", "Email to Maya"]),
    ("Stripe Interview Prep", _hist("i have a stripe onsite next week, help me prep", "Stripe Interview Prep",
        "Put together a study plan covering system design and the payments API."),
     "can u send me that plan again", ["Stripe Job Offer", "Stripe Recruiter Follow-up", "Plaid Interview Prep"]),
    ("Car Insurance Renewal", _hist("my car insurance is up for renewal, get me quotes", "Car Insurance Renewal",
        "Geico $92/mo, Progressive $88/mo. You currently pay $105."),
     "switch to the cheaper one", ["Car Service Appointment", "Oil Change", "Mortgage Refinance"]),
    ("Passport Renewal", _hist("start my passport renewal", "Passport Renewal",
        "Filled out the DS-82. I need a new photo from you."),
     "i uploaded the photo, keep going", ["Visa Application Japan", "Global Entry Renewal", "TSA PreCheck"]),
    ("Email to Sharanjeet", _hist("email sharanjeet and ask if friday works for the demo", "Email to Sharanjeet",
        "Sent. She replied that Friday 2pm works."),
     "tell her perfect, see her then", ["Email to Sharan", "Lunch with Sharan", "Call Sharan Back"]),
    ("Hotel in Lisbon", _hist("book a hotel in lisbon for the 3rd to the 6th", "Hotel in Lisbon",
        "Booked Hotel Avenida, confirmation A1B2."),
     "does that place have parking?", ["Flight to Lisbon", "Trip Planning Lisbon", "Hotel in Porto"]),
    ("Email to Alice", _hist("find alice's email about the invoice", "Email to Alice",
        "Found it. She needs approval by Friday."),
     "reply and say it's approved", ["Email to Alicia", "Lunch with Alice", "Call Alice Back"]),
    ("Email to Jai", _hist("ask jai when the deck is due", "Email to Jai", "He says Monday."),
     "tell him monday is fine", ["Email to Jay", "Email to Jaya", "Email to Jaime"]),
    ("Email to Professor Chen", _hist("ask prof chen for a rec letter", "Email to Professor Chen",
        "Asked. She wants your CV first."),
     "send her my cv then", ["Email to Chen", "Email to Cheng", "Lunch with Chen"]),
    ("Gym Membership Cancel", _hist("cancel my gym membership", "Gym Membership Cancel",
        "They need written notice. I drafted one."),
     "ok send it", ["Yoga Class Signup", "Netflix Cancel", "Wine Club Cancel"]),
    ("Email to Recruiter at Notion", _hist("reply to the notion recruiter, im interested", "Email to Recruiter at Notion",
        "Replied. She wants to schedule a call this week."),
     "thursday afternoon works", ["Notion Job Offer", "Notion Interview Prep", "Email to Noor"]),
]

# (turn 1 creates an agent, turn 2 paraphrased follow-up, keywords excluded from roster)
DRIFT = [
    (["book me a table for 2 at a thai place near soho tomorrow 7pm", "make it 3 people instead"], ["thai", "table", "restaurant", "soho"]),
    (["find out when the next f1 race is", "and where is it?"], ["f1", "race"]),
    (["email my accountant asking for last years tax return", "did they reply?"], ["accountant", "tax"]),
    (["whats the cheapest way to ship a package to berlin", "ok book that option"], ["ship", "package", "berlin"]),
    (["look up if the apple store has the new macbook in stock", "reserve one for pickup"], ["apple", "macbook"]),
    (["find a plumber that can come this weekend", "go with whoever is cheapest"], ["plumber"]),
    (["set up a call with the wedding photographer", "ask if she does video too"], ["wedding", "photographer"]),
    (["look for a used bike under 300 near me", "message the seller of the first one"], ["bike"]),
    (["get me a quote for a new phone screen repair", "book it for saturday"], ["phone", "screen", "repair"]),
    (["renew my library card", "did that go through?"], ["library"]),
]


# (candidates that fit equally, message with no history to break the tie, keywords excluded from distractors,
#  read_only). An action must be asked about; a question can instead be put to every candidate.
# Each pair is two things the user can tell apart; two agents splitting one job are a duplicate, not ambiguity.
AMBIGUOUS = [
    (["Email to Alice Park", "Email to Alice Wong"], "tell alice im running 10 min late", ["alice"], False),
    (["Dentist Appointment", "Dentist Appointment for Kids"], "move the dentist appointment to friday", ["dentist"], False),
    (["Hotel in Paris", "Hotel in Paris for Mom"], "does the paris hotel have late checkout?", ["paris"], True),
    (["Car Insurance Renewal", "Home Insurance Renewal"], "did the insurance renewal go through?", ["insurance"], True),
    (["Birthday Gift for Mom", "Birthday Dinner for Mom"], "whats the status on mom's birthday thing", ["mom", "birthday"], True),
    (["Flight to Tokyo", "Flight to Tokyo for Sam"], "is the tokyo flight confirmed?", ["tokyo"], True),
    (["Email to Professor Chen", "Email to Professor Cheng"], "follow up with the professor about the letter", ["chen", "professor"], False),
]

# What each agent answers when a question is put to it. The marker is a proper noun, code or price that
# survives paraphrase; a reply that lacks it dropped that agent's answer.
REPORTS = {
    "Hotel in Paris": ("Yes, the Hotel Lutetia allows late checkout until 2pm.", "Lutetia"),
    "Hotel in Paris for Mom": ("The Hotel Regina doesn't, but they hold luggage after checkout.", "Regina"),
    "Car Insurance Renewal": ("Renewed with Progressive at $88/mo.", "Progressive"),
    "Home Insurance Renewal": ("The renewal failed, the card on file was declined.", "declined"),
    "Birthday Gift for Mom": ("The Le Creuset set ships Tuesday.", "Le Creuset"),
    "Birthday Dinner for Mom": ("Table for 6 at Osteria Mozza booked for 7:30pm.", "Osteria"),
    "Flight to Tokyo": ("Confirmed, ANA 11:40am, seat 14A.", "14A"),
    "Flight to Tokyo for Sam": ("Not yet, ANA is holding the fare until Friday.", "Friday"),
    "Hotel in Lisbon": ("Confirmed at Hotel Avenida, confirmation code A1B2.", "A1B2"),
    "Vercel Job Offer": ("The recruiter agreed to a November 3 start.", "November"),
    "Stripe Interview Prep": ("The onsite is set for Thursday at 10am.", "Thursday"),
    "Email to Alice": ("She replied: the invoice is approved.", "approved"),
    "Email to Jai": ("He says the deck is due Monday.", "Monday"),
    "Dentist Appointment": ("Booked Thursday 3pm at Bright Smile Dental.", "Bright Smile"),
    "Gym Membership Cancel": ("Cancelled; a $45 refund lands within 5 days.", "$45"),
    "Passport Renewal": ("The renewal form is submitted, expected back in 6 weeks.", "6 weeks"),
}

# (two agents one question spans, the question, keywords kept out of distractors). Both must be checked,
# and the reply must carry both answers.
RELAY = [
    (["Flight to Tokyo", "Hotel in Lisbon"], "are the tokyo flight and the lisbon hotel both confirmed?", ["tokyo", "lisbon"]),
    (["Vercel Job Offer", "Stripe Interview Prep"], "any news from vercel or stripe?", ["vercel", "stripe"]),
    (["Email to Alice", "Email to Jai"], "did alice and jai get back to me?", ["alice", "jai"]),
    (["Dentist Appointment", "Gym Membership Cancel"], "is the dentist booked and the gym cancelled?", ["dentist", "gym"]),
    (["Car Insurance Renewal", "Passport Renewal"], "where are we on the car insurance and the passport?", ["insurance", "passport"]),
]


# (target, what the user asked back then, what the agent reported, the follow-up much later)
LONG = [
    ("Vercel Job Offer", "ask vercel if they can bump the start date to november",
     "Emailed the recruiter asking to push the start date to November. Waiting on a reply.", "any word back from vercel yet?"),
    ("Email to Landlord", "tell the landlord the sink is still leaking",
     "Sent. He says a plumber can come Tuesday.", "did the plumber thing with the landlord ever get sorted?"),
    ("Flight to Tokyo", "find me a flight to tokyo on the 12th",
     "Found ANA departing 11:40am for $840. Want me to book it?", "ok let's book that tokyo flight after all"),
    ("Dentist Appointment", "book me a dentist cleaning",
     "Booked Thursday 3pm at Bright Smile Dental.", "can we push my dentist appointment a week"),
    ("Stripe Interview Prep", "i have a stripe onsite next week, help me prep",
     "Put together a study plan covering system design and the payments API.", "send me that stripe study plan again"),
    ("Car Insurance Renewal", "my car insurance is up for renewal, get me quotes",
     "Geico $92/mo, Progressive $88/mo. You currently pay $105.", "go ahead and switch my car insurance to the cheaper one"),
    ("Passport Renewal", "start my passport renewal",
     "Filled out the DS-82. I need a new photo from you.", "i finally uploaded the passport photo, keep going"),
    ("Hotel in Lisbon", "book a hotel in lisbon for the 3rd to the 6th",
     "Booked Hotel Avenida, confirmation A1B2.", "does the lisbon hotel have parking?"),
    ("Email to Professor Chen", "ask prof chen for a rec letter",
     "Asked. She wants your CV first.", "send prof chen my cv for the letter"),
    ("Gym Membership Cancel", "cancel my gym membership",
     "They need written notice. I drafted one.", "ok send the gym the cancellation notice"),
]

# Small talk that names no agent's work, so the summariser has nothing to confuse it with.
SMALL_TALK = [
    ("whats the weather looking like", "Sunny and 72 this afternoon, light rain after 8pm."),
    ("any good podcasts lately", "People keep recommending Acquired's Costco episode."),
    ("how many ounces in a cup", "8 fluid ounces."),
    ("tell me a fun fact", "Octopuses have three hearts."),
    ("whats a good name for a grey cat", "Pebble, Ash, or Earl Grey."),
    ("is it too late for coffee", "It's 4pm, so probably one more is fine."),
    ("how do you say thank you in portuguese", "Obrigado, or obrigada if you're a woman."),
    ("whats the capital of australia", "Canberra, not Sydney."),
    ("im bored", "Go for a walk, the rain's stopped."),
    ("how long do you boil an egg for jammy", "Six and a half minutes, then ice water."),
    ("recommend a movie for tonight", "Past Lives if you want quiet, Heat if you don't."),
    ("what year did the berlin wall fall", "1989."),
    ("good morning", "Morning. Coffee first?"),
    ("how far is a 10k in miles", "6.2 miles."),
    ("whats the plural of octopus", "Octopuses. Octopi is common but not Greek."),
    ("do you like jazz", "Kind of Blue is a safe yes."),
    ("whats 15 percent of 84", "12.60."),
    ("night", "Night. Sleep well."),
]


def _long_history(user_then: str, agent_name: str, agent_text: str, names: list[str], pairs: int = 54) -> list[dict[str, str]]:
    """The scenario at the very start, then `pairs` exchanges of small talk, so the report is behind the summary."""
    history = _with_ids(_hist(user_then, agent_name, agent_text), names)
    for i in range(pairs):
        user, reply = SMALL_TALK[i % len(SMALL_TALK)]
        history.append({"role": "user", "text": user})
        history.append({"role": "assistant", "text": reply})
    return history


# (target, what the user asked back then, what the agent reported, a follow-up that names nothing)
STRESS = [
    ("Vercel Job Offer", "ask vercel if they can bump the start date to november",
     "Emailed the recruiter asking to push the start date to November. Waiting on a reply.", "any word back yet?"),
    ("Email to Landlord", "tell the landlord the sink is still leaking",
     "Sent. He says a plumber can come Tuesday.", "did that ever get sorted?"),
    ("Flight to Tokyo", "find me a flight to tokyo on the 12th",
     "Found ANA departing 11:40am for $840. Want me to book it?", "ok let's book it after all"),
    ("Passport Renewal", "start my passport renewal",
     "Filled out the DS-82. I need a new photo from you.", "i finally uploaded the photo, keep going"),
    ("Gym Membership Cancel", "cancel my gym membership",
     "They need written notice. I drafted one.", "ok go ahead and send it"),
]


def _reports(candidates: list[str]) -> list[dict[str, str]]:
    return [{"agent": c, "says": REPORTS[c][0], "must_relay": REPORTS[c][1]} for c in candidates]


# (agent that already owns the work, request phrased as brand new with no history, keywords kept out of distractors)
SOUNDS_NEW = [
    ("Hotel in Lisbon", "look into a place to stay in lisbon for my trip", ["lisbon"]),
    ("Passport Renewal", "i need to get my passport renewed, can you get that going", ["passport"]),
    ("Gym Membership Cancel", "i want to quit my gym, can you handle it", ["gym"]),
    ("Car Insurance Renewal", "shop around for cheaper auto insurance, mine renews soon", ["insurance"]),
    ("Birthday Gift for Mom", "help me figure out a present for my mom's birthday", ["mom", "gift"]),
    ("Stripe Interview Prep", "i need to get ready for my stripe interviews", ["stripe"]),
    ("Flight to Tokyo", "look up flights to tokyo for me", ["tokyo"]),
    ("Email to Landlord", "write to my landlord about the leaky sink", ["landlord"]),
    ("Vercel Job Offer", "negotiate the start date on my vercel offer", ["vercel"]),
    ("Email to Professor Chen", "reach out to professor chen about a reference letter", ["chen"]),
]


def _with_ids(history: list[dict[str, str]], names: list[str]) -> list[dict[str, str]]:
    """Agent reports carry `Name (a12):`, as batch_manager writes them."""
    out = []
    for h in history:
        name, sep, rest = h["text"].partition(": ")
        if h["role"] == "agent" and sep and name in names:
            h = {**h, "text": f"{name} ({agent_id(names, name)}): {rest}"}
        out.append(h)
    return out


def _with_candidates(rng: random.Random, size: int, pool: list[str], candidates: list[str], exclude: list[str]) -> list[str]:
    roster = build_roster(rng, size - len(candidates), pool, None, exclude=exclude, protect=candidates) + candidates
    rng.shuffle(roster)
    return roster


def _with_traps(rng: random.Random, size: int, pool: list[str], target: str, traps: list[str]) -> list[str]:
    roster = build_roster(rng, size - len(traps), [p for p in pool if p not in traps], target, exclude=[])
    roster += traps
    rng.shuffle(roster)
    return roster


def main() -> None:
    rng = random.Random(20260926)
    pool = distractor_pool()
    lines = []
    for i, (target, history, follow_up, traps) in enumerate(SCENARIOS, 1):
        for size in SIZES:
            names = build_roster(rng, size, pool, target, exclude=[])
            lines.append({
                "id": f"para-{i:02d}@{size}", "kind": "reuse", "family": "paraphrase",
                "target": target, "history": _with_ids(history, names), "message": follow_up,
                "roster_size": size, "roster": roster_entries(names),
            })
            names = _with_traps(rng, size, pool, target, traps)
            lines.append({
                "id": f"trap-{i:02d}@{size}", "kind": "reuse", "family": "trap",
                "target": target, "history": _with_ids(history, names), "message": follow_up,
                "roster_size": size, "roster": roster_entries(names),
            })
    for i, (turns, exclude) in enumerate(DRIFT, 1):
        for size in SIZES:
            lines.append({
                "id": f"drift-{i:02d}@{size}", "kind": "drift", "family": "drift",
                "target": None, "history": [], "message": turns[0], "turns": turns,
                "roster_size": size, "roster": roster_entries(build_roster(rng, size, pool, None, exclude=exclude)),
            })
    # own seed, so adding these never reshuffles the rosters above
    amb_rng = random.Random(20260927)
    for i, (candidates, message, exclude, read_only) in enumerate(AMBIGUOUS, 1):
        for size in SIZES:
            lines.append({
                "id": f"amb-{i:02d}@{size}", "kind": "ambiguous", "family": "ambiguous",
                "target": None, "candidates": candidates, "read_only": read_only, "history": [], "message": message,
                "reports": _reports(candidates) if read_only else [],
                "roster_size": size, "roster": roster_entries(_with_candidates(amb_rng, size, pool, candidates, exclude)),
            })
    new_rng = random.Random(20260928)
    for i, (target, message, exclude) in enumerate(SOUNDS_NEW, 1):
        for size in SIZES:
            lines.append({
                "id": f"new-{i:02d}@{size}", "kind": "reuse", "family": "sounds_new",
                "target": target, "history": [], "message": message,
                "roster_size": size, "roster": roster_entries(build_roster(new_rng, size, pool, target, exclude=exclude)),
            })
    relay_rng = random.Random(20260929)
    for i, (candidates, message, exclude) in enumerate(RELAY, 1):
        for size in SIZES:
            lines.append({
                "id": f"relay-{i:02d}@{size}", "kind": "relay", "family": "relay",
                "target": None, "candidates": candidates, "history": [], "message": message,
                "reports": _reports(candidates),
                "roster_size": size, "roster": roster_entries(_with_candidates(relay_rng, size, pool, candidates, exclude)),
            })
    long_rng = random.Random(20260930)
    for i, (target, user_then, agent_text, follow_up) in enumerate(LONG, 1):
        for size in SIZES:
            names = build_roster(long_rng, size, pool, target, exclude=[])
            lines.append({
                "id": f"long-{i:02d}@{size}", "kind": "reuse", "family": "long", "summarize": True,
                "target": target, "history": _long_history(user_then, target, agent_text, names), "message": follow_up,
                "roster_size": size, "roster": roster_entries(names),
            })
    stress_rng = random.Random(20261001)
    for i, (target, user_then, agent_text, follow_up) in enumerate(STRESS, 1):
        for size in SIZES:
            names = build_roster(stress_rng, size, pool, target, exclude=[])
            lines.append({
                "id": f"stress-{i:02d}@{size}", "kind": "reuse", "family": "stress", "summarize": True,
                "target": target, "history": _long_history(user_then, target, agent_text, names, pairs=155),
                "message": follow_up, "roster_size": size, "roster": roster_entries(names),
            })
    OUT.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    print(f"wrote {len(lines)} cases to {OUT}")


if __name__ == "__main__":
    main()
