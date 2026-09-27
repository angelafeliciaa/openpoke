"""Harder routing cases. Three families, roster sizes 5 and 500.

  paraphrase  follow-up shares no keyword with the agent name; history shows which agent did the work
  trap        same, but the roster also holds 3 near-duplicate names (Email to Alice vs Email to Alicia)
  drift       two turns: turn 1 creates an agent, turn 2 is a paraphrased follow-up that must find it

Run:  python -m evals.cases.generate_hard   -> evals/cases/hard.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from evals.cases.generate import build_roster, distractor_pool

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
            lines.append({
                "id": f"para-{i:02d}@{size}", "kind": "reuse", "family": "paraphrase",
                "target": target, "history": history, "message": follow_up,
                "roster_size": size, "roster": build_roster(rng, size, pool, target, exclude=[]),
            })
            lines.append({
                "id": f"trap-{i:02d}@{size}", "kind": "reuse", "family": "trap",
                "target": target, "history": history, "message": follow_up,
                "roster_size": size, "roster": _with_traps(rng, size, pool, target, traps),
            })
    for i, (turns, exclude) in enumerate(DRIFT, 1):
        for size in SIZES:
            lines.append({
                "id": f"drift-{i:02d}@{size}", "kind": "drift", "family": "drift",
                "target": None, "history": [], "message": turns[0], "turns": turns,
                "roster_size": size, "roster": build_roster(rng, size, pool, None, exclude=exclude),
            })
    OUT.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    print(f"wrote {len(lines)} cases to {OUT}")


if __name__ == "__main__":
    main()
