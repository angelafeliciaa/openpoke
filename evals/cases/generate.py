"""Generate routing cases: same 30 scenarios at roster sizes 5, 50 and 500.

Run:  python -m evals.cases.generate
Writes evals/cases/routing.jsonl. Deterministic (fixed seed) so recordings stay valid.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "routing.jsonl"
SIZES = (5, 50, 500)

# (existing agent name, follow-up message that should route back to it)
REUSE = [
    ("Email to Alice", "can u reply to alice and tell her the invoice is approved"),
    ("Vercel Job Offer", "ask vercel if they can bump the start date to november"),
    ("Flight to Tokyo", "did the tokyo flight get confirmed? check for the receipt"),
    ("Dentist Appointment", "move my dentist appointment to next thursday"),
    ("Email to Landlord", "tell the landlord the sink is still leaking"),
    ("Stripe Interview Prep", "any update from stripe on the onsite?"),
    ("Hotel in Lisbon", "cancel the lisbon hotel, plans changed"),
    ("Email to Professor Chen", "follow up with prof chen about the recommendation letter"),
    ("Gym Membership Cancel", "did they cancel my gym membership yet"),
    ("Birthday Gift for Mom", "order the gift for mom we talked about"),
    ("Email to Sharanjeet", "reply to sharanjeet and say yes to friday"),
    ("Car Insurance Renewal", "what did the insurance company say about the renewal"),
    ("Email to Recruiter at Notion", "tell the notion recruiter im still interested"),
    ("Passport Renewal", "whats the status on my passport"),
    ("Email to Jai", "ping jai again about the deck"),
]

# (message with no matching agent, keywords that must not appear in the roster)
CREATE = [
    ("book me a table for 2 at a thai place near soho tomorrow 7pm", ["thai", "table", "restaurant", "soho"]),
    ("find out when the next f1 race is", ["f1", "race"]),
    ("email my accountant asking for last years tax return", ["accountant", "tax"]),
    ("remind me to water the plants every monday", ["plant", "water"]),
    ("whats the cheapest way to ship a package to berlin", ["ship", "package", "berlin"]),
    ("look up if the apple store has the new macbook in stock", ["apple", "macbook"]),
    ("draft a message to my team saying im out sick today", ["team", "sick"]),
    ("check my inbox for anything from the bank this week", ["bank"]),
    ("find a plumber that can come this weekend", ["plumber"]),
    ("what time does the pharmacy on 5th close", ["pharmacy"]),
    ("summarize the latest email from HR", ["hr"]),
    ("renew my library card", ["library"]),
    ("get me a quote for a new phone screen repair", ["phone", "screen", "repair"]),
    ("set up a call with the wedding photographer", ["wedding", "photographer"]),
    ("look for a used bike under 300 near me", ["bike"]),
]

FIRST_NAMES = """Aarav Bea Carlos Dana Eli Farah Gus Hana Ivan Jade Kofi Lena Marco Nia Omar Priya Quinn Rosa
Sam Tara Uma Vik Wren Xia Yara Zane Amit Bella Chris Devi Emma Finn Gita Hugo Iris Jonas Kai Luca Maya Noor
Oscar Pia Ravi Sara Theo Ursula Vera Will Yusuf Zoe Alicia Alina Alison Jay Jaya Jaime Chen Cheng Sharan
Elena Diego Sofia Mateo Aiden Leo Mila Ayaan Anika Nikhil Riya Aditi Rohan Arjun Kavya Isha""".split()
COMPANIES = """Stripe Notion Figma Linear Vercel Datadog Airbnb Coinbase Shopify Cloudflare Plaid Ramp Brex
Retool Rippling Snowflake Databricks Anthropic Scale Cohere Discord Canva Duolingo Robinhood Instacart
DoorDash Lyft Pinterest Reddit Dropbox Asana Atlassian Twilio Okta Zoom Slack Miro Loom Webflow Gusto""".split()
CITIES = """Paris Rome Berlin Madrid Lisbon Amsterdam Vienna Prague Athens Dublin Oslo Stockholm Copenhagen
Zurich Geneva Milan Barcelona Porto Seoul Osaka Kyoto Bangkok Singapore Sydney Melbourne Auckland Vancouver
Toronto Montreal Denver Austin Seattle Portland Chicago Boston Miami Nashville Honolulu Cancun Reykjavik""".split()
SINGLES = """Dentist Follow-up|Eye Exam Booking|Vet Appointment for Milo|Tax Filing 2025|Lease Renewal|Wifi Outage Ticket
Amazon Return|Netflix Cancel|Spotify Family Plan|Piano Lessons|Marathon Registration|Concert Tickets Radiohead
Dry Cleaning Pickup|Grocery Order|Car Service Appointment|Oil Change|Parking Permit|Voter Registration
Moving Truck Rental|Storage Unit|Internet Provider Switch|Electricity Bill Dispute|Water Heater Repair
Yoga Class Signup|Ski Trip Planning|Camping Permit Yosemite|Museum Tickets|Theater Tickets Hamilton
Wine Club Cancel|Gift for Dad|Anniversary Dinner|Dog Walker Schedule|House Cleaner Booking
Laptop Warranty Claim|Headphones Return|Credit Card Dispute|Mortgage Refinance|Student Loan Payoff
Visa Application Japan|Global Entry Renewal|TSA PreCheck""".replace("\n", "|").split("|")


def distractor_pool() -> list[str]:
    pool: list[str] = []
    pool += [f"Email to {n}" for n in FIRST_NAMES]
    pool += [f"{c} Job Offer" for c in COMPANIES]
    pool += [f"{c} Interview Prep" for c in COMPANIES]
    pool += [f"{c} Recruiter Follow-up" for c in COMPANIES]
    pool += [f"Flight to {c}" for c in CITIES]
    pool += [f"Hotel in {c}" for c in CITIES]
    pool += [f"Trip Planning {c}" for c in CITIES]
    pool += [s.strip() for s in SINGLES if s.strip()]
    pool += [f"Lunch with {n}" for n in FIRST_NAMES]
    pool += [f"Call {n} Back" for n in FIRST_NAMES]
    pool += [f"Reminder {i}" for i in range(1, 40)]
    return sorted(set(pool))


def _ok_distractor(name: str, exclude: list[str]) -> bool:
    low = name.lower()
    return not any(k in low for k in exclude)


def build_roster(rng: random.Random, size: int, pool: list[str], must_have: str | None, exclude: list[str]) -> list[str]:
    candidates = [p for p in pool if p != must_have and _ok_distractor(p, exclude)]
    need = size - (1 if must_have else 0)
    picked = rng.sample(candidates, need)
    if must_have:
        picked.append(must_have)
    rng.shuffle(picked)
    return picked


def main() -> None:
    rng = random.Random(20260925)
    pool = distractor_pool()
    assert len(pool) >= max(SIZES) + 20, len(pool)
    lines = []
    for i, (target, message) in enumerate(REUSE, 1):
        for size in SIZES:
            lines.append({
                "id": f"reuse-{i:02d}@{size}",
                "kind": "reuse",
                "target": target,
                "message": message,
                "roster_size": size,
                "roster": build_roster(rng, size, pool, target, exclude=[]),
            })
    for i, (message, exclude) in enumerate(CREATE, 1):
        for size in SIZES:
            lines.append({
                "id": f"create-{i:02d}@{size}",
                "kind": "create",
                "target": None,
                "message": message,
                "roster_size": size,
                "roster": build_roster(rng, size, pool, None, exclude=exclude),
            })
    OUT.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    print(f"wrote {len(lines)} cases to {OUT}")


if __name__ == "__main__":
    main()
