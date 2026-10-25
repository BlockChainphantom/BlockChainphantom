"""Guidance on protecting your phone number – the core idea from the video.

The point: your phone number is effectively a *key to your identity*. Given a
number, people-search tools can surface the accounts, addresses and relatives
tied to it. The defence is to stop handing out your real number and to use a
masking (VoIP) number for sign-ups and strangers instead.

This module only returns structured text – it makes no network calls and
collects no data about you.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Topic:
    """A single piece of guidance with a title and bullet points."""

    title: str
    bullets: list[str] = field(default_factory=list)


# Why the phone number matters.
WHY_IT_MATTERS = Topic(
    title="Why your phone number is sensitive",
    bullets=[
        "Treat your number like a password to your identity – many accounts are "
        "keyed to it.",
        "People-search and data-broker tools can turn a number into your name, "
        "address, email and relatives.",
        "Contact-upload features on social apps can link a number to 'suggested' "
        "accounts, exposing profiles you meant to keep separate.",
    ],
)

# VoIP / masking numbers – the practical fix.
MASKING_NUMBERS = Topic(
    title="Use a masking (VoIP) number instead of your real one",
    bullets=[
        "A VoIP number is a second number not directly tied to your identity. "
        "Give it out online and to strangers; keep your real SIM number private.",
        "MySudo is a popular privacy-focused option (multiple numbers per "
        "identity). It is one choice, not the only one.",
        "Other options include Google Voice, Hushed and Burner – compare price, "
        "jurisdiction and whether they require your real number to sign up.",
        "Keep your real number for close contacts, your bank and account "
        "recovery only.",
    ],
)

# SS7 / telecom awareness – educational, defensive framing.
TELECOM_AWARENESS = Topic(
    title="Know the limits of the phone network (SS7)",
    bullets=[
        "The signalling protocols behind telecoms (SS7) are decades old and have "
        "known weaknesses that can, in some cases, be abused to locate or "
        "intercept.",
        "You cannot fix SS7 yourself – but you can reduce your exposure: prefer "
        "app-based or hardware 2FA over SMS codes.",
        "Use end-to-end encrypted messaging/calling apps rather than relying on "
        "the plain cellular network for sensitive conversations.",
    ],
)

# A concrete, do-this-today checklist.
CHECKLIST = Topic(
    title="Checklist: shrink your phone-number exposure",
    bullets=[
        "Stop giving your real number for online sign-ups, loyalty cards and "
        "forms – use a masking number.",
        "Audit which accounts use your real number and migrate non-essential "
        "ones to the masking number.",
        "Turn off 'find me by phone number' / contact discovery in social apps.",
        "Switch 2FA from SMS to an authenticator app or hardware key where "
        "possible.",
        "Consider a reputable data-broker removal service to clean up records "
        "already published about you.",
    ],
)

ALL_TOPICS: list[Topic] = [
    WHY_IT_MATTERS,
    MASKING_NUMBERS,
    TELECOM_AWARENESS,
    CHECKLIST,
]


def render(topics: list[Topic] | None = None) -> str:
    """Return the guidance as a readable, plain-text block."""

    topics = topics if topics is not None else ALL_TOPICS
    lines: list[str] = []
    for topic in topics:
        lines.append(topic.title)
        lines.append("-" * len(topic.title))
        for bullet in topic.bullets:
            lines.append(f"  * {bullet}")
        lines.append("")  # blank line between topics
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# VoIP / masking-number comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VoipService:
    """One VoIP / masking-number option, described in comparable columns.

    The fields are deliberately short strings so they line up in a table.
    Features, pricing and availability change often, so treat this as a
    starting point and verify the current details on each provider's site.
    """

    name: str
    multiple_numbers: str
    privacy_focus: str
    needs_real_number: str
    platforms: str
    price: str
    notes: str


# NOTE: details below are a general snapshot for comparison, not live data.
# Always confirm current plans, pricing and region support before signing up.
VOIP_SERVICES: list[VoipService] = [
    VoipService(
        name="MySudo",
        multiple_numbers="Yes (up to 9)",
        privacy_focus="High",
        needs_real_number="No",
        platforms="iOS, Android, web",
        price="Free tier + paid ~$1-15/mo (verify)",
        notes="Privacy-first 'identities'; bundles encrypted email/messaging.",
    ),
    VoipService(
        name="Hushed",
        multiple_numbers="Yes",
        privacy_focus="High",
        needs_real_number="No",
        platforms="iOS, Android",
        price="Pay-as-you-go from ~$2; subs ~$5/mo (verify)",
        notes="Second numbers; many country options.",
    ),
    VoipService(
        name="Burner",
        multiple_numbers="Yes",
        privacy_focus="Medium-High",
        needs_real_number="No",
        platforms="iOS, Android",
        price="Subscription ~$5/mo or ~$48/yr (verify)",
        notes="Disposable numbers you can rotate/'burn'; US & Canada focus.",
    ),
    VoipService(
        name="Google Voice",
        multiple_numbers="No (one)",
        privacy_focus="Low-Medium",
        needs_real_number="Yes (links an existing number)",
        platforms="iOS, Android, web",
        price="Free (personal US); Workspace plans paid (verify)",
        notes="Tied to your Google identity.",
    ),
]


def render_voip_table(services: list[VoipService] | None = None) -> str:
    """Render the VoIP options as an aligned plain-text table."""

    services = services if services is not None else VOIP_SERVICES
    headers = [
        "Service",
        "Multiple #",
        "Privacy",
        "Needs real #?",
        "Platforms",
        "Price (verify)",
        "Notes",
    ]
    rows = [
        [
            s.name,
            s.multiple_numbers,
            s.privacy_focus,
            s.needs_real_number,
            s.platforms,
            s.price,
            s.notes,
        ]
        for s in services
    ]

    # Compute each column's width from the widest cell (header or value).
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt(row) for row in rows]
    lines.append("")
    lines.append(
        "Details change often — confirm current plans, pricing and region "
        "support on each provider's site before signing up."
    )
    return "\n".join(lines) + "\n"
