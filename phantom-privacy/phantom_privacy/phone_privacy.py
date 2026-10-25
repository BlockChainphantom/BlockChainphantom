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
