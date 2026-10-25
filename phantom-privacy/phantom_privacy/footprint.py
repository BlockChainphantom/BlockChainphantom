"""A self-assessment of your digital footprint.

The user answers yes/no questions; each answer either helps or hurts a score.
The scoring logic is kept separate from any input/output so it is easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    """One assessment question.

    ``good_answer`` is the answer (True=yes / False=no) that is good for privacy.
    ``weight`` is how much the question counts toward the score.
    ``advice`` is shown when the user's answer is *not* the good one.
    """

    key: str
    text: str
    good_answer: bool
    weight: int
    advice: str


QUESTIONS: list[Question] = [
    Question(
        key="real_number_signup",
        text="Do you give your real phone number when signing up for online services?",
        good_answer=False,
        weight=3,
        advice="Use a masking/VoIP number for sign-ups and keep your real number private.",
    ),
    Question(
        key="masking_number",
        text="Do you use a separate masking/VoIP number for strangers and forms?",
        good_answer=True,
        weight=2,
        advice="Set up a VoIP number (e.g. MySudo or an alternative) and hand that out instead.",
    ),
    Question(
        key="sms_2fa_only",
        text="Is SMS your only method of two-factor authentication?",
        good_answer=False,
        weight=3,
        advice="Move to an authenticator app or a hardware key; SMS 2FA is vulnerable to SIM/SS7 attacks.",
    ),
    Question(
        key="password_manager",
        text="Do you use a password manager with unique passwords per site?",
        good_answer=True,
        weight=3,
        advice="Adopt a password manager so a breach of one site can't unlock the others.",
    ),
    Question(
        key="contact_discovery",
        text="Do you allow apps to find you by phone number / upload contacts?",
        good_answer=False,
        weight=2,
        advice="Turn off contact discovery so apps can't link your number to hidden accounts.",
    ),
    Question(
        key="audited_accounts",
        text="Have you checked which accounts are tied to your phone number?",
        good_answer=True,
        weight=1,
        advice="Audit your accounts and migrate non-essential ones off your real number.",
    ),
    Question(
        key="data_broker_removal",
        text="Have you removed yourself from data-broker / people-search sites?",
        good_answer=True,
        weight=2,
        advice="Request removal from major data brokers (manually or via a reputable removal service).",
    ),
    Question(
        key="e2e_messaging",
        text="Do you use end-to-end encrypted messaging for sensitive conversations?",
        good_answer=True,
        weight=1,
        advice="Prefer end-to-end encrypted apps over plain SMS/cellular for anything sensitive.",
    ),
]


@dataclass
class Result:
    """Outcome of an assessment."""

    score: int
    max_score: int
    recommendations: list[str]

    @property
    def percent(self) -> int:
        if self.max_score == 0:
            return 0
        return round(100 * self.score / self.max_score)

    @property
    def rating(self) -> str:
        pct = self.percent
        if pct >= 80:
            return "Strong"
        if pct >= 50:
            return "Fair"
        return "Exposed"


def assess(answers: dict[str, bool], questions: list[Question] | None = None) -> Result:
    """Score a set of answers.

    ``answers`` maps a question ``key`` to the user's yes(True)/no(False) answer.
    Questions missing from ``answers`` are skipped (they count toward neither the
    score nor the maximum), so partial assessments are still meaningful.
    """

    questions = questions if questions is not None else QUESTIONS
    score = 0
    max_score = 0
    recommendations: list[str] = []

    for q in questions:
        if q.key not in answers:
            continue
        max_score += q.weight
        if answers[q.key] == q.good_answer:
            score += q.weight
        else:
            recommendations.append(q.advice)

    return Result(score=score, max_score=max_score, recommendations=recommendations)
