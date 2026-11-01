"""Safety guardrails: distress detection + reassurance-for-compulsions filter."""
import re
from config import DISTRESS_KEYWORDS, THERAPIST_CONTACT


def is_distress(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in DISTRESS_KEYWORDS)


def distress_response() -> str:
    return (
        "I hear that this is really hard right now. You're not alone. "
        "Please take a breath. If you or your son are in immediate danger, "
        "consult with his counsellor. "
        "I'm here to help you think through the next step - "
        "would you like to talk about what's happening?"
    )


_REASSURANCE_PATTERNS = [
    r"you checked (enough|plenty|many times)",
    r"it('s| is) (definitely|certainly|surely) (locked|off|clean|safe)",
    r"don't worry,? (it's|it is) (fine|okay|ok|safe)",
    r"you (can|should) stop (checking|washing|counting)",
    r"there'?s no need to (check|wash|count) again",
    r"i (promise|assure you) (it's|it is)",
]


def contains_reassurance(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _REASSURANCE_PATTERNS)


SAFE_FALLBACK = (
    "I hear how anxious he is. Rather than reassuring the worry, let's try a small step: "
    "acknowledge the feeling, then offer a 5-minute delay before any checking/washing, "
    "and redirect to a calming activity. Would you like a specific script to say to him?"
)


def sanitize_output(text: str) -> str:
    if contains_reassurance(text):
        return SAFE_FALLBACK
    return text


def check_input(text: str):
    if is_distress(text):
        return False, distress_response()
    return True, None
