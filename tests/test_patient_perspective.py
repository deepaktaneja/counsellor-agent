import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.advisor import _patient_perspective


def test_patient_perspective_converts_pronouns():
    """Patient-mode output must address the patient as you/your, never he/she/they."""
    assert _patient_perspective("He is anxious and his OCD is strong.") == (
        "You is anxious and your OCD is strong."
    )
    assert _patient_perspective("She feels overwhelmed.") == "You feels overwhelmed."
    assert _patient_perspective("They should try grounding.") == "You should try grounding."
    assert _patient_perspective("Tell him to breathe; give her time.") == (
        "Tell you to breathe; give you time."
    )
    assert _patient_perspective("His anxiety, her progress, their plan.") == (
        "Your anxiety, you progress, your plan."
    )
    assert _patient_perspective("It's his; those are theirs; it's hers.") == (
        "It's your; those are yours; it's yours."
    )


def test_patient_perspective_contractions_and_reflexives():
    assert _patient_perspective("He's working on it. She'll be fine. They'd like this.") == (
        "You're working on it. You'll be fine. You'd like this."
    )
    assert _patient_perspective("Do it yourself") == "Do it yourself"
    assert _patient_perspective("They blamed themselves") == "You blamed yourselves"


def test_patient_perspective_preserves_case_and_unrelated_words():
    assert _patient_perspective("HE SHE THEY") == "YOU YOU YOU"
    assert _patient_perspective("The theme is whether he can cope.") == (
        "The theme is whether you can cope."
    )
    # Already correct text is left untouched
    assert _patient_perspective("You already use your words.") == (
        "You already use your words."
    )
    assert _patient_perspective("") == ""
    assert _patient_perspective(None) is None
