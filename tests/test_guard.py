import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safety.guard import is_distress, contains_reassurance, sanitize_output, check_input


def test_distress_detected():
    assert is_distress("I want to kill myself") is True
    assert is_distress("he is refusing to eat") is False


def test_reassurance_detected():
    assert contains_reassurance("don't worry, it's definitely locked") is True
    assert contains_reassurance("let's try a 5 minute delay") is False


def test_sanitize_replaces():
    out = sanitize_output("don't worry, it's definitely locked, you checked enough")
    assert "locked" not in out or "5-minute delay" in out


def test_check_input_crisis():
    safe, msg = check_input("I want to die")
    assert safe is False
    assert msg is not None
    assert "counsellor" in msg.lower() or "consult" in msg.lower()


def test_check_input_ok():
    safe, msg = check_input("he refused to leave the house")
    assert safe is True
    assert msg is None
