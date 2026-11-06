"""
P10 — query gate timeout + Socratic dialogue wiring (mocked input / client).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pipe_reader(monkeypatch):
    """A gate reading from a pipe we write to, as a user typing would."""
    import os
    import src.socratic.gate as gate
    r, w = os.pipe()
    monkeypatch.setattr(gate, "_reader", gate._LineReader(r, encoding="utf-8"))
    return gate, (lambda text: os.write(w, text.encode("utf-8"))), w


def test_gate_timeout_advances(monkeypatch):
    gate, _type, _ = _pipe_reader(monkeypatch)
    assert gate.ask_gate(timeout=0.3) is False


def test_gate_yes_and_other_key(monkeypatch):
    gate, type_, _ = _pipe_reader(monkeypatch)
    type_("y\n")
    time.sleep(0.2)
    # typed before the prompt: drained, so it answers nothing
    assert gate.ask_gate(timeout=0.3) is False
    import threading
    threading.Timer(0.1, lambda: type_("Y\n")).start()
    assert gate.ask_gate(timeout=3) is True
    threading.Timer(0.1, lambda: type_("n\n")).start()
    assert gate.ask_gate(timeout=3) is False


def test_gate_answer_after_a_timeout_reaches_the_next_gate(monkeypatch):
    """The old per-call thread stayed blocked after a timeout and ate gate 2's "y"."""
    import threading
    gate, type_, _ = _pipe_reader(monkeypatch)
    assert gate.ask_gate(timeout=0.2) is False
    threading.Timer(0.2, lambda: type_("y\r\n")).start()
    assert gate.ask_gate(timeout=3) is True


def test_gate_eof_and_dialogue_lines(monkeypatch):
    import os
    import threading
    gate, type_, w = _pipe_reader(monkeypatch)
    threading.Timer(0.1, lambda: type_("why?\n")).start()
    assert gate.read_line("you> ") == "why?"
    os.close(w)
    assert gate.read_line("you> ") is None
    assert gate.ask_gate(timeout=0.5) is False


def test_socratic_verdict_is_regenerated_once_and_flagged(monkeypatch):
    import src.socratic.dialogue as dlg

    class _Fake:
        def __init__(self, replies):
            self.replies, self.users = list(replies), []

        def generate(self, user, system=None, max_tokens=0):
            self.users.append(user)
            return self.replies.pop(0)

    fake = _Fake(["You should steal the drug.", "What weighs more for you here?"])
    monkeypatch.setattr(dlg, "get_client", lambda role: fake)
    s = dlg.SocraticSession(central_script="script", testimony_summary="t")
    assert s.respond("Should Heinz steal?") == "What weighs more for you here?"
    assert len(fake.users) == 2 and "REMINDER" in fake.users[1] and s.flags == [False]
    fake.replies = ["The best option is d2.", "I recommend d2."]
    s.respond("Which is best?")
    assert s.flags == [False, True]                   # survived the retry: recorded


def test_socratic_session_logs(monkeypatch):
    import src.socratic.dialogue as dlg

    class _Fake:
        def generate(self, user, system=None, max_tokens=0):
            # A Socratic-style reply (a question, no verdict).
            return "What matters more to you here — the life at stake, or the rule against theft?"

    monkeypatch.setattr(dlg, "get_client", lambda role: _Fake())
    s = dlg.SocraticSession(central_script="script", testimony_summary="[psych] stress harms family")
    r = s.respond("Should Heinz steal the drug?")
    assert "?" in r and len(s.history) == 1
    assert s.history[0][0].startswith("Should Heinz")


def test_testimony_summary_from_bundle():
    from src.socratic.dialogue import testimony_summary_from_bundle
    bundle = {"decisions": [{"testimony": [
        {"discipline": "sociology", "claims": [{"finding": "Theft erodes institutional trust."}]}]}]}
    s = testimony_summary_from_bundle(bundle)
    assert "sociology" in s and "trust" in s
