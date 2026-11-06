"""
P17 — the six-case experiment dataset.

Locks in the properties six_cases.csv was built for, so a regeneration or a hand edit
that reintroduces pre-written options, mojibake or an unknown stratum fails here rather
than silently changing what the experiment measures.

No API calls: these read the CSV through the pipeline's own loader.
"""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.storage.cases import load_cases  # noqa: E402

CSV = ROOT / "data" / "six_cases.csv"
OPTION_LINE = re.compile(r"^\s*[A-E][.)]\s", re.M)


@pytest.fixture(scope="module")
def cases():
    if not CSV.exists():
        pytest.skip("data/six_cases.csv not present")
    return load_cases(CSV)


def test_exactly_six_cases(cases):
    assert len(cases) == 6
    assert [c.sample_id for c in cases] == ["001", "002", "003", "004", "005", "006"]


def test_no_prewritten_options(cases):
    """Cases 2-6 shipped with 'Your Choices: A. ... D. ...' in the source text, which is
    the output of action_decomposition and decision_generation. Leaving it in would hand
    two stages their own answer and inflate the decision count."""
    for c in cases:
        assert not OPTION_LINE.search(c.text), f"sample {c.sample_id} still lists options"
        assert "your choices" not in c.text.lower()


def test_no_mojibake(cases):
    """The source text carried 'â€™' for apostrophes. repair_text handles that variant,
    but the CSV is written pre-repaired so nothing depends on it running later."""
    for c in cases:
        for bad in ("â", "€", "Â", "�"):
            assert bad not in c.text, f"sample {c.sample_id} contains mojibake {bad!r}"


def test_strata_resolved(cases):
    """src/eval/metrics.py breaks every result down by stratum; 'unknown' would collapse
    the whole run into one undifferentiated bucket."""
    strata = {c.sample_id: c.stratum for c in cases}
    assert "unknown" not in strata.values(), strata
    assert strata["001"] == "trolley", "case 1 is Foot's original trolley problem"
    assert all(strata[s] == "demonstration" for s in
               ("002", "003", "004", "005", "006"))


def test_cases_retain_their_dilemma(cases):
    """Stripping options removed up to 54% of a case; what remains must still describe
    the situation well enough for action decomposition to have something to work with."""
    for c in cases:
        assert len(c.text) > 400, f"sample {c.sample_id} is only {len(c.text)} chars"
        assert len(c.text.split()) > 70


def test_runner_defaults_to_this_dataset():
    for script in ("run_full.py", "run_baseline.py"):
        src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert '"six_cases.csv"' in src, f"{script} does not default to six_cases.csv"
        assert '"all_combined_moral_cases.csv"' not in src
