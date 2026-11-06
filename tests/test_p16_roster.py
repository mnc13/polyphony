"""
P16 — the proposed OpenRouter roster satisfies the circularity constraints.

These read the `target_family` column of config/models.yaml DIRECTLY rather than through
the active mode, so they hold while `active` is still interim_openai and the roster is
awaiting approval. They lock in the properties the roster was selected for, so a later
edit that quietly reintroduces a shared family fails here instead of mid-run.

No API calls: these tests only read the YAML and one source file.
"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.services.model_registry import ModelRegistry, family_of  # noqa: E402

ROLES = ["stakeholder_extraction", "action_decomposition", "decision_generation",
         "premise_generation", "scientist_pool", "proponent", "opponent", "arbiter", "socratic"]


@pytest.fixture(scope="module")
def target():
    """role -> target_family spec, independent of which mode is currently active."""
    cfg = yaml.safe_load((ROOT / "config" / "models.yaml").read_text(encoding="utf-8"))
    return {r: b["target_family"] for r, b in cfg["roles"].items()}


def fam(target, role):
    return family_of(target[role]["model"])


def test_active_mode_still_loads(target):
    """Whatever mode is active must resolve cleanly — the roster edit must not have
    broken the interim path that runs today."""
    reg = ModelRegistry()
    assert reg.mode in ("interim_openai", "target_family")
    for r in ROLES:
        assert reg.get(r).model


def test_every_target_role_present_and_mapped(target):
    assert set(target) == set(ROLES)
    for r in ROLES:
        assert target[r]["provider"] == "openrouter"
        assert fam(target, r) != "unknown", f"{r} -> {target[r]['model']} has no family mapping"


def test_all_families_distinct(target):
    seen = {}
    for r in ROLES:
        f = fam(target, r)
        assert f not in seen, f"{r} shares family '{f}' with {seen[f]}"
        seen[f] = r
    assert len(seen) == len(ROLES)


def test_arbiter_isolated_from_what_it_compiles(target):
    """The arbiter weighs and compiles text these roles wrote; a shared family would let
    same-family over-trust decide which arguments survive into the central script."""
    arb = fam(target, "arbiter")
    for r in ("proponent", "opponent", "scientist_pool"):
        assert fam(target, r) != arb, f"arbiter shares a family with {r}"


def test_debaters_differ(target):
    assert fam(target, "proponent") != fam(target, "opponent")


def test_scientist_never_testifies_on_own_premises(target):
    assert target["premise_generation"]["model"] != target["scientist_pool"]["model"]
    assert fam(target, "premise_generation") != fam(target, "scientist_pool")


def test_scientists_outside_both_debaters(target):
    sci = fam(target, "scientist_pool")
    assert sci != fam(target, "proponent")
    assert sci != fam(target, "opponent")


def test_zhipu_and_meta_prefixes_unify():
    """OpenRouter ships Zhipu as z-ai/, not zhipu/. Both spellings must land on one
    family or the distinct-family validator would pass a z-ai + zhipu pair."""
    assert family_of("z-ai/glm-5.3") == family_of("zhipu/glm-4") == "zhipu"
    assert family_of("meta/llama-4") == family_of("meta-llama/llama-3.3-70b-instruct") == "meta"


def test_premise_cap_has_headroom():
    """The prompt asks for 4-10 premises. At 2000 the payload truncated (4 of 8 calls on
    sample 001 returned exactly 2000) and the JSON-mode fallback re-ran the whole call."""
    src = (ROOT / "src" / "premises" / "generate.py").read_text(encoding="utf-8")
    assert "max_tokens=6000" in src
    assert "max_tokens=2000" not in src
