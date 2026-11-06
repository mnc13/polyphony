"""P11 — Tier-1 metrics + aggregation + anchor sensitivity (all deterministic, no LLM)."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.anchor_sensitivity import compare_anchors
from src.eval.faithfulness import FAITHFULNESS_DEF
from src.eval.metrics import aggregate, compute_metrics

_P1 = "Diverting the trolley prevents fatal impact injuries to the five workmen."
_P2 = "Shared meals strengthen trust among the town residents."
_N1 = ("Blunt-force impact by rail vehicles is a leading cause of fatal injury among "
       "track workers; preventing the collision prevents the fatal injuries.")


def _bundle(cov=1.0, faith=0.9):
    return {
        "stakeholder_register": {"stakeholders": [{"id": "S1"}, {"id": "S2"}]},
        "decisions": [{
            "decision_id": "d1",
            "tps": {"premises": [{"id": "p1", "text": _P1, "dimension": "NEURO"},
                                 {"id": "p2", "text": _P2, "dimension": "SOCIO"}]},
            "testimony": [{"claims": [{"nli_supported": True, "cited_nodes": ["TEG-1"]},
                                      {"nli_supported": False}]}],
            "debate": {"coverage": cov, "turns": [
                {"round": 1, "speaker": "proponent", "premises_addressed": ["p1", "p2"],
                 "evidence_cited": ["TEG-1"],
                 # the impact paper is cited for p1 AND (wrongly) for p2
                 "claims": [{"premise_id": "p1", "evidence_nodes": ["TEG-1"]},
                            {"premise_id": "p2", "evidence_nodes": ["TEG-1"]}]}]},
            "teg": {"nodes": [{"node_id": "TEG-1", "doi": "10.1/x", "text": _N1},
                              {"node_id": "TEG-2", "doi": None, "text": "Unrelated."}]},
        }],
        "sit_rows": [
            {"decision_id": "d1", "speaker": "proponent", "stakeholder_id": "S1",
             "dimension": "NEURO", "polarity": 1,
             "evidence_node": "TEG-1", "rebutted": False, "nli_supported": True},
            {"decision_id": "d1", "speaker": "opponent", "stakeholder_id": "S2",
             "dimension": "SOCIO", "polarity": -1,
             "evidence_node": None, "rebutted": False, "nli_supported": False},
        ],
        "faithfulness": {"fraction_entailed": faith, "measured": True,
                         "definition": FAITHFULNESS_DEF},
    }


def test_compute_metrics():
    m = compute_metrics(_bundle())
    # recomputed: p1 is addressed with a topically relevant node; p2 is only LISTED by
    # a turn whose single citation is about trolley impact, not shared meals
    assert m["premise_coverage"] == 0.5
    assert m["premise_coverage_as_run"] == 1.0
    assert m["citation_integrity"] == 1.0          # 1 cited row, it is nli_supported
    assert m["stakeholder_coverage"] == 0.5         # S1 supported, S2 not (2 SR)
    assert m["script_faithfulness"] == 0.9
    assert m["provenance_rate"] == 0.5              # 1 of 2 distinct chunks has an id
    assert m["doi_rate"] == 0.5
    assert m["testimony_grounding"] == 0.5
    assert m["sit_profile"]["cited"] == 1 and m["sit_profile"]["asserted"] == 1


def test_aggregate_by_stratum():
    agg = aggregate([("ordinary", _bundle(cov=0.8)), ("trolley", _bundle(cov=1.0))])
    assert agg["n_samples"] == 2
    assert set(agg["by_stratum"]) == {"ordinary", "trolley"}
    assert agg["by_stratum"]["ordinary"]["premise_coverage_as_run"] == 0.8


def test_placeholder_faithfulness_is_not_averaged_in():
    measured = _bundle(faith=0.8)
    placeholder = _bundle(faith=1.0)
    placeholder["faithfulness"] = {"fraction_entailed": 1.0, "ungrounded_sentences": []}
    agg = aggregate([("x", measured), ("x", placeholder)])
    assert agg["overall"]["script_faithfulness"] == 0.8
    assert agg["n_measured"]["script_faithfulness"] == 1
    explicit = _bundle(faith=1.0)
    assert compute_metrics(explicit)["script_faithfulness"] == 1.0
    assert math.isnan(compute_metrics(placeholder)["script_faithfulness"])


def test_faithfulness_under_an_older_definition_is_not_averaged_in():
    old = _bundle(faith=0.95)
    old["faithfulness"]["definition"] = "v2-sentences-windows-substantive-context"
    agg = aggregate([("x", _bundle(faith=0.6)), ("x", old)])
    assert agg["overall"]["script_faithfulness"] == 0.6
    assert agg["n_measured"]["script_faithfulness"] == 1


def test_legacy_transcripts_without_claim_premise_ids_use_the_turn_rule():
    b = _bundle()
    b["decisions"][0]["debate"]["turns"][0]["claims"] = [{"evidence_nodes": ["TEG-1"]}]
    assert compute_metrics(b)["premise_coverage"] == 0.5


def test_pooled_integrity_counts():
    a, b = _bundle(), _bundle()
    b["sit_rows"][0]["nli_supported"] = False
    agg = aggregate([("x", a), ("x", b)])
    assert agg["counts"]["sit_cited"] == 2 and agg["counts"]["sit_cited_supported"] == 1
    assert agg["pooled"]["citation_integrity"] == 0.5


def test_nan_metric_does_not_poison_the_mean():
    a = _bundle()
    b = _bundle()
    b["decisions"][0]["testimony"] = [{"claims": [{"nli_supported": False}]}]  # cites nothing
    agg = aggregate([("x", a), ("x", b)])
    assert agg["overall"]["testimony_citation_integrity"] == 1.0
    assert agg["n_measured"]["testimony_citation_integrity"] == 1


def test_anchor_sensitivity():
    b_surv = _bundle()
    b_well = _bundle()
    # flip a polarity so signatures differ
    b_well["sit_rows"][0]["polarity"] = -1
    cmp = compare_anchors({"survivability": b_surv, "well_being": b_well})
    key = "survivability|well_being"
    assert key in cmp["claim_signature_overlap"]
    assert 0.0 <= cmp["claim_signature_overlap"][key] < 1.0
