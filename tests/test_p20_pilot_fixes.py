"""
Fixes from the case-001 pilot of the audited pipeline (2026-10-21).

  1. testimony support: "does not establish" hedges, and weak claims opening with a hedge
     (tests in test_p19_audit_fixes.py, next to the earlier hedge tests);
  2. sentence splitting no longer breaks after abbreviations, in faithfulness too;
  3. a model whose reasoning cannot be disabled is sent a reasoning effort instead of
     being left to reason without limit;
  4. a local network outage is waited out, in evidence fetches and in LLM calls.
"""
from __future__ import annotations

import requests


# ------------------------------------------------------------- faithfulness split
def test_script_sentences_keep_abbreviations_together():
    from src.eval.faithfulness import script_sentences
    s = script_sentences("It argued that the driver may face distress, while Mrs. Foot may "
                         "face ostracism. The evidence was sparse in every branch.")
    assert len(s) == 2 and "Mrs. Foot" in s[0]


# ------------------------------------------------------------ reasoning effort
def _resp(content="ok"):
    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})()
    choice = type("C", (), {"message": type("M", (), {"content": content})(),
                            "finish_reason": "stop"})()
    return type("R", (), {"usage": usage, "choices": [choice]})()


def _openrouter_client(monkeypatch, tmp_path, outcomes):
    import src.services.llm_client as lc
    import src.services.telemetry as tel
    from src.services.model_registry import ModelSpec
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    for reg in ("_REASONING_FLAG_UNUSABLE", "_REQUIRE_PARAMS_UNUSABLE",
                "_REASONING_MANDATORY", "_REASONING_EFFORT_UNUSABLE"):
        monkeypatch.setattr(lc, reg, set())
    sent = []

    class _Completions:
        def create(self, **kw):
            sent.append(kw)
            out = outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            return out

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    hp = type("HP", (), {"mandatory_reasoning_effort": "low", "network_outage_wait_s": 0})()
    monkeypatch.setattr(lc, "OpenAI", lambda **kw: _Client())
    monkeypatch.setattr(lc, "get_settings",
                        lambda: type("S", (), {"openrouter_api_key": "k", "hp": hp})())
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    c = lc.LLMClient(ModelSpec(role="opponent", provider="openrouter", model="z-ai/glm-5.3",
                               temperature=0.5))
    return c, sent


def test_mandatory_reasoning_is_capped_by_effort_not_left_unbounded(monkeypatch, tmp_path):
    mandatory = Exception("Error code: 400 - Reasoning is mandatory for this endpoint "
                          "and cannot be disabled")
    c, sent = _openrouter_client(monkeypatch, tmp_path, [mandatory, _resp(), _resp()])
    assert c.generate("argue") == "ok"
    assert sent[0]["extra_body"]["reasoning"] == {"enabled": False}
    assert sent[1]["extra_body"]["reasoning"] == {"effort": "low"}
    # remembered for the rest of the process
    c.generate("argue again")
    assert sent[2]["extra_body"]["reasoning"] == {"effort": "low"}


def test_routing_failure_drops_the_effort_before_require_parameters(monkeypatch, tmp_path):
    mandatory = Exception("Reasoning is mandatory for this endpoint and cannot be disabled")
    routing = Exception("404 No endpoints found that can handle the requested parameters")
    c, sent = _openrouter_client(monkeypatch, tmp_path, [mandatory, routing, _resp()])
    assert c.generate("argue") == "ok"
    assert "reasoning" not in sent[2]["extra_body"]
    assert sent[2]["extra_body"]["provider"] == {"require_parameters": True}


# --------------------------------------------------------------- network outage
def test_wait_for_network_polls_until_the_host_resolves(monkeypatch, tmp_path):
    import src.services.network as net
    import src.services.telemetry as tel
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    polls = {"n": 0}

    def offline_then_back(host):       # every host is down until the 3rd check
        if host != "www.ebi.ac.uk":
            return False
        polls["n"] += 1
        return polls["n"] > 2
    monkeypatch.setattr(net, "resolves", offline_then_back)
    monkeypatch.setattr(net.time, "sleep", lambda s: None)
    monkeypatch.setattr(net, "_GAVE_UP_AT", {})
    assert net.wait_for_network("www.ebi.ac.uk", "europepmc") is True


def test_after_a_wait_runs_out_later_callers_fail_fast(monkeypatch, tmp_path):
    import src.services.network as net
    import src.services.telemetry as tel
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(net, "_GAVE_UP_AT", {})
    monkeypatch.setattr(net, "resolves", lambda host: False)
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(net.time, "monotonic", lambda: next(clock))
    sleeps = []
    monkeypatch.setattr(net.time, "sleep", lambda s: sleeps.append(s))
    assert net.wait_for_network("api.openalex.org", "openalex") is False
    n = len(sleeps)
    assert n > 0
    assert net.wait_for_network("api.openalex.org", "openalex") is False
    assert len(sleeps) == n, "the second caller must not wait the full limit again"


def test_evidence_request_repeats_its_attempt_after_an_outage(monkeypatch, tmp_path):
    import src.evidence.sources.base as base
    import src.services.network as net
    import src.services.telemetry as tel
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    calls = {"get": 0, "resolve": 0}

    class _Ok:
        status_code = 200
        headers: dict = {}

        def raise_for_status(self):
            return None

    def fake_get(*a, **kw):
        calls["get"] += 1
        if calls["get"] == 1:
            raise requests.ConnectionError("Failed to resolve 'www.ebi.ac.uk'")
        return _Ok()

    def fake_resolves(host):
        if host != "www.ebi.ac.uk":
            return False                  # the whole machine is offline ...
        calls["resolve"] += 1
        return calls["resolve"] > 2       # ... for the first two checks

    monkeypatch.setattr(base.requests, "get", fake_get)
    monkeypatch.setattr(net, "resolves", fake_resolves)
    monkeypatch.setattr(net, "_GAVE_UP_AT", {})
    monkeypatch.setattr(net.time, "sleep", lambda s: None)
    client = base.SourceClient()
    client.retries = 0                    # no retry to spend: only the outage wait helps
    before = base.failure_snapshot().get("base", 0)
    assert client._get("https://www.ebi.ac.uk/x") is not None
    assert calls["get"] == 2 and base.failure_snapshot().get("base", 0) == before


def test_llm_backoff_waits_out_an_outage(monkeypatch, tmp_path):
    import src.services.llm_client as lc
    waited = []
    monkeypatch.setattr(lc.network, "wait_for_network",
                        lambda host, caller: waited.append((host, caller)) or True)
    conn = Exception("Connection error.")
    c, sent = _openrouter_client(monkeypatch, tmp_path, [conn, _resp()])
    assert c.generate("argue") == "ok"
    assert waited == [("openrouter.ai", "llm:opponent")]


def test_one_unresolvable_host_while_online_is_not_waited_for(monkeypatch, tmp_path):
    import src.services.network as net
    import src.services.telemetry as tel
    monkeypatch.setattr(tel, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(net, "_GAVE_UP_AT", {})
    monkeypatch.setattr(net, "resolves", lambda host: host != "api.semanticscholar.org")
    slept = []
    monkeypatch.setattr(net.time, "sleep", lambda s: slept.append(s))
    assert net.wait_for_network("api.semanticscholar.org", "s2") is False
    assert slept == []
