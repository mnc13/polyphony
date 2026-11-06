"""
LLM client wrapper.

One class, `LLMClient`, built from a ModelSpec. Responsibilities:
  - structured output: `generate_structured(...)` returns a validated Pydantic
    object (PROCLAIM used free text everywhere — a major source of its brittleness);
  - plain text: `generate(...)` for the few free-form needs (arbiter prose, socratic);
  - per-model quirk handling: reasoning models (o1/o3/o4) take no temperature and
    use max_completion_tokens; the system role is merged into the user turn for them;
  - retry with exponential backoff;
  - per-call logging (role, model, latency, tokens) to runs/llm_calls.jsonl.

Only OpenAI is wired for the interim. The OpenRouter branch mirrors the interface
so switching modes needs no caller change.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Type, TypeVar

from openai import APIConnectionError, LengthFinishReasonError, OpenAI
from pydantic import BaseModel

from config.settings import get_settings, RUNS_DIR
from src.services import network
from src.services.model_registry import ModelSpec, get_model, get_registry

T = TypeVar("T", bound=BaseModel)

_CALL_LOG = RUNS_DIR / "llm_calls.jsonl"


def _is_reasoning(spec: ModelSpec) -> bool:
    m = spec.model.lower()
    return spec.reasoning or m.startswith(("o1", "o3", "o4"))


# Models that reject `response_format={"type": "json_schema"}`. Calling .parse() on
# these is a guaranteed 400, so we skip straight to the JSON-mode fallback instead of
# burning a failed round-trip plus retry backoff on every structured call.
_NO_JSON_SCHEMA = ("gpt-4-turbo", "gpt-4-0", "gpt-4-1106", "gpt-4-vision", "gpt-3.5")


def _supports_json_schema(spec: ModelSpec) -> bool:
    m = spec.model.lower()
    if any(m.startswith(p) for p in _NO_JSON_SCHEMA):
        return False
    # Bare "gpt-4" (no suffix) is the legacy model; "gpt-4o"/"gpt-4.1" are fine.
    return m != "gpt-4"


# Models for which the reasoning flag cannot be sent at all. Two different causes, same
# remedy — omit it:
#   1. the endpoint mandates reasoning (qwen3.8-max-0902 returns 400 "Reasoning is
#      mandatory for this endpoint and cannot be disabled");
#   2. no provider for that model accepts a reasoning parameter, and because
#      require_parameters applies to EVERY parameter in the request, sending it filters
#      every endpoint out — a 404 whose failed_routing_step is "Filter by Parameters"
#      (seen on claude-sonnet-5, gpt-5.4-mini and mistral-large-2512).
# Neither case is discoverable from the catalogue, so both are learned on first refusal
# and remembered for the rest of the process.
_REASONING_FLAG_UNUSABLE: set[str] = set()

# Models for which provider.require_parameters must NOT be sent. require_parameters
# applies to every parameter in the request, and for some models no single endpoint
# accepts the full set the SDK emits — claude-sonnet-5 and gpt-5.4-mini both 404 with
# failed_routing_step "Filter by Parameters" on the SDK's beta .parse() payload and on
# the json_object fallback, yet both honour json_schema correctly without the flag.
# The two groups are disjoint in practice: models that NEED require_parameters (they
# return empty bodies without it) never raise this routing error.
_REQUIRE_PARAMS_UNUSABLE: set[str] = set()

# Models whose endpoint refused reasoning.enabled=false as mandatory (cause 1 above).
# They are not left to reason unbounded: they are sent reasoning.effort instead
# (hp.mandatory_reasoning_effort). On the case-001 pilot, glm-5.3 as opponent reasoned
# for 14-24k tokens a turn, 54% of the case's cost, and 5 of its 14 calls hit the 24k
# ceiling and were billed for nothing. A probe on the same model: 9,647 reasoning tokens
# with the flag omitted, 6 with effort=low, and an answer of the same length. It also
# restores the design's debater parity: the proponent's reasoning is switched off.
_REASONING_MANDATORY: set[str] = set()
# ... and those for which no provider accepts the effort parameter either.
_REASONING_EFFORT_UNUSABLE: set[str] = set()


def _is_reasoning_mandatory(err: Exception) -> bool:
    m = str(err).lower()
    return "reasoning" in m and ("mandatory" in m or "cannot be disabled" in m)


def _is_param_routing_failure(err: Exception) -> bool:
    m = str(err).lower()
    return "filter by parameters" in m or "no endpoints found that can handle" in m


# A truncated call is retried with a larger budget, up to this multiple of what its
# caller asked for (and never past the global ceiling). It used to grow to 24k for every
# role: on case 004 of the no_decision_decomp re-run, a scientist call whose answers run
# 158-465 tokens looped to 2.2k, 8.8k and then 24k (more than half the case's cost)
# before an ordinary attempt answered in 346. A runaway fills any budget; a role that
# genuinely writes long output (debate 12k, arbiter and premises 6k) still reaches 24k.
_ESCALATION_FACTOR = 4
_TOKEN_CEILING = 24000


def _ceiling(requested: int) -> int:
    return min(_TOKEN_CEILING, requested * _ESCALATION_FACTOR)


# Models whose output limit sits below _TOKEN_CEILING. Asking them for more is a 400
# ("max_tokens is too large"), which is not transient, so an escalating retry crashed
# the case instead of returning the cut text. B4 and B5 run on gpt-4o (16,384).
_MODEL_OUTPUT_CAP = {"gpt-4o": 16384, "gpt-4o-mini": 16384, "gpt-4-turbo": 4096}


def _output_cap(model: str) -> int | None:
    return _MODEL_OUTPUT_CAP.get(model.split("/", 1)[-1])


class _BudgetExhausted(RuntimeError):
    """Empty body with finish_reason=length — the token budget was spent on reasoning."""


class _Truncated(_BudgetExhausted):
    """
    Non-empty output cut off at the token limit (finish_reason=length).

    generate() used to return such output as if it were complete: four of the six
    central scripts of the first main run stopped mid-sentence at exactly 3000 tokens
    (case 004 lost decision d6's whole analysis plus sections 5-6), and the completeness
    gate did not notice because the d6 heading had already been written. A truncated
    answer is now retried with a larger budget, like an exhausted one.
    """


_TRANSIENT = ("rate limit", "429", "timeout", "timed out", "502", "503", "504",
              "connection", "overloaded", "temporarily",
              "mid-generation")   # finish_reason=error, see generate()


def _is_transient(err: Exception | None) -> bool:
    """True for transport-level failures that a retry can plausibly fix."""
    if err is None:
        return True
    msg = str(err).lower()
    return any(t in msg for t in _TRANSIENT)


def _log_call(record: dict) -> None:
    try:
        with open(_CALL_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging must never break a run


def _telemetry(spec: ModelSpec, resp, latency_s: float, ok: bool = True,
               note: str | None = None) -> None:
    """Record tokens (prompt/completion/total) + latency to the tagged telemetry log.
    ok=False: billed, but the output was unusable (still counts toward cost)."""
    try:
        from src.services import telemetry
        u = getattr(resp, "usage", None)
        telemetry.record(
            role=spec.role, model=spec.model,
            prompt_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            completion_tokens=getattr(u, "completion_tokens", 0) if u else 0,
            total_tokens=getattr(u, "total_tokens", 0) if u else 0,
            latency_s=latency_s, ok=ok, note=note,
        )
    except Exception:
        pass


def _event(kind: str, **fields) -> None:
    try:
        from src.services import telemetry
        telemetry.record_event(kind, **fields)
    except Exception:
        pass


class LLMClient:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        settings = get_settings()
        if spec.provider == "openai":
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY missing — set it in .env")
            self._client = OpenAI(api_key=settings.openai_api_key)
        elif spec.provider == "openrouter":
            if not settings.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY missing — set it in .env")
            self._client = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url="https://openrouter.ai/api/v1",
            )
        else:
            raise ValueError(f"Unsupported provider: {spec.provider}")

    # ------------------------------------------------------------------ helpers
    def _backoff(self, err: Exception | None, attempt: int) -> None:
        """Sleep before the next attempt; a local network outage is waited out first,
        so the retries are not all spent while this machine is offline."""
        if isinstance(err, APIConnectionError) or "connection" in str(err).lower():
            host = "openrouter.ai" if self.spec.provider == "openrouter" else "api.openai.com"
            network.wait_for_network(host, f"llm:{self.spec.role}")
        time.sleep(min(2 * (2 ** attempt), 30))

    def _messages(self, system: str | None, user: str) -> list[dict]:
        if _is_reasoning(self.spec):
            # Reasoning models: fold the system prompt into the user turn.
            content = f"{system}\n\n{user}" if system else user
            return [{"role": "user", "content": content}]
        msgs: list[dict] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        return msgs

    def _token_param(self, max_tokens: int) -> dict:
        key = "max_completion_tokens" if _is_reasoning(self.spec) else "max_tokens"
        return {key: max_tokens}

    def _extra_body(self) -> dict:
        """
        OpenRouter-only request options. Two of them, both load-bearing.

        require_parameters: OpenRouter fans each model id out across several upstream
        providers. Without this, a request carrying response_format=json_schema can be
        routed to a provider that does not implement it; the call succeeds, the schema is
        silently ignored, and the model returns prose or an empty body. That is what made
        deepseek, kimi and glm all fail validation with input_value={} on the first real
        run. With it, OpenRouter only routes to providers that honour every parameter sent.

        reasoning disabled: most of this roster can emit reasoning tokens, which are billed
        as output AND consume the max_tokens budget before the first character of JSON is
        written. That is the other failure seen on that run -- glm-5.3 truncated mid-string
        at column 2317. Roles that genuinely want reasoning carry reasoning: true in
        models.yaml and are excluded here.
        """
        if self.spec.provider != "openrouter":
            return {}
        extra: dict = {}
        if self.spec.model not in _REQUIRE_PARAMS_UNUSABLE:
            extra["provider"] = {"require_parameters": True}
        if not _is_reasoning(self.spec):
            if self.spec.model not in _REASONING_FLAG_UNUSABLE:
                extra["reasoning"] = {"enabled": False}
            elif (self.spec.model in _REASONING_MANDATORY
                  and self.spec.model not in _REASONING_EFFORT_UNUSABLE):
                effort = get_settings().hp.mandatory_reasoning_effort
                if effort:
                    extra["reasoning"] = {"effort": effort}
        return extra

    def _budget(self, max_tokens: int) -> int:
        """
        Headroom for models whose reasoning we cannot switch off.

        Reasoning tokens are drawn from the same max_tokens pool as the output, so a
        budget sized for the JSON alone gets spent before the first character is
        written. This bit at the worst possible moment: glm-5.3 hit a routing 404, the
        escalation relaxed BOTH optional flags to get it routed, and dropping
        reasoning.enabled=false then let it reason without limit — it consumed all 4000
        tokens of a debate turn and returned an empty body, repeatedly. Granting the
        headroom up front is better than discovering it through _BudgetExhausted
        retries, which can exhaust max_retries when transient errors are also in play.
        Output bills on usage, not on the ceiling, so this is free when unused.
        """
        if self.spec.model in _REASONING_FLAG_UNUSABLE or _is_reasoning(self.spec):
            return min(max(max_tokens, max_tokens * 3), 24000)
        return max_tokens

    def _ceiling(self, requested: int) -> int:
        cap = _output_cap(self.spec.model)
        return min(_ceiling(requested), cap) if cap else _ceiling(requested)

    def _base_params(self, max_tokens: int) -> dict:
        budget = self._budget(max_tokens)
        cap = _output_cap(self.spec.model)
        params: dict = {"model": self.spec.model,
                        **self._token_param(min(budget, cap) if cap else budget)}
        if not _is_reasoning(self.spec):
            params["temperature"] = self.spec.temperature
        extra = self._extra_body()
        if extra:
            params["extra_body"] = extra
        return params

    # -------------------------------------------------------------------- calls
    def _escalate(self, err: Exception | None, max_tokens: int,
                  ceiling: int = _TOKEN_CEILING) -> tuple[str, int]:
        """
        Decide what to do after a failed attempt. Returns (action, max_tokens) where
        action is "retry" (immediately, with whatever was relaxed), "wait" (transient,
        back off first) or "stop" (deterministic, further attempts cost money and
        change nothing).

        Shared by generate() and generate_structured() on purpose: the arbiter reached
        this code through generate() and 404'd on a parameter set that
        generate_structured() already knew how to relax, because the escalation had been
        written into only one of the two.
        """
        if err is None:
            return "wait", max_tokens
        if isinstance(err, _Truncated) and max_tokens < ceiling:
            new = min(max_tokens * 2, ceiling)
            _event("token_budget_raised", role=self.spec.role, model=self.spec.model,
                   reason="truncated", old=max_tokens, new=new)
            return "retry", new
        if isinstance(err, _BudgetExhausted) and max_tokens < ceiling:
            new = min(max_tokens * 4, ceiling)
            _event("token_budget_raised", role=self.spec.role, model=self.spec.model,
                   reason="budget_exhausted", old=max_tokens, new=new)
            return "retry", new
        # The relaxations below change the request for this model for the rest of the
        # process -- including switching its reasoning back on, which also triples its
        # budget (see _budget). That changes how the role behaves, so each one is
        # recorded as an event instead of happening silently.
        if _is_reasoning_mandatory(err) and self.spec.model not in _REASONING_FLAG_UNUSABLE:
            _REASONING_FLAG_UNUSABLE.add(self.spec.model)
            _REASONING_MANDATORY.add(self.spec.model)
            _event("param_relaxation", role=self.spec.role, model=self.spec.model,
                   dropped=["reasoning.enabled=false"], cause="reasoning mandatory",
                   replaced_with=f"reasoning.effort={get_settings().hp.mandatory_reasoning_effort}")
            return "retry", max(max_tokens, int(max_tokens * 2.5))
        if _is_param_routing_failure(err):
            # One parameter at a time, so a model's reasoning control is not given up as
            # collateral. Dropping reasoning.enabled=false together with
            # require_parameters on a routing 404 used to leave a mandatory-reasoning
            # model (glm-5.3 has produced both errors) reasoning without limit: the 400
            # that leads to reasoning.effort could no longer occur.
            m = self.spec.model
            # 1. The effort parameter, if one was sent, is the newest suspect.
            if m in _REASONING_MANDATORY and m not in _REASONING_EFFORT_UNUSABLE:
                _REASONING_EFFORT_UNUSABLE.add(m)
                _event("param_relaxation", role=self.spec.role, model=m,
                       dropped=["reasoning.effort"], cause="routing failure")
                return "retry", max_tokens
            # 2. require_parameters, keeping the reasoning flag. Without it a provider
            #    ignores a flag it does not support, and a mandatory endpoint answers
            #    with the 400 above, so the effort cap still applies; the effort is sent
            #    again, since it was only dropped as a suspect.
            if m not in _REQUIRE_PARAMS_UNUSABLE:
                _REQUIRE_PARAMS_UNUSABLE.add(m)
                _REASONING_EFFORT_UNUSABLE.discard(m)
                _event("param_relaxation", role=self.spec.role, model=m,
                       dropped=["provider.require_parameters"], cause="routing failure")
                return "retry", max_tokens
            # 3. Only then the reasoning flag itself.
            if m not in _REASONING_FLAG_UNUSABLE:
                _REASONING_FLAG_UNUSABLE.add(m)
                _event("param_relaxation", role=self.spec.role, model=m,
                       dropped=["reasoning.enabled=false"], cause="routing failure")
                return "retry", max_tokens
        return ("wait", max_tokens) if _is_transient(err) else ("stop", max_tokens)

    def generate(
        self,
        user: str,
        system: str | None = None,
        max_tokens: int = 1500,
        max_retries: int = 6,
        retry_truncated: bool = False,
    ) -> str:
        """
        Plain-text completion with retry.

        retry_truncated: treat output cut off at the token limit as a failure and retry
        with a doubled budget (up to 4x the request, at most 24k; there the cut text is returned
        rather than crashing the case). Opt-in, because a cap is part of some callers'
        specification -- the baselines replicate fixed-length prompts and the Socratic
        turn is meant to be short -- while for the arbiter a cut-off script is a
        defect: four of six first-run scripts lost their endings that way.
        """
        last_err: Exception | None = None
        ceiling = self._ceiling(max_tokens)
        for attempt in range(max_retries):
            # Rebuilt every attempt: _base_params reads the relaxation registries, and
            # building it once outside the loop meant a relaxation could never take
            # effect on a later attempt.
            params = self._base_params(max_tokens)
            params["messages"] = self._messages(system, user)
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(**params)
                content = resp.choices[0].message.content or ""
                dt = time.time() - t0
                _log_call({
                    "role": self.spec.role, "model": self.spec.model, "kind": "text",
                    "latency_s": round(dt, 2),
                    "tokens": getattr(resp.usage, "total_tokens", None), "ok": True,
                })
                finish = getattr(resp.choices[0], "finish_reason", None)
                truncated = finish == "length"
                # OpenRouter reports an upstream failure mid-generation as
                # finish_reason="error" with the partial text as content, and that was
                # returned as the answer -- the likely cause of a B1 analysis that ended
                # mid-sentence at 832 of 3,000 tokens (finish reasons were not logged then;
                # any other than "stop" now goes into the call's telemetry note).
                failed = finish == "error"
                will_retry = retry_truncated and truncated and max_tokens < ceiling
                _telemetry(self.spec, resp, dt,
                           ok=bool(content.strip()) and not will_retry and not failed,
                           note="truncated" if truncated else (
                               f"finish_reason={finish}" if finish not in ("stop", None) else None))
                if failed:
                    raise ValueError(f"{self.spec.role}/{self.spec.model}: provider failed "
                                     f"mid-generation (finish_reason=error)")
                if not content.strip():
                    if truncated:
                        raise _BudgetExhausted(
                            f"{self.spec.role}/{self.spec.model}: reasoning consumed the "
                            f"entire {max_tokens}-token budget (empty completion)")
                    raise ValueError("empty completion")
                if will_retry:
                    raise _Truncated(
                        f"{self.spec.role}/{self.spec.model}: output cut off at the "
                        f"{max_tokens}-token limit (finish_reason=length)")
                if truncated:
                    _event("output_truncated", role=self.spec.role, model=self.spec.model,
                           max_tokens=max_tokens, retried=False)
                return content
            except Exception as e:  # noqa: BLE001
                last_err = e
                _log_call({"role": self.spec.role, "model": self.spec.model,
                           "kind": "text", "ok": False, "error": str(e)})
            action, max_tokens = self._escalate(last_err, max_tokens, ceiling)
            if action == "stop":
                break
            if action == "wait":
                self._backoff(last_err, attempt)
        raise RuntimeError(f"generate failed for {self.spec.role}/{self.spec.model}: {last_err}")

    def generate_structured(
        self,
        user: str,
        schema: Type[T],
        system: str | None = None,
        max_tokens: int = 2000,
        max_retries: int = 8,
    ) -> T:
        """
        Structured completion validated against `schema`. Prefers the SDK's native
        parse (json_schema-constrained); falls back to JSON mode + manual validation.

        Models without json_schema support skip the native attempt entirely — for
        those the fallback is the ONLY working path, and attempting .parse() first
        just spends a guaranteed 400 plus backoff on every single call.
        """
        native = _supports_json_schema(self.spec)
        last_err: Exception | None = None
        ceiling = self._ceiling(max_tokens)
        for attempt in range(max_retries):
            if native:
                t0 = time.time()
                resp = None
                try:
                    params = self._base_params(max_tokens)
                    params["messages"] = self._messages(system, user)
                    # Native structured output (OpenAI SDK >= 1.40 supports .parse).
                    resp = self._client.beta.chat.completions.parse(
                        response_format=schema, **params
                    )
                    parsed = resp.choices[0].message.parsed
                    if parsed is None:
                        if resp.choices[0].finish_reason == "length":
                            raise _BudgetExhausted(
                                f"{self.spec.role}/{self.spec.model}: native parse "
                                f"truncated at {max_tokens} tokens "
                                f"(finish_reason=length)"
                            )
                        raise ValueError("parse returned None")
                    dt = time.time() - t0
                    _log_call({
                        "role": self.spec.role, "model": self.spec.model,
                        "kind": "structured", "latency_s": round(dt, 2),
                        "tokens": getattr(resp.usage, "total_tokens", None), "ok": True,
                    })
                    _telemetry(self.spec, resp, dt)
                    return parsed
                except LengthFinishReasonError as e:
                    # The SDK raises this itself when output hits the length limit, so the
                    # _BudgetExhausted branch above never ran for it: the billed call
                    # reached neither telemetry nor the escalation, and the JSON fallback
                    # then re-ran at the SAME budget (126 such errors are on record).
                    last_err = _BudgetExhausted(
                        f"{self.spec.role}/{self.spec.model}: native parse truncated at "
                        f"{max_tokens} tokens (finish_reason=length)")
                    _log_call({"role": self.spec.role, "model": self.spec.model,
                               "kind": "structured", "ok": False, "error": str(last_err)})
                    _telemetry(self.spec, getattr(e, "completion", None), time.time() - t0,
                               ok=False, note="truncated")
                    action, max_tokens = self._escalate(last_err, max_tokens, ceiling)
                    if action != "stop":
                        continue
                    # at the ceiling: one JSON-mode attempt below, then stop
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    _log_call({"role": self.spec.role, "model": self.spec.model,
                               "kind": "structured", "ok": False, "error": str(e)})
                    # A response that came back and was unusable was still billed.
                    billed = resp if resp is not None else getattr(e, "completion", None)
                    if billed is not None:
                        _telemetry(self.spec, billed, time.time() - t0, ok=False,
                                   note=type(e).__name__)
                    if isinstance(e, _BudgetExhausted):
                        # Same budget, same truncation: widen it before trying again
                        # rather than paying for a JSON-mode call that will cut off too.
                        action, max_tokens = self._escalate(e, max_tokens, ceiling)
                        if action != "stop":
                            continue
            # JSON-mode fallback — the primary path for models without json_schema.
            try:
                fb = self._json_mode_fallback(system, user, schema, max_tokens)
                if fb is not None:
                    return fb
            except Exception as e:  # noqa: BLE001
                last_err = e
                _log_call({"role": self.spec.role, "model": self.spec.model,
                           "kind": "structured_fallback", "ok": False, "error": str(e)})
            # One shared escalation path with generate(), so the two cannot drift:
            # relax a rejected parameter, widen an exhausted budget, back off on a
            # transient error, and stop immediately on anything deterministic, where
            # further attempts would bill more calls and fail in exactly the same way.
            action, max_tokens = self._escalate(last_err, max_tokens, ceiling)
            if action == "stop":
                break
            if action == "wait":
                self._backoff(last_err, attempt)
        raise RuntimeError(
            f"generate_structured failed for {self.spec.role}/{self.spec.model}: {last_err}"
        )

    def _json_mode_fallback(self, system, user, schema: Type[T], max_tokens) -> T | None:
        instruction = (
            f"{user}\n\nReturn ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        params = self._base_params(max_tokens)
        params["messages"] = self._messages(system, instruction)
        params["response_format"] = {"type": "json_object"}
        t0 = time.time()
        resp = self._client.chat.completions.create(**params)
        dt = time.time() - t0
        # Record BEFORE validation: the call already happened and cost tokens whether
        # or not the payload validates. Roles pinned to models without json_schema
        # support run entirely through here, and were previously invisible to the
        # token report — which understated the per-sample budget.
        _log_call({
            "role": self.spec.role, "model": self.spec.model, "kind": "structured",
            "via": "json_mode_fallback", "latency_s": round(dt, 2),
            "tokens": getattr(resp.usage, "total_tokens", None), "ok": True,
        })
        truncated = getattr(resp.choices[0], "finish_reason", None) == "length"
        _telemetry(self.spec, resp, dt, ok=not truncated, note="truncated" if truncated else None)
        raw = resp.choices[0].message.content
        if (not raw or not raw.strip()) and resp.choices[0].finish_reason == "length":
            # Mandatory-reasoning endpoints draw reasoning tokens from the SAME
            # max_tokens pool, so a budget sized for the JSON alone is spent before the
            # first character is written and the body comes back empty. Seen on the whole
            # qwen flagship line: 1000 of 1000 completion tokens were reasoning.
            raise _BudgetExhausted(
                f"{self.spec.role}/{self.spec.model}: reasoning consumed the entire "
                f"{max_tokens}-token budget (finish_reason=length, empty body)"
            )
        if not raw or not raw.strip():
            # Substituting "{}" here used to turn an empty body into a misleading
            # "Field required [input_value={}]" validation error, which hid the real
            # cause: the provider returned no content at all.
            raise ValueError(
                f"{self.spec.role}/{self.spec.model}: provider returned an EMPTY body "
                f"(finish_reason={resp.choices[0].finish_reason}). Usually either the "
                f"routed provider ignored response_format, or reasoning tokens consumed "
                f"the whole max_tokens budget."
            )
        try:
            return schema.model_validate_json(raw)
        except Exception as e:
            snippet = " ".join(raw.strip()[:240].split())
            if resp.choices[0].finish_reason == "length":
                # Not a schema mismatch: the model wrote valid JSON and ran out of
                # budget mid-string, so the payload is simply cut off. Escalating the
                # budget is the fix; re-prompting at the same cap would truncate again.
                raise _BudgetExhausted(
                    f"{self.spec.role}/{self.spec.model}: response TRUNCATED at "
                    f"{max_tokens} tokens (finish_reason=length). Starts: {snippet!r}"
                ) from e
            raise ValueError(
                f"{self.spec.role}/{self.spec.model}: body did not match "
                f"{schema.__name__} (finish_reason={resp.choices[0].finish_reason}). "
                f"Raw starts: {snippet!r}"
            ) from e


_clients: dict[str, LLMClient] = {}

# When set, EVERY role resolves to this one model — used by the B4 homogeneous-debate
# baseline (the point of which is to remove model diversity). Because all callers
# invoke this same get_client function, flipping the flag affects the whole pipeline.
_force_model: str | None = None


def single_model_spec(role: str, model: str, temperature: float) -> ModelSpec:
    """
    A spec for one named model, routed the way the active roster is.

    B4 and B5 name an OpenAI model ("gpt-4o"). They used to build an OpenAI-provider
    client unconditionally, so under the OpenRouter roster they needed a second, separate
    key and bill; the same model is reached through OpenRouter as "openai/gpt-4o".
    """
    if get_registry().mode == "target_family":
        return ModelSpec(role=role, provider="openrouter", temperature=temperature,
                         model=model if "/" in model else f"openai/{model}")
    return ModelSpec(role=role, provider="openai", temperature=temperature,
                     model=model.split("/", 1)[1] if model.startswith("openai/") else model)


def set_force_model(model: str | None) -> None:
    global _force_model
    _force_model = model or None
    _clients.clear()


def forced_model() -> str | None:
    return _force_model


def resolved_spec(role: str) -> ModelSpec:
    """
    The spec get_client(role) runs on.

    Under a forced model each role keeps its registry temperature. The force used to
    set 0.4 for every role, so B4 also changed stakeholder extraction (0.0), the
    arbiter (0.2) and the debaters (0.5) and confounded model diversity with sampling.
    """
    base = get_model(role)
    if _force_model is None:
        return base
    # A reasoning role's registry temperature (1.0, fixed by those models) means
    # nothing for a forced non-reasoning model; the debate default stands in.
    temperature = 0.4 if base.reasoning else base.temperature
    return single_model_spec(role, _force_model, temperature)


def get_client(role: str) -> LLMClient:
    """Cached client per role, built from the registry (or the forced model, if set)."""
    key = f"__force__{role}" if _force_model is not None else role
    if key not in _clients:
        _clients[key] = LLMClient(resolved_spec(role))
    return _clients[key]
