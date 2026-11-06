"""
Model registry + heterogeneity enforcement (Design Law 2).

Loads config/models.yaml and resolves each role to a concrete model spec for the
active mode. Then validates heterogeneity:

  - target_family mode : NO two roles may share an organizational FAMILY
                         (the full form of design law 2).
  - interim_openai mode: family law is impossible (one provider), so we enforce
                         the achievable proxy — NO two roles share the same MODEL,
                         and premise_generation != scientist_pool is hard-required.

The validator FAILS LOUDLY at startup. Every later phase resolves its model
through get_model(role), never by hardcoding a model name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from config.settings import MODELS_YAML

# Map a concrete model id -> organizational family. Used by the family-level
# validator (target mode) and to label interim picks with their real family.
_FAMILY_BY_PREFIX: list[tuple[str, str]] = [
    ("openai/", "openai"),
    ("anthropic/", "anthropic"),
    ("google/", "google"),
    ("meta-llama/", "meta"),
    ("mistralai/", "mistral"),
    ("deepseek/", "deepseek"),
    ("qwen/", "alibaba"),
    ("x-ai/", "xai"),
    ("moonshotai/", "moonshot"),
    ("nousresearch/", "nous"),
    # Zhipu ships as "z-ai/" on OpenRouter, not "zhipu/" — both map to one family so a
    # future swap between them still trips the distinct-family validator.
    ("zhipu/", "zhipu"),
    ("z-ai/", "zhipu"),
    ("minimax/", "minimax"),
    ("bytedance-seed/", "bytedance"),
    ("bytedance/", "bytedance"),
    ("cohere/", "cohere"),
    ("nvidia/", "nvidia"),
    ("tencent/", "tencent"),
    ("xiaomi/", "xiaomi"),
    ("baidu/", "baidu"),
    ("amazon/", "amazon"),
    ("microsoft/", "microsoft"),
    ("perplexity/", "perplexity"),
    ("upstage/", "upstage"),
    ("inclusionai/", "inclusionai"),
    ("ai21/", "ai21"),
    # "meta/" and "meta-llama/" are the same org; keep them on one family.
    ("meta/", "meta"),
    # Bare OpenAI model ids (interim mode)
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
]


def family_of(model: str) -> str:
    m = model.lower()
    for prefix, fam in _FAMILY_BY_PREFIX:
        if m.startswith(prefix):
            return fam
    return "unknown"


@dataclass(frozen=True)
class ModelSpec:
    role: str
    provider: str
    model: str
    temperature: float
    reasoning: bool = False

    @property
    def family(self) -> str:
        return family_of(self.model)


class ModelRegistry:
    # Roles that must never coincide regardless of mode.
    HARD_DISTINCT_PAIRS = [("premise_generation", "scientist_pool")]

    def __init__(self, yaml_path: Path | None = None):
        self.path = Path(yaml_path) if yaml_path else MODELS_YAML
        with open(self.path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.mode: str = cfg["active"]
        if self.mode not in ("interim_openai", "target_family"):
            raise ValueError(f"Unknown active mode: {self.mode}")
        self._specs: dict[str, ModelSpec] = {}
        for role, block in cfg["roles"].items():
            spec_cfg = block[self.mode]
            self._specs[role] = ModelSpec(
                role=role,
                provider=spec_cfg["provider"],
                model=spec_cfg["model"],
                temperature=float(spec_cfg.get("temperature", 0.2)),
                reasoning=bool(spec_cfg.get("reasoning", False)),
            )
        self.validate()

    def get(self, role: str) -> ModelSpec:
        if role not in self._specs:
            raise KeyError(f"No model configured for role '{role}'. Roles: {list(self._specs)}")
        return self._specs[role]

    def roles(self) -> list[str]:
        return list(self._specs)

    def validate(self) -> None:
        specs = list(self._specs.values())

        # Hard-distinct pairs (both modes).
        for a, b in self.HARD_DISTINCT_PAIRS:
            if a in self._specs and b in self._specs:
                if self._specs[a].model == self._specs[b].model:
                    raise ValueError(
                        f"HETEROGENEITY VIOLATION: roles '{a}' and '{b}' resolve to the "
                        f"same model '{self._specs[a].model}'. A scientist must never "
                        f"testify on premises it wrote itself."
                    )

        if self.mode == "target_family":
            seen: dict[str, str] = {}
            for s in specs:
                fam = s.family
                if fam == "unknown":
                    raise ValueError(f"Role '{s.role}' model '{s.model}' has an unknown family.")
                if fam in seen:
                    raise ValueError(
                        f"HETEROGENEITY VIOLATION (family): '{s.role}' and '{seen[fam]}' "
                        f"both use family '{fam}'. Design law 2 requires distinct families."
                    )
                seen[fam] = s.role
        else:  # interim_openai — enforce distinct models (family law relaxed)
            seen_m: dict[str, str] = {}
            for s in specs:
                if s.model in seen_m:
                    raise ValueError(
                        f"REPETITION VIOLATION: '{s.role}' and '{seen_m[s.model]}' both use "
                        f"model '{s.model}'. Interim mode requires a distinct model per role."
                    )
                seen_m[s.model] = s.role

    def table(self) -> str:
        header = f"{'ROLE':<24}{'PROVIDER':<12}{'MODEL':<28}{'FAMILY':<10}{'TEMP':<6}{'REASON'}"
        lines = [header, "-" * len(header)]
        for role in self.roles():
            s = self.get(role)
            lines.append(
                f"{role:<24}{s.provider:<12}{s.model:<28}{s.family:<10}"
                f"{s.temperature:<6}{'yes' if s.reasoning else ''}"
            )
        lines.append(f"\nmode = {self.mode}   roles = {len(self._specs)}")
        return "\n".join(lines)


_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_model(role: str) -> ModelSpec:
    return get_registry().get(role)
