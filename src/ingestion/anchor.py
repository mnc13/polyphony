"""
P4 — Anchor selection (Step 1).

The user chooses the empirical yardstick the case is judged against, BEFORE anything
else runs. This is the blueprint's "bounded relativism via naturalistic anchors":
the value choice (which anchor) is the user's, but the menu contains only
empirically measurable anchors, so whatever is chosen stays objectively tractable.

Design note — dimensions vs anchors:
  All three anchors are measured across the SAME four empirical science domains
  (NEURO / PSYCH / SOCIO / EVOBIO), because those are the disciplines that supply
  evidence. What differs per anchor is the NORMATIVE FRAMING — what counts as a
  positive vs negative effect on those dimensions. Survivability asks "does this
  help the agent/collective persist?"; well-being asks "does this improve welfare
  and reduce suffering?"; flourishing asks "does this realize potential / thriving?".
  That framing string flows into premise generation and scientist testimony, so the
  same evidence can be weighed differently under different anchors — which is exactly
  what the anchor-sensitivity evaluation later measures.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.schemas.core import Anchor, Dimension


@dataclass(frozen=True)
class AnchorConfig:
    anchor: Anchor
    label: str
    definition: str            # injected into prompts as the normative yardstick
    positive_direction: str    # what "+1 / enhances" means under this anchor
    dimensions: tuple[Dimension, ...]


_ALL_DIMS = (Dimension.NEURO, Dimension.PSYCH, Dimension.SOCIO, Dimension.EVOBIO)

_CONFIGS: dict[Anchor, AnchorConfig] = {
    Anchor.survivability: AnchorConfig(
        anchor=Anchor.survivability,
        label="Survivability",
        definition=(
            "Survivability is the capacity of actions to protect or improve the "
            "persistence of conscious biological agents and the collectives they "
            "belong to — spanning individual survival (health, cognition, safety) "
            "and collective survival (cooperation, social cohesion, group fitness)."
        ),
        positive_direction=(
            "An effect is POSITIVE (+1) when it increases an agent's or group's "
            "capacity to survive and persist; NEGATIVE (-1) when it decreases it."
        ),
        dimensions=_ALL_DIMS,
    ),
    Anchor.well_being: AnchorConfig(
        anchor=Anchor.well_being,
        label="Well-being",
        definition=(
            "Well-being is the balance of an agent's welfare — subjective (life "
            "satisfaction, positive vs negative affect) and objective (health, "
            "security, functioning) — and the reduction of suffering."
        ),
        positive_direction=(
            "An effect is POSITIVE (+1) when it improves welfare or reduces "
            "suffering; NEGATIVE (-1) when it worsens welfare or increases suffering."
        ),
        dimensions=_ALL_DIMS,
    ),
    Anchor.flourishing: AnchorConfig(
        anchor=Anchor.flourishing,
        label="Flourishing",
        definition=(
            "Flourishing (eudaimonic thriving) is the realization of an agent's "
            "potential — growth, meaning, autonomy, competence, and relationships — "
            "beyond mere survival or momentary satisfaction."
        ),
        positive_direction=(
            "An effect is POSITIVE (+1) when it advances an agent's growth, meaning, "
            "or realization of potential; NEGATIVE (-1) when it stunts it."
        ),
        dimensions=_ALL_DIMS,
    ),
}


def get_anchor_config(anchor: Anchor) -> AnchorConfig:
    return _CONFIGS[anchor]


def menu() -> str:
    lines = ["Choose the empirical anchor to judge this case against:"]
    for i, a in enumerate(Anchor, start=1):
        cfg = _CONFIGS[a]
        lines.append(f"  {i}. {cfg.label} — {cfg.definition}")
    return "\n".join(lines)


def select_anchor(choice: str | int | Anchor) -> AnchorConfig:
    """Resolve a menu choice (index, name, or Anchor) to its config."""
    if isinstance(choice, Anchor):
        return _CONFIGS[choice]
    if isinstance(choice, int) or (isinstance(choice, str) and choice.isdigit()):
        idx = int(choice) - 1
        anchors = list(Anchor)
        if not 0 <= idx < len(anchors):
            raise ValueError(f"anchor choice out of range: {choice}")
        return _CONFIGS[anchors[idx]]
    key = str(choice).strip().lower().replace("-", "_").replace(" ", "_")
    for a in Anchor:
        if a.value == key or _CONFIGS[a].label.lower().replace("-", "_") == key:
            return _CONFIGS[a]
    raise ValueError(f"unknown anchor: {choice}")
