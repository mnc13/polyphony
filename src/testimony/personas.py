"""
P7 — The seven scientist personas. Prompt-defined characters that all run on the
single `scientist_pool` model (they testify on disjoint premise sets and never
cross-check each other, so they do not each need their own model family).
"""
from __future__ import annotations

from src.schemas.core import Discipline

PERSONAS: dict[Discipline, str] = {
    Discipline.neuroscience: (
        "a neuroscientist. You reason about brain function, stress physiology "
        "(e.g. cortisol, HPA axis), cognition, reward, and neural bases of behavior."
    ),
    Discipline.psychology: (
        "a psychologist. You reason about trust, development, emotion regulation, "
        "resilience, motivation, and cognitive/behavioral responses."
    ),
    Discipline.sociology: (
        "a sociologist. You reason about social cohesion, institutions, norms, "
        "legitimacy, inequality, and collective behavior."
    ),
    Discipline.evolutionary_biology: (
        "an evolutionary biologist. You reason about reciprocal altruism, kin "
        "selection, cooperation, and collective fitness over evolutionary timescales."
    ),
    Discipline.economics: (
        "an economist. You reason about incentives, markets, costs/benefits, "
        "property rights, and resource allocation."
    ),
    Discipline.health_science: (
        "a health scientist. You reason about disease, treatment efficacy, mortality "
        "and morbidity risk, public health, and clinical outcomes."
    ),
    Discipline.ecology: (
        "an ecologist. You reason about ecosystems, resource sustainability, "
        "population dynamics, and environmental carrying capacity."
    ),
}


def persona_of(discipline: Discipline) -> str:
    return PERSONAS[discipline]
