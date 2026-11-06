"""
Shared Pydantic models — the typed data that flows between phases.

Kept in one place so every phase agrees on structure and structured-output calls
validate against the same definitions. Dimension codes are anchor-supplied; the
Survivability anchor's set is the default enum below.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- Anchors & dimensions ----------------------------------------------------
class Anchor(str, Enum):
    survivability = "survivability"
    well_being = "well_being"
    flourishing = "flourishing"


class Dimension(str, Enum):
    """Survivability anchor's dimension set (default). Other anchors extend this."""
    NEURO = "NEURO"
    PSYCH = "PSYCH"
    SOCIO = "SOCIO"
    EVOBIO = "EVOBIO"


# The seven-persona pool. Discipline tag on a premise selects who testifies.
class Discipline(str, Enum):
    neuroscience = "neuroscience"
    psychology = "psychology"
    sociology = "sociology"
    evolutionary_biology = "evolutionary_biology"
    economics = "economics"
    health_science = "health_science"
    ecology = "ecology"


# Which disciplines can settle claims on which dimension (used for witness grants).
DISCIPLINE_TO_DIMENSION: dict[Discipline, Dimension] = {
    Discipline.neuroscience: Dimension.NEURO,
    Discipline.psychology: Dimension.PSYCH,
    Discipline.sociology: Dimension.SOCIO,
    Discipline.evolutionary_biology: Dimension.EVOBIO,
    Discipline.economics: Dimension.SOCIO,
    Discipline.health_science: Dimension.NEURO,
    Discipline.ecology: Dimension.EVOBIO,
}

Polarity = Literal[-1, 0, 1]  # -1 harms, 0 ambiguous, +1 helps the anchor


# --- Stakeholders ------------------------------------------------------------
class Stakeholder(BaseModel):
    id: str = Field(description="Stable id, e.g. S1")
    description: str
    type: Literal["individual", "group", "institution"]
    role: Literal["agent", "patient", "bystander"]
    vulnerability_flag: bool = False


class StakeholderRegister(BaseModel):
    case_id: str
    stakeholders: list[Stakeholder]


# --- Decisions ---------------------------------------------------------------
class ActionInventory(BaseModel):
    performed: list[str] = Field(default_factory=list)
    omitted: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    id: str = Field(description="d1, d2, ...")
    statement: str = Field(description="A debatable proposition")


class DecisionSet(BaseModel):
    case_id: str
    decisions: list[Decision]


# --- Premises (TPS) ----------------------------------------------------------
class Premise(BaseModel):
    id: str = Field(description="p1, p2, ... within a decision")
    text: str = Field(description="Falsifiable empirical proposition, not a moral claim")
    dimension: Dimension
    discipline: Discipline = Field(description="Whose literature can settle this")
    stakeholder_ids: list[str]
    polarity: Polarity


class TypedPremiseSet(BaseModel):
    case_id: str
    decision_id: str
    premises: list[Premise]


# --- Evidence ----------------------------------------------------------------
class EvidenceChunk(BaseModel):
    node_id: str
    text: str
    doi: Optional[str] = None
    source: str = Field(description="pubmed | openalex | semanticscholar | europepmc | crossref")
    venue: Optional[str] = None
    year: Optional[int] = None
    citation_count: int = 0
    dimension: Optional[Dimension] = None
    stakeholder_ids: list[str] = Field(default_factory=list)
    novelty_score: float = 1.0
    # The source's own record id (PMID, PMC id, OpenAlex W-id, S2 paperId): provenance
    # for records that carry no DOI.
    ext_id: Optional[str] = None


# --- Structured-output helpers (what LLM roles return) -----------------------
class StakeholderExtraction(BaseModel):
    stakeholders: list[Stakeholder]


class PremiseGeneration(BaseModel):
    premises: list[Premise]


class DecisionGeneration(BaseModel):
    """Raw decision statements from the LLM; ids assigned deterministically after."""
    decisions: list[str]


class PremiseDraft(BaseModel):
    """
    What the premise-generation LLM returns. The `dimension` is NOT asked of the
    model — it is derived deterministically from `discipline` (DISCIPLINE_TO_DIMENSION)
    so the two can never disagree. Ids are assigned after generation.
    """
    text: str = Field(description="Falsifiable empirical proposition, not a moral claim")
    discipline: Discipline = Field(description="Whose literature can settle this claim")
    stakeholder_ids: list[str] = Field(description="SR ids this premise bears on")
    polarity: Polarity = Field(description="+1 enhances, -1 degrades, 0 ambiguous, for the anchor")


class PremiseDraftSet(BaseModel):
    premises: list[PremiseDraft]


class PremiseRelation(BaseModel):
    src: str  # premise id
    dst: str  # premise id
    type: Literal["supports", "conflicts"]
    score: float


# --- Scientist testimony (P7) ------------------------------------------------
class ScientistSelection(BaseModel):
    """Which disciplines testify on a decision, and why (premise counts). Deterministic."""
    case_id: str
    decision_id: str
    selected: list[Discipline]
    premise_counts: dict[str, int]      # discipline -> #premises
    floor_applied: bool = False         # True if the floor rule added a discipline


EvidenceStrength = Literal["strong", "moderate", "weak", "insufficient"]


class TestimonyClaim(BaseModel):
    premise_id: str
    finding: str = Field(description="What the literature says about this premise")
    direction: Polarity = Field(description="Evidence-based effect on the anchor")
    evidence_strength: EvidenceStrength
    cited_nodes: list[str] = Field(default_factory=list, description="TEG node ids")
    # filled deterministically after generation (NLI gate), not by the LLM:
    nli_supported: bool = False


class ScientistTestimonyDraft(BaseModel):
    """What one scientist persona returns (before the NLI gate stamps support)."""
    claims: list[TestimonyClaim]


class ScientistTestimony(BaseModel):
    discipline: Discipline
    claims: list[TestimonyClaim]


class TestimonySet(BaseModel):
    case_id: str
    decision_id: str
    testimonies: list[ScientistTestimony]


# --- Debate (P8) -------------------------------------------------------------
class TurnClaim(BaseModel):
    stakeholder_id: str
    dimension: Dimension
    polarity: Polarity
    evidence_nodes: list[str] = Field(default_factory=list, description="TEG node ids cited")
    # The claim's own content. Without it the SIT updater had nothing claim-specific to
    # check against the cited evidence and scored the turn sentence nearest the node
    # instead, so every claim citing that node in a turn got the same verdict whatever
    # its stakeholder or polarity.
    statement: str = Field(default="", description="One sentence stating exactly what "
                           "this claim asserts about this stakeholder")
    # Which premise the claim (and so its cited evidence) bears on. Coverage used to pair
    # every node cited in a turn with every premise the turn listed; debaters list the
    # whole checklist, so two trolley-philosophy papers "covered" PTSD and prefrontal
    # stress premises. Coverage now counts a premise only through claims made about it.
    premise_id: str = Field(default="", description="The TPS premise id this claim bears "
                            "on, e.g. p3")


class DebateTurnDraft(BaseModel):
    """Structured output of one debater turn (before deterministic post-processing)."""
    premises_addressed: list[str] = Field(default_factory=list, description="TPS premise ids")
    evidence_cited: list[str] = Field(default_factory=list, description="TEG node ids")
    claims: list[TurnClaim] = Field(default_factory=list)
    evidential_gap: str = Field(default="", description="What evidence is still needed, "
                                "each need tagged with its premise id, e.g. '(p3)'")
    argument_text: str


class DebateTurn(BaseModel):
    round: int
    speaker: Literal["proponent", "opponent"]
    premises_addressed: list[str]
    evidence_cited: list[str]
    claims: list[TurnClaim]
    evidential_gap: str
    argument_text: str


class DebateResult(BaseModel):
    case_id: str
    decision_id: str
    rounds_run: int
    stop_reason: str
    coverage: float
    turns: list[DebateTurn]
    # Turns that still argued the other side after every retry ({"round", "side"}).
    role_reversals: list[dict] = Field(default_factory=list)
