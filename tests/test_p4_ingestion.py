import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.ingestion.anchor import get_anchor_config, menu, select_anchor
from src.ingestion.normalise import normalise
from src.schemas.core import Anchor, Dimension


def test_anchor_selection_forms():
    assert select_anchor(1).anchor == Anchor.survivability
    assert select_anchor("well-being").anchor == Anchor.well_being
    assert select_anchor("flourishing").anchor == Anchor.flourishing
    assert select_anchor(Anchor.survivability).label == "Survivability"
    with pytest.raises(ValueError):
        select_anchor("nonsense")


def test_anchor_config_has_dimensions_and_framing():
    cfg = get_anchor_config(Anchor.survivability)
    assert Dimension.NEURO in cfg.dimensions and len(cfg.dimensions) == 4
    assert cfg.definition and cfg.positive_direction
    assert "1." in menu() and "Flourishing" in menu()


def test_normalise_simple_case():
    n = normalise("  I usually   bring in my neighbor's mail,\r\n but I didn't today.  ")
    assert "  " not in n.full_text
    assert n.setting == "" and n.body


def test_normalise_setting_and_choices():
    text = (
        "Setting:\nYou are a doctor in an isolated village struck by disease. "
        "A shaman offers a ritual.\n"
        "Your Choices:\nA. Authorize the ritual.\nB. Refuse and use medicine."
    )
    n = normalise(text)
    assert "doctor" in n.setting.lower()
    assert "Authorize" in n.choices
    assert "Your Choices" in n.full_text


def test_normalise_quoted_dialogue():
    n = normalise('The druggist said: "No, I discovered the drug and I will profit."')
    assert any("discovered the drug" in q for q in n.quoted_spans)
