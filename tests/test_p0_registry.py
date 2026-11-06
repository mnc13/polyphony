import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.services.model_registry import ModelRegistry, family_of, get_registry


def test_registry_loads_and_validates():
    reg = get_registry()
    assert len(reg.roles()) == 9
    # interim mode -> distinct models, no repeats
    models = [reg.get(r).model for r in reg.roles()]
    assert len(models) == len(set(models)), "interim models must be distinct"


def test_premise_and_scientist_differ():
    reg = get_registry()
    assert reg.get("premise_generation").model != reg.get("scientist_pool").model


def test_family_mapping():
    assert family_of("gpt-4o") == "openai"
    assert family_of("anthropic/claude-3.7-sonnet") == "anthropic"
    assert family_of("deepseek/deepseek-chat") == "deepseek"
    assert family_of("qwen/qwen-2.5-72b-instruct") == "alibaba"


def test_repetition_rejected(tmp_path):
    yaml_text = """
active: interim_openai
roles:
  premise_generation:
    interim_openai: { provider: openai, model: gpt-4o, temperature: 0.3 }
    target_family:  { provider: openrouter, model: x-ai/grok-4, temperature: 0.3 }
  scientist_pool:
    interim_openai: { provider: openai, model: gpt-4o, temperature: 0.3 }
    target_family:  { provider: openrouter, model: moonshotai/kimi-k2, temperature: 0.3 }
"""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError):
        ModelRegistry(p)  # premise == scientist -> hard-distinct violation
