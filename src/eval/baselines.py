"""
P11 — Baselines. Each writes the same scorable outputs as the full pipeline (see
save_outputs): its final text as 09_arbiter/<sid>/central_script.md, the texts it was
compiled from as source_materials.json (what the faithfulness recompute grounds it
against), an audit bundle the eval reads, and the raw result under 12_baseline/.

  B1  Single LLM + CoT self-consistency    — the strongest monolithic reference
  B2  RAG-only single LLM (same TEG)        — separates retrieval from debate
  B3  Standard MAD (Du et al.)              — unstructured multi-agent debate
  B4  Homogeneous-model debate              — full framework, one model; this is the
                                              pipeline itself with every role forced onto
                                              one model (run_batch(force_model=...)), so
                                              it has no function here
  B5  Original CogniAlign (1 round, no RAG) — the parent

B1-B3 are told the same anchor as the main pipeline (its arbiter and B5 are anchored),
and every call opts into retry_truncated, so no rater compares a cut-off baseline
against a complete central script.

No LLM scores anything. B1's and B3's votes (part of those baselines' method, not of
the evaluation) group final answers with an LLM same-course judgement and pick the
medoid of the majority by embedding similarity.
"""
from __future__ import annotations

import json
import re

from config.settings import OUTPUTS_DIR
from src.services.llm_client import LLMClient, get_client, single_model_spec
from src.services.model_registry import ModelSpec, get_model

_FINAL_LINE = ("End with exactly one line of the form\n"
               "FINAL POSITION: <one sentence naming the course of action you judge best>")
_FINAL = re.compile(r"^\W*FINAL POSITION\W*:\s*(.+?)\s*$", re.I | re.M)
_SAME_PROMPT = """\
Two answers to the same moral case each ended with a final position.

A: {a}

B: {b}

Do A and B recommend the same course of action? They do when they make the same main
choice, even if one adds details, safeguards, conditions or follow-up steps the other
leaves out. They do not when they make different main choices, or when one permits
something the other rules out. Ignore wording, how strongly it is endorsed and the
reasons given. Reply with exactly one word: YES or NO."""


def _anchor(anchor=None):
    if anchor is None:
        from src.ingestion.anchor import select_anchor
        anchor = select_anchor("survivability")
    return anchor


def _anchor_block(anchor) -> str:
    return f"ANCHOR — {anchor.label}: {anchor.definition}"


def _client(base_role: str, temperature: float | None = None, name: str | None = None):
    """The base role's model, optionally at another temperature (a separate client)."""
    if temperature is None:
        return get_client(base_role)
    spec = get_model(base_role)
    return LLMClient(ModelSpec(role=name or base_role, provider=spec.provider,
                               model=spec.model, temperature=temperature,
                               reasoning=spec.reasoning))


def final_position(text: str) -> str:
    found = _FINAL.findall(text or "")
    return found[-1].strip(" *_") if found else ""


_VOTE_ROLE = "decision_generation"


def vote_judge():
    """
    The one judge for B1's and B3's votes: a non-reasoning model at temperature 0.

    qwen-plus split answers that made the same main choice whenever one added a step
    (B3 case 005: "refuse the nationwide rollout, delay, allow a limited human-reviewed
    version" twice, one adding "for the most serious violent crimes", fell into two
    groups). claude-sonnet-5 with the main-choice wording groups it; the probe pairs
    (scratchpad, 15 labelled) were 14/15 for both, with qwen's miss on that real case.
    """
    return _client(_VOTE_ROLE, 0.0, "baseline_vote")


def vote_label() -> str:
    """Recorded in the variant config, so a resume under another judge is refused."""
    return "llm-same-main-choice/" + get_model(_VOTE_ROLE).model


def same_answer(a: str, b: str, judge) -> bool:
    reply = judge.generate(user=_SAME_PROMPT.format(a=a, b=b),
                           system="You compare two answers. Reply YES or NO.",
                           max_tokens=10, retry_truncated=True)
    return (reply or "").strip().strip("*_ .").upper().startswith("YES")


def majority_vote(answers: list[str], texts: list[str], judge=None) -> dict:
    """
    Self-consistency vote over free-text final answers.

    The six cases list no fixed options, so answers are grouped by meaning: an answer
    joins the first group whose first answer the judge says recommends the same course
    of action (the LLM variant of semantic clustering, Farquhar et al. 2024). The
    largest group wins; a tie goes to the group formed first. An answer with no FINAL
    POSITION line never wins unless no answer has one. Of the winning group, the text
    most similar to the rest (MiniLM medoid) is returned. The old B1 counted identical
    80-character first lines, so when they all differed the "majority" was simply
    sample 1.

    Bidirectional NLI (Kuhn et al. 2023) was used before and cannot do this: on case
    001 five answers that all said "turn the trolley, save five at the cost of one" fell
    into three groups (one pair scored 0.41 and 0.001), and answers naming different
    options of the engineer case scored neutral both ways, like paraphrases of one
    option. MiniLM cosine did not separate them either (0.51 same option, 0.55 different).
    """
    judge = judge or vote_judge()
    clusters: list[list[int]] = []
    for i, a in enumerate(answers):
        home = None
        if a:
            for c in clusters:
                b = answers[c[0]]
                if b and same_answer(b, a, judge):
                    home = c
                    break
        if home is None:
            clusters.append([i])
        else:
            home.append(i)
    parsed = [c for c in clusters if answers[c[0]]]
    pool = max(parsed, key=len) if parsed else list(range(len(texts)))
    pick = pool[0]
    if len(pool) > 1:
        from src.services.embedder import embed
        v = embed([texts[i] for i in pool])
        pick = pool[int((v @ v.T).sum(axis=1).argmax())]
    return {"clusters": clusters, "majority": pool, "selected": pick,
            "vote_share": round(len(pool) / len(answers), 3) if parsed else 0.0}


# --- B1: single LLM + CoT self-consistency ---------------------------------
_B1_SYSTEM = """\
You are a careful moral analyst. Reason step by step about the case's alternative
courses of action and their consequences for everyone affected, judged against the
anchor below, then give a concise analysis. Do not fabricate evidence.

{anchor}

{final}"""


def run_b1_monolithic(case_text: str, anchor=None, k: int = 5,
                      temperature: float = 0.7) -> dict:
    """
    B1: one model reasons with chain-of-thought, sampled k times at a diverse
    temperature (the old run sampled at the role's 0.2, so the k answers were near
    copies); the majority final position is taken by majority_vote and its analysis
    returned. No source materials: faithfulness is not applicable.
    """
    anchor = _anchor(anchor)
    client = _client("action_decomposition", temperature, "b1_monolithic")
    system = _B1_SYSTEM.format(anchor=_anchor_block(anchor), final=_FINAL_LINE)
    analyses = [client.generate(user=f"Analyze this moral case step by step:\n\n{case_text}",
                                system=system, max_tokens=3000,
                                retry_truncated=True).strip()
                for _ in range(k)]
    answers = [final_position(a) for a in analyses]
    vote = majority_vote(answers, analyses)
    return {"baseline": "B1_monolithic_cot_selfconsistency",
            "model": getattr(getattr(client, "spec", None), "model", None),
            "temperature": temperature, "k": k, "analyses": analyses,
            "final_positions": answers, **vote,
            "script": analyses[vote["selected"]], "sources": []}


# --- B2: RAG-only single LLM ------------------------------------------------
def load_teg_evidence(teg_variant: str, sample_id: str) -> list[dict]:
    """
    The distinct TEG chunks the given pipeline variant gathered for this sample.

    B2 used to run its own live search with the case's first 24 words, which the
    sources match to nothing (prose queries return zero records), so it usually ran on
    "(no evidence retrieved)" and the sample was still ledgered. It now reads the
    finished variant's TEG, so B2 and that variant argue from the same evidence, and a
    missing TEG raises (the sample is not ledgered).
    """
    base = OUTPUTS_DIR / teg_variant
    ledger = base / "processed_samples.txt"
    done = set(ledger.read_text(encoding="utf-8").split()) if ledger.exists() else set()
    if sample_id not in done:
        raise FileNotFoundError(
            f"{teg_variant} has not finished sample {sample_id}: B2 reads that run's "
            f"evidence and retrieves none, so run it first")
    files = sorted((base / "06_evidence_teg" / sample_id).glob("TEG_*.json"))
    if not files:
        raise FileNotFoundError(f"no TEG_*.json under {base / '06_evidence_teg' / sample_id}")
    seen, out = set(), []
    for f in files:
        for n in json.loads(f.read_text(encoding="utf-8")).get("nodes", []):
            text = (n.get("text") or "").strip()
            key = (n.get("doi") or n.get("ext_id") or "", text[:200])
            if text and key not in seen:
                seen.add(key)
                out.append(n)
    if not out:
        raise ValueError(f"the TEG of {teg_variant}/{sample_id} holds no evidence text")
    return out


def run_b2_rag_only(case_text: str, sample_id: str, teg_variant: str, anchor=None) -> dict:
    """
    B2: hand the pipeline's own TEG evidence (no premises, no testimony, no debate) to
    ONE model to analyze. Isolates the debate's contribution — if B2 ≈ the full system,
    the adversarial deliberation adds nothing beyond the evidence.
    """
    anchor = _anchor(anchor)
    nodes = load_teg_evidence(teg_variant, sample_id)
    labels = {f"E{i}": n.get("node_id") for i, n in enumerate(nodes, 1)}
    ev_block = "\n\n".join(
        f"[E{i}] ({n.get('source')}, {n.get('year') or 'n.d.'}"
        f"{', DOI ' + n['doi'] if n.get('doi') else ''}) {n['text'].strip()}"
        for i, n in enumerate(nodes, 1))
    llm = get_client("scientist_pool")  # one single model, no debate stack
    analysis = llm.generate(
        user=f"CASE:\n{case_text}\n\n{_anchor_block(anchor)}\n\nEVIDENCE:\n{ev_block}\n\n"
             "Analyze the case's alternative courses of action and their consequences "
             "for everyone affected, judged against the anchor, using ONLY the evidence "
             "above. Cite the evidence label, e.g. [E3], after each claim it supports, "
             "and say where the evidence is silent. No debate.",
        system="You are a moral analyst. Ground every claim in the provided evidence.",
        max_tokens=4000, retry_truncated=True).strip()
    return {"baseline": "B2_rag_only", "teg_variant": teg_variant,
            "model": getattr(getattr(llm, "spec", None), "model", None),
            "n_evidence": len(nodes), "evidence_labels": labels, "analysis": analysis,
            "script": analysis, "sources": [n["text"] for n in nodes]}


# --- B3: standard multi-agent debate (Du et al.) ----------------------------
_B3_SYSTEM = """\
You are one participant in a multi-agent debate about a moral case. You have no fixed
role. Reason freely, judging the options against the anchor below.

{anchor}

{final}"""


def run_b3_standard_mad(case_text: str, anchor=None, n_agents: int = 3, rounds: int = 2,
                        temperature: float = 0.7) -> dict:
    """
    B3: Du et al. (2023) multi-agent debate. N instances of one model answer
    independently in round 1; in each later round every agent sees its own previous
    answer and the other agents' previous answers in full, and gives an updated answer.
    The final answer is the majority final position of the last round (majority_vote).

    The old loop updated answers within a round (round 1 was not independent), never
    showed an agent its own answer, cut the others' to 300 characters and aggregated
    nothing. Rounds default to 2, the paper's main setting.

    Sources for faithfulness: every turn except the one returned as the script (it
    would ground itself).
    """
    anchor = _anchor(anchor)
    llm = _client("scientist_pool", temperature, "b3_agent")
    system = _B3_SYSTEM.format(anchor=_anchor_block(anchor), final=_FINAL_LINE)
    agents = [f"Agent{i+1}" for i in range(n_agents)]
    transcript, last = [], {}
    for r in range(1, rounds + 1):
        prev = dict(last)   # everyone answers this round from the previous round only
        for a in agents:
            if r == 1:
                prompt = (f"CASE:\n{case_text}\n\nWhat is the right course of action? "
                          "Explain your reasoning step by step.")
            else:
                others = "\n\n".join(f"{k}:\n{v}" for k, v in prev.items() if k != a)
                prompt = (f"CASE:\n{case_text}\n\nYOUR PREVIOUS ANSWER:\n{prev[a]}\n\n"
                          f"THESE ARE THE ANSWERS FROM THE OTHER AGENTS:\n\n{others}\n\n"
                          "Using the reasoning from the other agents as additional advice, "
                          "give an updated answer. Examine your answer and the other "
                          "agents' answers step by step.")
            txt = llm.generate(user=prompt, system=system, max_tokens=1500,
                               retry_truncated=True).strip()
            last[a] = txt
            transcript.append({"round": r, "agent": a, "text": txt})
    finals = [last[a] for a in agents]
    vote = majority_vote([final_position(t) for t in finals], finals)
    chosen = agents[vote["selected"]]
    sources = [t["text"] for t in transcript if not (t["round"] == rounds and t["agent"] == chosen)]
    return {"baseline": "B3_standard_mad",
            "model": getattr(getattr(llm, "spec", None), "model", None),
            "temperature": temperature, "n_agents": n_agents, "rounds": rounds,
            "transcript": transcript, "final": last,
            "final_positions": [final_position(t) for t in finals], **vote,
            "selected_agent": chosen, "script": last[chosen], "sources": sources}


# --- B5: original CogniAlign (faithful replication) -------------------------
def run_b5_cognialign(case_text: str, model: str = "gpt-4o", temperature: float = 0.2) -> dict:
    """
    B5: a FAITHFUL replication of the published CogniAlign
    (Scientific-Moral-Debate repo). Uses that repo's VERBATIM prompts and its exact
    configuration — 4 discipline scientists each produce an argument, then each a
    rebuttal to the others, then a single arbiter concludes with a verdict — ALL on ONE
    model (gpt-4o, temp 0.2), no premises, no retrieval. The moral question is passed as
    a separate user turn (the repo's HumanMessage); the persona/node text is the system
    turn (the repo's SystemMessage). Its anchor is survivability, from those prompts.

    The model is reached the way the active roster is (single_model_spec): through
    OpenRouter as openai/gpt-4o under the OpenRouter roster.
    """
    from src.eval import cognialign_prompts as P

    llm = LLMClient(single_model_spec("b5_cognialign", model, temperature))
    panel = ["neuroscience", "psychology", "sociology", "evolutionary_biology"]

    # 1) arguments — persona prompt with ARGUMENT_NODE as node_prompt; case = user turn
    args = {}
    for disc in panel:
        system = P.PERSONA[disc].format(node_prompt=P.ARGUMENT_NODE)
        args[disc] = llm.generate(user=case_text, system=system, max_tokens=800,
                                  retry_truncated=True).strip()

    # 2) rebuttals — REBUTTAL_NODE with the OTHER agents' arguments as combined_arguments
    rebut = {}
    for disc in panel:
        combined = "\n\n".join(f"{k}: {v}" for k, v in args.items() if k != disc)
        node = P.REBUTTAL_NODE.format(combined_arguments=combined)
        system = P.PERSONA[disc].format(node_prompt=node)
        rebut[disc] = llm.generate(user=case_text, system=system, max_tokens=700,
                                   retry_truncated=True).strip()

    # 3) arbiter verdict — debate_history = all arguments + rebuttals; case = user turn
    debate_history = (
        "\n\n".join(f"[argument] {k}: {v}" for k, v in args.items())
        + "\n\n" + "\n\n".join(f"[rebuttal] {k}: {v}" for k, v in rebut.items())
    )
    verdict = llm.generate(user=case_text,
                           system=P.ARBITER.format(debate_history=debate_history),
                           max_tokens=1000, retry_truncated=True).strip()

    return {"baseline": "B5_cognialign_faithful", "model": model,
            "arguments": args, "rebuttals": rebut, "final_conclusion": verdict,
            "script": verdict, "sources": list(args.values()) + list(rebut.values())}


# --- scorable outputs --------------------------------------------------------
def save_outputs(store, case, anchor, name: str, result: dict) -> dict:
    """
    Write a baseline sample in the layout the eval and the faithfulness recompute read.

    Baselines used to write only 12_baseline/<sid>/bN.json, so build_eval reported 0
    samples and the recompute found no arbiter outputs: only B4 could be scored. The
    bundle has no decisions, SIT rows or register, so those metrics are NaN (not
    measurable); script_faithfulness is filled by scripts/recompute_faithfulness.py,
    against source_materials.json (B1 has none: not applicable).
    """
    from src.ingestion.normalise import normalise
    sid = case.sample_id or case.case_id
    norm = normalise(case.text)
    store.save("ingestion", sid, "normalised", {
        "setting": norm.setting, "body": norm.body, "choices": norm.choices,
        "full_text": norm.full_text, "quoted_spans": list(norm.quoted_spans),
        "stratum": case.stratum, "case_id": case.case_id}, echo=False)
    store.save("ingestion", sid, "anchor", {
        "anchor": anchor.anchor.value, "label": anchor.label,
        "dimensions": [d.value for d in anchor.dimensions]}, echo=False)
    store.save("baseline", sid, name, result, echo=False)
    store.save("arbiter", sid, "central_script", result["script"], fmt="md", echo=False)
    store.save("arbiter", sid, "source_materials",
               {"baseline": result["baseline"], "texts": result["sources"]}, echo=False)
    fb = {"fraction_entailed": None, "measured": False, "applicable": bool(result["sources"])}
    store.save("arbiter", sid, "faithfulness", fb, echo=False)
    store.save("arbiter", sid, "audit_bundle", {
        "case_id": case.case_id, "sample_id": sid, "baseline": result["baseline"],
        "anchor": anchor.label, "decisions": [], "sit_rows": [],
        "stakeholder_register": {"stakeholders": []}, "faithfulness": fb}, echo=False)
    summary = {"sample_id": sid, "case_id": case.case_id, "stratum": case.stratum,
               "anchor": anchor.label, "variant": store.variant, "baseline": name,
               "model": result.get("model"), "script_faithfulness": None}
    store.save("eval", sid, "run_summary", summary, echo=False)
    return summary
