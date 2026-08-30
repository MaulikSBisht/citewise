"""The arm citewise is measured against.

Same evidence, one Sonnet call, "write a cited report", no verification loop.

The baseline's output schema deliberately allows an empty `evidence_ids` list
and does not check IDs against the pool. citewise rejects both at the schema
level, and if the baseline enforced them too the comparison would measure
nothing — uncited and mis-cited claims are exactly the failure mode a no-loop
system exhibits, so the baseline has to be able to produce them.
"""

import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citewise.config import Config  # noqa: E402
from citewise.llm import StructuredLLM  # noqa: E402
from citewise.nodes.writer import render_evidence_pool  # noqa: E402
from citewise.state import Claim, EvidenceChunk, ReportDraft, Section  # noqa: E402

BASELINE_SYSTEM = """You are a research writer. Given a topic and a pool of evidence, \
write a cited report.

Break the report into claims. Cite the evidence you used for each claim by its ID."""


class BaselineClaim(BaseModel):
    text: str
    evidence_ids: list[str]


class BaselineOutput(BaseModel):
    title: str
    claims: list[BaselineClaim]


def build_baseline_prompt(topic: str, evidence: list[EvidenceChunk]) -> str:
    ids = ", ".join(c.id for c in evidence)
    return (
        f"Topic: {topic}\n\n"
        f"Evidence pool ({len(evidence)} chunks). Evidence IDs: {ids}\n\n"
        f"{render_evidence_pool(evidence)}\n\n"
        "Write a cited report on this topic."
    )


def to_draft(output: BaselineOutput) -> ReportDraft:
    """Lift the baseline's flat claim list into a ReportDraft for shared scoring.

    Claims with no citations are kept — dropping them here would quietly launder
    the baseline's citation-coverage number.
    """
    claims = [
        Claim(
            id=f"c{i + 1}",
            text=claim.text,
            # Claim requires a non-empty list, so an uncited claim is recorded
            # with a sentinel that resolves to nothing in the pool. Citation
            # coverage counts it as uncited, which is the honest reading.
            evidence_ids=claim.evidence_ids or ["__uncited__"],
            verdict="PENDING",
        )
        for i, claim in enumerate(output.claims)
        if claim.text.strip()
    ]
    return ReportDraft(
        title=output.title,
        sections=[Section(heading="Report", claim_ids=[c.id for c in claims])],
        claims=claims,
    )


def run_baseline(
    topic: str,
    evidence: list[EvidenceChunk],
    llm: StructuredLLM,
    config: Config,
) -> ReportDraft:
    """One writer call over the given evidence. No loop, no revision."""
    output = llm.complete_structured(
        system=BASELINE_SYSTEM,
        user=build_baseline_prompt(topic, evidence),
        schema=BaselineOutput,
        model=config.writer_model,
    )
    return to_draft(output)
