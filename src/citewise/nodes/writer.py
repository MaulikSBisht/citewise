"""Evidence -> ReportDraft. Sonnet, structured output.

Rule 3: the writer never emits prose. It emits a `ReportDraft`; markdown exists
only in `render.py`.

Compound claims are the main thing that makes verification mushy — "X rose and
Y fell" can be half-true, and a single verdict cannot express that — so the
prompt pushes hard on atomicity.
"""

from pydantic import BaseModel, Field, ValidationError

from citewise.config import Config
from citewise.llm import StructuredLLM
from citewise.state import (
    Claim,
    EvidenceChunk,
    ReportDraft,
    ResearchState,
    Section,
    UnknownEvidenceIDError,
    validate_draft_against_evidence,
)

WRITER_SYSTEM = """You write research reports as structured claims bound to evidence.
You never write prose paragraphs. You emit claims.

Hard requirements:
- Every claim MUST cite at least one evidence ID, drawn only from the pool you \
are given. Never invent an ID. Never cite an ID that is not in the pool.
- Every claim MUST be atomic: exactly one assertion. If a sentence contains \
"and", "but", "while", or a semicolon joining two facts, split it into separate \
claims. "Commits rose 8% and review latency doubled" is two claims, not one.
- Every claim MUST be supported by the text of the evidence you cite for it. Do \
not generalise beyond what the snippet says. If a snippet describes one company, \
do not write a claim about an industry.
- Prefer specific figures, dates, and named sources over vague summary.
- If the evidence disagrees with itself, write both claims and cite each side. \
Do not average them into a false consensus.
- Group claims into 2-4 sections. Every claim id must appear in exactly one section."""


class DraftClaim(BaseModel):
    """What the writer is asked to produce — no verdict field.

    Verdicts are the verifier's to assign; letting the writer emit them would
    invite it to mark its own work.
    """

    id: str
    text: str
    evidence_ids: list[str] = Field(min_length=1)


class DraftSection(BaseModel):
    heading: str
    claim_ids: list[str]


class WriterOutput(BaseModel):
    title: str
    sections: list[DraftSection]
    claims: list[DraftClaim]


def render_evidence_pool(evidence: list[EvidenceChunk]) -> str:
    return "\n\n".join(
        f"[{c.id}] {c.title}\nURL: {c.url}\nSub-question: {c.sub_question}\n{c.snippet}"
        for c in evidence
    )


def build_writer_prompt(topic: str, evidence: list[EvidenceChunk]) -> str:
    ids = ", ".join(c.id for c in evidence)
    return (
        f"Topic: {topic}\n\n"
        f"Evidence pool ({len(evidence)} chunks). Valid evidence IDs: {ids}\n\n"
        f"{render_evidence_pool(evidence)}\n\n"
        "Write the report as structured claims. Cite only the IDs listed above."
    )


def to_report_draft(output: WriterOutput) -> ReportDraft:
    """Lift the writer's narrow output into the full draft schema."""
    return ReportDraft(
        title=output.title,
        sections=[Section(heading=s.heading, claim_ids=s.claim_ids) for s in output.sections],
        claims=[
            Claim(id=c.id, text=c.text, evidence_ids=c.evidence_ids, verdict="PENDING")
            for c in output.claims
        ],
    )


def make_writer_node(llm: StructuredLLM, config: Config):
    def write(state: ResearchState) -> dict:
        evidence = state["evidence"]

        output = llm.complete_structured(
            system=WRITER_SYSTEM,
            user=build_writer_prompt(state["topic"], evidence),
            schema=WriterOutput,
            model=config.writer_model,
        )

        # A draft citing an ID that is not in the pool is unverifiable by
        # construction, so it is rejected rather than carried forward.
        try:
            draft = to_report_draft(output)
            validate_draft_against_evidence(draft, evidence)
        except (UnknownEvidenceIDError, ValidationError, ValueError) as exc:
            return {
                "draft": None,
                "aborted_reason": f"writer produced an invalid draft: {exc}",
                "trace": [{"node": "writer", "rejected": str(exc)}],
            }

        return {
            "draft": draft,
            "trace": [
                {
                    "node": "writer",
                    "title": draft.title,
                    "claims": len(draft.claims),
                    "sections": len(draft.sections),
                }
            ],
        }

    return write
