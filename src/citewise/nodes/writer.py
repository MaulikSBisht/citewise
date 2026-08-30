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


REVISION_SYSTEM = """You are revising specific claims in a research report that failed \
verification. A blind fact-checker read each claim next to the evidence cited for it \
and rejected it.

You will be given the evidence pool and the failing claims with the reason each was \
rejected. Produce a replacement for every failing claim, keeping its id.

Rules:
- Keep each claim's id exactly as given. Return one replacement per failing id, and \
nothing else.
- Fix the actual problem named in the reason. If the claim generalised past its \
evidence, narrow it to what the evidence says. If it cited the wrong chunk, cite the \
right one. If no evidence in the pool supports it, replace it with a different, \
narrower claim that the evidence does support.
- Every replacement must still be atomic — one assertion — and must cite at least one \
evidence ID from the pool.
- Do not restate a claim you were not asked to revise, and do not weaken a claim into \
vagueness to make it pass. "Some studies suggest a possible effect" is not a fix."""


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


class RevisionOutput(BaseModel):
    """Replacements for the failing claims only — the rest of the draft is frozen."""

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


def build_revision_prompt(
    evidence: list[EvidenceChunk], failing: list[Claim], frozen: list[Claim]
) -> str:
    ids = ", ".join(c.id for c in evidence)
    parts = [
        f"Evidence pool ({len(evidence)} chunks). Valid evidence IDs: {ids}",
        "",
        render_evidence_pool(evidence),
        "",
        "FAILING CLAIMS — replace each of these, keeping its id:",
    ]
    for claim in failing:
        parts.append(
            f"\nid: {claim.id}\n"
            f"current text: {claim.text}\n"
            f"currently cites: {', '.join(claim.evidence_ids)}\n"
            f"verdict: {claim.verdict}\n"
            f"reason it failed: {claim.verdict_reason or '(no reason given)'}"
        )

    if frozen:
        parts.append(
            "\nAlready-verified claims. These are frozen — do not revise them, do not "
            "return them, and do not duplicate what they already say:"
        )
        parts.extend(f"- [{c.id}] {c.text}" for c in frozen)

    parts.append(
        f"\nReturn exactly {len(failing)} replacement claim(s), with ids: "
        f"{', '.join(c.id for c in failing)}."
    )
    return "\n".join(parts)


def apply_revisions(draft: ReportDraft, replacements: list[DraftClaim]) -> ReportDraft:
    """Swap revised claims into the draft, leaving supported claims untouched.

    Claim order and section membership are preserved by rebuilding in place. A
    failing claim the writer declined to replace keeps its old text and verdict,
    so it stays visible to the finalizer rather than silently vanishing.
    """
    by_id = {r.id: r for r in replacements}
    claims: list[Claim] = []
    for claim in draft.claims:
        replacement = by_id.get(claim.id)
        if replacement is None or claim.verdict == "SUPPORTED":
            claims.append(claim)
            continue
        claims.append(
            Claim(
                id=claim.id,
                text=replacement.text,
                evidence_ids=replacement.evidence_ids,
                verdict="PENDING",
                verdict_reason=None,
            )
        )
    return draft.model_copy(update={"claims": claims})


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
    """Writes a fresh draft, or revises only the failing claims on re-entry.

    Re-entry is detected from the state itself: a draft carrying verdicts means
    the verifier has already run. No extra feedback channel is needed — the
    failing claims carry their own reasons.
    """

    def write(state: ResearchState) -> dict:
        evidence = state["evidence"]
        existing: ReportDraft | None = state.get("draft")
        revising = existing is not None and any(c.verdict != "PENDING" for c in existing.claims)

        if revising:
            failing = [c for c in existing.claims if c.verdict != "SUPPORTED"]
            frozen = [c for c in existing.claims if c.verdict == "SUPPORTED"]
            revision = llm.complete_structured(
                system=REVISION_SYSTEM,
                user=build_revision_prompt(evidence, failing, frozen),
                schema=RevisionOutput,
                model=config.writer_model,
            )
            candidate = apply_revisions(existing, revision.claims)
            trace: dict = {
                "node": "writer",
                "mode": "revision",
                "round": state["retry_count"] + 1,
                "revised": [c.id for c in revision.claims],
                "frozen": [c.id for c in frozen],
            }
        else:
            output = llm.complete_structured(
                system=WRITER_SYSTEM,
                user=build_writer_prompt(state["topic"], evidence),
                schema=WriterOutput,
                model=config.writer_model,
            )
            candidate = None
            trace = {"node": "writer", "mode": "draft"}

        # A draft citing an ID that is not in the pool is unverifiable by
        # construction, so it is rejected rather than carried forward.
        try:
            draft = candidate if revising else to_report_draft(output)
            validate_draft_against_evidence(draft, evidence)
        except (UnknownEvidenceIDError, ValidationError, ValueError) as exc:
            return {
                "draft": None,
                "aborted_reason": f"writer produced an invalid draft: {exc}",
                "trace": [{**trace, "rejected": str(exc)}],
            }

        update: dict = {
            "draft": draft,
            "trace": [
                {
                    **trace,
                    "title": draft.title,
                    "claims": len(draft.claims),
                    "sections": len(draft.sections),
                }
            ],
        }
        if revising:
            # The counter advances here, at the one place the loop re-enters, so
            # every round through the writer is counted exactly once.
            update["retry_count"] = state["retry_count"] + 1
        return update

    return write
