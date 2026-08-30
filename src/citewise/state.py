"""Data model for citewise.

Every claim is bound to evidence. Rule 5: a claim with an empty `evidence_ids`
list is invalid and is rejected here, at the schema level, rather than being
tolerated and papered over downstream.
"""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

Verdict = Literal["PENDING", "SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]

NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class UnknownEvidenceIDError(ValueError):
    """A draft cited an evidence ID that is not in the retrieved pool."""


class EvidenceChunk(BaseModel):
    id: str
    url: str
    title: str
    snippet: str
    sub_question: str


class Claim(BaseModel):
    id: str
    text: NonBlankStr  # atomic — one assertion
    evidence_ids: list[str] = Field(min_length=1)
    verdict: Verdict = "PENDING"
    verdict_reason: str | None = None

    @field_validator("evidence_ids")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for eid in v:
            seen.setdefault(eid, None)
        return list(seen)


class Section(BaseModel):
    heading: str
    claim_ids: list[str]


class ReportDraft(BaseModel):
    title: str
    sections: list[Section]
    claims: list[Claim]

    @model_validator(mode="after")
    def _sections_resolve_to_real_claims(self) -> "ReportDraft":
        known: set[str] = set()
        for claim in self.claims:
            if claim.id in known:
                raise ValueError(f"duplicate claim id {claim.id!r}")
            known.add(claim.id)

        for section in self.sections:
            for cid in section.claim_ids:
                if cid not in known:
                    raise ValueError(f"section {section.heading!r} cites unknown claim {cid!r}")
        return self

    def claim_map(self) -> dict[str, Claim]:
        return {c.id: c for c in self.claims}


def validate_draft_against_evidence(draft: ReportDraft, evidence: list[EvidenceChunk]) -> None:
    """Boundary check: every cited evidence ID must exist in the retrieved pool.

    Kept out of `ReportDraft` because the pool lives in the graph state, not on
    the draft. Raises `UnknownEvidenceIDError` naming the offending claim and ID.
    """
    pool = {chunk.id for chunk in evidence}
    for claim in draft.claims:
        unknown = [eid for eid in claim.evidence_ids if eid not in pool]
        if unknown:
            raise UnknownEvidenceIDError(
                f"claim {claim.id!r} cites evidence ID(s) "
                f"{', '.join(repr(u) for u in unknown)} not in the evidence pool "
                f"({len(pool)} chunks: {', '.join(sorted(pool))})"
            )


class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    evidence: list[EvidenceChunk]
    draft: ReportDraft | None
    retry_count: int
    final_report: str | None
    trace: list[dict]
    aborted_reason: str | None
