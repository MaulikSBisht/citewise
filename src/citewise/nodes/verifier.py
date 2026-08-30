"""Blind claim verification.

Rule 4 is the whole point of the project: the verifier receives a claim and
evidence chunks and nothing else. It never sees the draft, the writer's
reasoning, the other claims, or the framing of the topic. That isolation is what
makes a SUPPORTED verdict mean something, so `build_verifier_prompt` takes only
a claim and the evidence pool — it is not given the state, and cannot leak from it.

Each claim gets its own API call with a fresh message list, so no claim's verdict
can influence another's.
"""

from typing import Literal

from pydantic import BaseModel

from citewise.config import Config
from citewise.llm import StructuredLLM
from citewise.state import Claim, EvidenceChunk, ReportDraft, ResearchState

VERIFIER_SYSTEM = """You are a fact-checker. You are shown ONE claim and the evidence \
cited for it. Judge only whether that evidence establishes that claim.

Verdicts:
- SUPPORTED: the cited evidence states the claim, or straightforwardly entails it. \
Figures, dates, and quantities must match.
- UNSUPPORTED: the cited evidence does not establish the claim. This includes a \
claim that generalises past its evidence (one study or one company presented as a \
general finding), a claim whose numbers do not match the evidence, and a claim the \
evidence simply does not address.
- CONTRADICTED: some evidence asserts something incompatible with the claim — an \
opposite direction of effect, or an incompatible figure.

Judge the claim exactly as written. Do not give it the benefit of the doubt, do not \
imagine a charitable reading, and do not use knowledge of your own: if the cited \
evidence does not establish it, it is not supported. A claim that is true in the \
world but unsupported by its evidence is UNSUPPORTED.

You are deliberately shown this claim on its own. Do not speculate about the wider \
report it came from.

Give a verdict and a one-sentence reason."""


class VerifierOutput(BaseModel):
    verdict: Literal["SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    reason: str


def build_verifier_prompt(claim: Claim, evidence: list[EvidenceChunk]) -> str:
    """Claim text plus its cited evidence, then the rest of the pool.

    The rest of the pool is included so the verifier can spot a contradiction it
    would otherwise miss, but it is labelled clearly so uncited evidence is not
    mistaken for support.
    """
    by_id = {c.id: c for c in evidence}
    cited = [by_id[eid] for eid in claim.evidence_ids if eid in by_id]
    others = [c for c in evidence if c.id not in set(claim.evidence_ids)]

    def block(chunk: EvidenceChunk) -> str:
        return f"[{chunk.id}] {chunk.title}\n{chunk.snippet}"

    parts = [f"CLAIM:\n{claim.text}", ""]

    if cited:
        parts.append("EVIDENCE CITED FOR THIS CLAIM:")
        parts.append("\n\n".join(block(c) for c in cited))
    else:
        parts.append("EVIDENCE CITED FOR THIS CLAIM:\n(none of the cited IDs exist)")

    if others:
        parts.append("")
        parts.append(
            "OTHER AVAILABLE EVIDENCE (not cited for this claim; use it only to "
            "detect contradiction, never as support):"
        )
        parts.append("\n\n".join(block(c) for c in others))

    return "\n".join(parts)


def make_verifier_node(llm: StructuredLLM, config: Config):
    def verify(state: ResearchState) -> dict:
        draft: ReportDraft | None = state["draft"]
        if draft is None:
            return {"trace": [{"node": "verifier", "skipped": "no draft"}]}

        evidence = state["evidence"]
        verified: list[Claim] = []
        results: list[dict] = []

        for claim in draft.claims:
            # Frozen claims keep the verdict they already earned. Their text is
            # byte-identical, so a second call could only cost money or flap.
            if claim.verdict == "SUPPORTED":
                verified.append(claim)
                continue

            outcome = llm.complete_structured(
                system=VERIFIER_SYSTEM,
                user=build_verifier_prompt(claim, evidence),
                schema=VerifierOutput,
                model=config.verifier_model,
            )
            verified.append(
                claim.model_copy(
                    update={"verdict": outcome.verdict, "verdict_reason": outcome.reason}
                )
            )
            results.append(
                {"claim_id": claim.id, "verdict": outcome.verdict, "reason": outcome.reason}
            )

        return {
            "draft": draft.model_copy(update={"claims": verified}),
            "trace": [
                {
                    "node": "verifier",
                    "round": state["retry_count"],
                    "verified": len(results),
                    "results": results,
                }
            ],
        }

    return verify


def failing_claims(draft: ReportDraft) -> list[Claim]:
    return [c for c in draft.claims if c.verdict != "SUPPORTED"]
