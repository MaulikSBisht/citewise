"""ReportDraft -> markdown.

Rule 3: this is the only place markdown exists. Citations are numbered by first
appearance and resolve to source URLs; any claim that survived verification
without support is called out in a footer rather than presented as fact.
"""

from citewise.state import Claim, EvidenceChunk, ReportDraft

UNVERIFIED_MARKER = "[unverified]"


def _citation_numbering(draft: ReportDraft, evidence_ids: set[str]) -> dict[str, int]:
    """Number evidence in order of first citation, so [1] is the first source read."""
    numbering: dict[str, int] = {}
    for section in draft.sections:
        for claim_id in section.claim_ids:
            claim = next((c for c in draft.claims if c.id == claim_id), None)
            if claim is None:
                continue
            for eid in claim.evidence_ids:
                if eid in evidence_ids and eid not in numbering:
                    numbering[eid] = len(numbering) + 1
    # Claims not placed in any section still need their sources numbered.
    for claim in draft.claims:
        for eid in claim.evidence_ids:
            if eid in evidence_ids and eid not in numbering:
                numbering[eid] = len(numbering) + 1
    return numbering


def _render_claim(claim: Claim, numbering: dict[str, int]) -> str:
    marks = sorted(numbering[eid] for eid in claim.evidence_ids if eid in numbering)
    citation = " " + "".join(f"[{n}]" for n in marks) if marks else ""
    prefix = f"{UNVERIFIED_MARKER} " if claim.verdict == "UNSUPPORTED" else ""
    return f"- {prefix}{claim.text}{citation}"


def render_report(draft: ReportDraft, evidence: list[EvidenceChunk]) -> str:
    by_id = {c.id: c for c in evidence}
    numbering = _citation_numbering(draft, set(by_id))
    claims = draft.claim_map()

    lines: list[str] = [f"# {draft.title}", ""]

    for section in draft.sections:
        lines.append(f"## {section.heading}")
        lines.append("")
        for claim_id in section.claim_ids:
            claim = claims.get(claim_id)
            if claim is not None:
                lines.append(_render_claim(claim, numbering))
        lines.append("")

    if numbering:
        lines.append("## Sources")
        lines.append("")
        for eid, number in sorted(numbering.items(), key=lambda kv: kv[1]):
            chunk = by_id[eid]
            lines.append(f"{number}. [{chunk.title}]({chunk.url})")
        lines.append("")

    unverified = [c for c in draft.claims if c.verdict == "UNSUPPORTED"]
    if unverified:
        lines.append("## Unverified claims")
        lines.append("")
        lines.append(
            "Blind verification could not confirm the following claims against the "
            "evidence cited for them. They are marked "
            f"`{UNVERIFIED_MARKER}` above and should not be relied on."
        )
        lines.append("")
        for claim in unverified:
            reason = claim.verdict_reason or "no reason recorded"
            lines.append(f"- **{claim.text}** — {reason}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
