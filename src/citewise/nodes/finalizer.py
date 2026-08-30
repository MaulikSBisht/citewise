"""Decide what survives.

Claims still CONTRADICTED are dropped. Claims still UNSUPPORTED are kept but
marked explicitly as unverified in the output. Nothing that failed verification
is ever shipped silently.
"""

from citewise.config import Config
from citewise.render import render_report
from citewise.state import ReportDraft, ResearchState, Section


def finalize_draft(draft: ReportDraft) -> tuple[ReportDraft, list[str]]:
    """Drop contradicted claims and prune the sections that referenced them.

    Pruning matters: `ReportDraft` rejects a section citing a claim that no
    longer exists, so dropping a claim without fixing its section would make the
    surviving draft unconstructable.
    """
    dropped = [c.id for c in draft.claims if c.verdict == "CONTRADICTED"]
    kept = [c for c in draft.claims if c.verdict != "CONTRADICTED"]
    kept_ids = {c.id for c in kept}

    sections = [
        Section(heading=s.heading, claim_ids=[cid for cid in s.claim_ids if cid in kept_ids])
        for s in draft.sections
    ]
    # A section whose every claim was contradicted has nothing left to say.
    sections = [s for s in sections if s.claim_ids]

    return ReportDraft(title=draft.title, sections=sections, claims=kept), dropped


def make_finalizer_node(config: Config):
    def finalize(state: ResearchState) -> dict:
        draft: ReportDraft | None = state["draft"]
        if draft is None:
            return {
                "final_report": None,
                "trace": [{"node": "finalizer", "skipped": "no draft"}],
            }

        final_draft, dropped = finalize_draft(draft)
        unverified = [c.id for c in final_draft.claims if c.verdict == "UNSUPPORTED"]

        return {
            "draft": final_draft,
            "final_report": render_report(final_draft, state["evidence"]),
            "trace": [
                {
                    "node": "finalizer",
                    "dropped_contradicted": dropped,
                    "kept_unverified": unverified,
                    "claims_shipped": len(final_draft.claims),
                }
            ],
        }

    return finalize
