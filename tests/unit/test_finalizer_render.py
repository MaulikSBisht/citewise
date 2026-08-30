"""Finalizer and renderer: nothing that failed verification ships silently."""

from citewise.config import Config
from citewise.nodes.finalizer import finalize_draft, make_finalizer_node
from citewise.render import UNVERIFIED_MARKER, render_report
from citewise.state import Claim, ReportDraft, Section, initial_state
from tests.unit.test_nodes_forward import evidence_pool


def claim(cid, text, evidence_ids=("e1",), verdict="SUPPORTED", reason=None):
    return Claim(
        id=cid,
        text=text,
        evidence_ids=list(evidence_ids),
        verdict=verdict,
        verdict_reason=reason,
    )


def draft(*claims, sections=None):
    sections = sections or [Section(heading="Findings", claim_ids=[c.id for c in claims])]
    return ReportDraft(title="A Report", sections=sections, claims=list(claims))


class TestFinalizeDraft:
    def test_contradicted_claims_are_dropped(self):
        final, dropped = finalize_draft(
            draft(claim("c1", "Kept."), claim("c2", "Bad.", verdict="CONTRADICTED"))
        )
        assert dropped == ["c2"]
        assert [c.id for c in final.claims] == ["c1"]

    def test_dropping_a_claim_prunes_it_from_its_section(self):
        """ReportDraft rejects dangling claim_ids, so the section must be pruned too."""
        final, _ = finalize_draft(
            draft(claim("c1", "Kept."), claim("c2", "Bad.", verdict="CONTRADICTED"))
        )
        assert final.sections[0].claim_ids == ["c1"]

    def test_section_emptied_by_drops_is_removed(self):
        final, _ = finalize_draft(draft(claim("c1", "Bad.", verdict="CONTRADICTED")))
        assert final.sections == []
        assert final.claims == []

    def test_unsupported_claims_are_kept(self):
        final, dropped = finalize_draft(
            draft(claim("c1", "Shaky.", verdict="UNSUPPORTED", reason="thin"))
        )
        assert dropped == []
        assert final.claims[0].verdict == "UNSUPPORTED"


class TestFinalizerNode:
    def test_sets_final_report_and_trace(self):
        node = make_finalizer_node(Config())
        state = initial_state("t") | {
            "draft": draft(claim("c1", "A supported claim.")),
            "evidence": evidence_pool(2),
        }
        out = node(state)
        assert out["final_report"].startswith("# A Report")
        assert out["trace"][0]["claims_shipped"] == 1

    def test_no_draft_yields_no_report(self):
        out = make_finalizer_node(Config())(initial_state("t"))
        assert out["final_report"] is None


class TestRender:
    def test_citations_are_numbered_by_first_appearance(self):
        md = render_report(
            draft(claim("c1", "First.", ("e2",)), claim("c2", "Second.", ("e1",))),
            evidence_pool(3),
        )
        assert "- First. [1]" in md
        assert "- Second. [2]" in md

    def test_sources_resolve_to_urls(self):
        md = render_report(draft(claim("c1", "First.", ("e1",))), evidence_pool(3))
        assert "1. [A controlled study of remote software development output]" in md
        assert "https://example.org/controlled-study-2023" in md

    def test_multiple_citations_on_one_claim(self):
        md = render_report(draft(claim("c1", "Both.", ("e1", "e2"))), evidence_pool(3))
        assert "- Both. [1][2]" in md

    def test_unsupported_claims_are_marked_inline(self):
        md = render_report(
            draft(claim("c1", "Shaky.", verdict="UNSUPPORTED", reason="evidence is thin")),
            evidence_pool(2),
        )
        assert f"- {UNVERIFIED_MARKER} Shaky." in md

    def test_unverified_footer_lists_reasons(self):
        md = render_report(
            draft(claim("c1", "Shaky.", verdict="UNSUPPORTED", reason="evidence is thin")),
            evidence_pool(2),
        )
        assert "## Unverified claims" in md
        assert "evidence is thin" in md

    def test_no_footer_when_everything_is_supported(self):
        md = render_report(draft(claim("c1", "Solid.")), evidence_pool(2))
        assert "## Unverified claims" not in md
        assert UNVERIFIED_MARKER not in md

    def test_headings_and_structure(self):
        md = render_report(draft(claim("c1", "Solid.")), evidence_pool(2))
        assert md.startswith("# A Report")
        assert "## Findings" in md
        assert "## Sources" in md
