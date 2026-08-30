"""Schema tests. Rule 5: a claim with empty evidence_ids is invalid at the
schema level, not tolerated downstream."""

import pytest
from pydantic import ValidationError

from citewise.state import (
    Claim,
    EvidenceChunk,
    ReportDraft,
    Section,
    UnknownEvidenceIDError,
    validate_draft_against_evidence,
)


def _chunk(cid: str) -> EvidenceChunk:
    return EvidenceChunk(
        id=cid,
        url=f"https://example.com/{cid}",
        title=f"Title {cid}",
        snippet=f"Snippet text for {cid}.",
        sub_question="What is the thing?",
    )


def _claim(cid: str, evidence_ids: list[str]) -> Claim:
    return Claim(id=cid, text=f"Assertion {cid}.", evidence_ids=evidence_ids)


class TestClaim:
    def test_claim_defaults_to_pending(self):
        c = _claim("c1", ["e1"])
        assert c.verdict == "PENDING"
        assert c.verdict_reason is None

    def test_empty_evidence_ids_rejected(self):
        with pytest.raises(ValidationError):
            Claim(id="c1", text="Unsourced assertion.", evidence_ids=[])

    def test_missing_evidence_ids_rejected(self):
        with pytest.raises(ValidationError):
            Claim(id="c1", text="Unsourced assertion.")

    def test_blank_text_rejected(self):
        with pytest.raises(ValidationError):
            Claim(id="c1", text="   ", evidence_ids=["e1"])

    def test_duplicate_evidence_ids_collapsed(self):
        c = Claim(id="c1", text="Assertion.", evidence_ids=["e1", "e1", "e2"])
        assert c.evidence_ids == ["e1", "e2"]

    @pytest.mark.parametrize("verdict", ["PENDING", "SUPPORTED", "UNSUPPORTED", "CONTRADICTED"])
    def test_valid_verdicts(self, verdict):
        assert Claim(id="c1", text="A.", evidence_ids=["e1"], verdict=verdict).verdict == verdict

    def test_invalid_verdict_rejected(self):
        with pytest.raises(ValidationError):
            Claim(id="c1", text="A.", evidence_ids=["e1"], verdict="MAYBE")


class TestReportDraft:
    def test_valid_draft(self):
        draft = ReportDraft(
            title="A Report",
            sections=[Section(heading="Background", claim_ids=["c1", "c2"])],
            claims=[_claim("c1", ["e1"]), _claim("c2", ["e2"])],
        )
        assert len(draft.claims) == 2

    def test_section_citing_unknown_claim_rejected(self):
        with pytest.raises(ValidationError, match="unknown claim"):
            ReportDraft(
                title="A Report",
                sections=[Section(heading="Background", claim_ids=["c1", "c99"])],
                claims=[_claim("c1", ["e1"])],
            )

    def test_duplicate_claim_ids_rejected(self):
        with pytest.raises(ValidationError, match="duplicate claim id"):
            ReportDraft(
                title="A Report",
                sections=[Section(heading="B", claim_ids=["c1"])],
                claims=[_claim("c1", ["e1"]), _claim("c1", ["e2"])],
            )

    def test_claim_map_lookup(self):
        draft = ReportDraft(
            title="A Report",
            sections=[Section(heading="B", claim_ids=["c1"])],
            claims=[_claim("c1", ["e1"])],
        )
        assert draft.claim_map()["c1"].text == "Assertion c1."


class TestValidateDraftAgainstEvidence:
    def test_accepts_draft_citing_known_evidence(self):
        draft = ReportDraft(
            title="R",
            sections=[Section(heading="B", claim_ids=["c1"])],
            claims=[_claim("c1", ["e1", "e2"])],
        )
        validate_draft_against_evidence(draft, [_chunk("e1"), _chunk("e2")])

    def test_rejects_draft_citing_unknown_evidence(self):
        """Phase 2 done-condition: a draft citing a nonexistent evidence ID is rejected."""
        draft = ReportDraft(
            title="R",
            sections=[Section(heading="B", claim_ids=["c1"])],
            claims=[_claim("c1", ["e1", "e404"])],
        )
        with pytest.raises(UnknownEvidenceIDError) as exc:
            validate_draft_against_evidence(draft, [_chunk("e1")])
        assert "e404" in str(exc.value)
        assert "c1" in str(exc.value)
