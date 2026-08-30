"""Eval-regression tier (section 6): replay a saved run and assert the rendered
output is stable.

The fixture lives in `tests/fixtures/`, not `runs/` — Phase 0 gitignores `runs/`,
so a checked-in replay fixture cannot live there.
"""

import json
from pathlib import Path

from citewise.render import render_report
from citewise.state import EvidenceChunk, ReportDraft

SAVED_RUN = Path(__file__).parent.parent / "fixtures" / "saved_run.json"

EXPECTED_REPORT = """# Remote Work and Developer Productivity

## Measured Output

- A 2023 controlled study found remote developers committed 8% more code per week \
than their in-office peers. [1]
- Self-reported productivity rose for 62% of surveyed remote developers. [2]

## Sources

1. [A controlled study of remote software development output](https://example.org/controlled-study-2023)
2. [Developer Survey 2024: working arrangements](https://example.org/developer-survey-2024)
"""


def load_saved() -> dict:
    return json.loads(SAVED_RUN.read_text(encoding="utf-8"))


class TestReplay:
    def test_saved_run_rerenders_identically(self):
        saved = load_saved()
        draft = ReportDraft.model_validate(saved["draft"])
        evidence = [EvidenceChunk.model_validate(c) for c in saved["evidence"]]

        assert render_report(draft, evidence) == EXPECTED_REPORT

    def test_stored_report_matches_a_fresh_render(self):
        """The persisted final_report must not drift from what render.py produces."""
        saved = load_saved()
        draft = ReportDraft.model_validate(saved["draft"])
        evidence = [EvidenceChunk.model_validate(c) for c in saved["evidence"]]

        assert saved["final_report"] == render_report(draft, evidence)

    def test_saved_run_records_the_loop(self):
        saved = load_saved()
        assert saved["retry_count"] == 1
        assert [t["node"] for t in saved["trace"]] == [
            "planner",
            "searcher",
            "writer",
            "verifier",
            "writer",
            "verifier",
            "finalizer",
        ]

    def test_every_shipped_claim_cites_real_evidence(self):
        saved = load_saved()
        pool = {c["id"] for c in saved["evidence"]}
        for claim in saved["draft"]["claims"]:
            assert claim["evidence_ids"]
            assert set(claim["evidence_ids"]) <= pool
