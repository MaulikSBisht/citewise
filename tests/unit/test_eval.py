"""Eval tooling tests. No API calls — the metric and scoring functions are pure,
and the labeling scorer is exercised against a small synthetic filled sheet."""

import csv
import sys
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from baseline import BaselineClaim, BaselineOutput, to_draft  # noqa: E402
from citewise.state import Claim, EvidenceChunk, ReportDraft, Section  # noqa: E402
from run_eval import (  # noqa: E402
    LABELING_COLUMNS,
    aggregate,
    build_labeling_rows,
    cost_usd,
    load_topics,
    markdown_table,
    retry_distribution,
    score_draft,
    write_labeling_sheet,
)
from score_labels import (  # noqa: E402
    LabelingSheetError,
    format_report,
    read_labeled_rows,
    score,
)


def chunk(cid):
    return EvidenceChunk(
        id=cid, url=f"https://e/{cid}", title=cid, snippet=f"text {cid}", sub_question="q"
    )


def draft(*specs):
    claims = [
        Claim(id=f"c{i + 1}", text=f"claim {i + 1}", evidence_ids=list(eids), verdict=verdict)
        for i, (verdict, eids) in enumerate(specs)
    ]
    return ReportDraft(
        title="R",
        sections=[Section(heading="S", claim_ids=[c.id for c in claims])],
        claims=claims,
    )


class TestTopics:
    def test_ten_topics_with_a_contested_half(self):
        topics = load_topics()
        assert len(topics) == 10
        kinds = [t["kind"] for t in topics]
        assert kinds.count("clean") == 5
        assert kinds.count("contested") == 5

    def test_three_topics_are_marked_for_labeling(self):
        assert sum(1 for t in load_topics() if t.get("labeling_sample")) == 3

    def test_ids_are_unique(self):
        ids = [t["id"] for t in load_topics()]
        assert len(set(ids)) == len(ids)


class TestCost:
    def test_prices_input_and_output_separately(self):
        usage = [{"model": "claude-sonnet-4-6", "input_tokens": 1_000_000, "output_tokens": 0}]
        assert cost_usd(usage) == 3.0

        usage = [{"model": "claude-sonnet-4-6", "input_tokens": 0, "output_tokens": 1_000_000}]
        assert cost_usd(usage) == 15.0

    def test_sums_across_models(self):
        usage = [
            {"model": "claude-haiku-4-5", "input_tokens": 1_000_000, "output_tokens": 0},
            {"model": "claude-sonnet-4-6", "input_tokens": 1_000_000, "output_tokens": 0},
        ]
        assert cost_usd(usage) == 4.0

    def test_unknown_model_prices_at_zero_rather_than_crashing(self):
        assert cost_usd([{"model": "nope", "input_tokens": 999, "output_tokens": 999}]) == 0.0

    def test_empty_usage_is_free(self):
        assert cost_usd([]) == 0.0


class TestScoreDraft:
    def test_counts_verdicts_and_rates(self):
        scored = score_draft(
            draft(("SUPPORTED", ["e1"]), ("UNSUPPORTED", ["e1"]), ("CONTRADICTED", ["e1"])),
            [chunk("e1")],
        )
        assert scored["claims"] == 3
        assert scored["unsupported_claims"] == 1
        assert scored["contradicted_claims"] == 1
        assert scored["unsupported_claim_rate"] == pytest.approx(0.3333, abs=1e-4)

    def test_citation_coverage_counts_only_ids_in_the_pool(self):
        """An invented ID is not a citation."""
        scored = score_draft(draft(("SUPPORTED", ["e1"]), ("SUPPORTED", ["e404"])), [chunk("e1")])
        assert scored["citation_coverage_pct"] == 50.0

    def test_full_coverage(self):
        scored = score_draft(draft(("SUPPORTED", ["e1"])), [chunk("e1")])
        assert scored["citation_coverage_pct"] == 100.0

    def test_none_draft_scores_empty_rather_than_dividing_by_zero(self):
        scored = score_draft(None, [])
        assert scored["claims"] == 0
        assert scored["unsupported_claim_rate"] is None


class TestBaselineDraft:
    def test_uncited_claims_are_kept_and_counted_as_uncited(self):
        """Dropping them would launder the baseline's citation-coverage number."""
        built = to_draft(
            BaselineOutput(
                title="R",
                claims=[
                    BaselineClaim(text="cited", evidence_ids=["e1"]),
                    BaselineClaim(text="uncited", evidence_ids=[]),
                ],
            )
        )
        assert len(built.claims) == 2
        assert score_draft(built, [chunk("e1")])["citation_coverage_pct"] == 50.0

    def test_blank_claims_are_dropped(self):
        built = to_draft(
            BaselineOutput(
                title="R",
                claims=[
                    BaselineClaim(text="real", evidence_ids=["e1"]),
                    BaselineClaim(text="   ", evidence_ids=["e1"]),
                ],
            )
        )
        assert len(built.claims) == 1


class TestAggregate:
    ROWS = [
        {
            "arm": "citewise",
            "error": None,
            "claims": 4,
            "unsupported_claim_rate": 0.0,
            "contradicted_claims": 0,
            "citation_coverage_pct": 100.0,
            "latency_s": 10.0,
            "cost_usd": 0.5,
            "retry_count": 1,
        },
        {
            "arm": "citewise",
            "error": None,
            "claims": 2,
            "unsupported_claim_rate": 0.5,
            "contradicted_claims": 1,
            "citation_coverage_pct": 100.0,
            "latency_s": 20.0,
            "cost_usd": 0.5,
            "retry_count": 2,
        },
        {
            "arm": "baseline",
            "error": None,
            "claims": 5,
            "unsupported_claim_rate": 0.4,
            "contradicted_claims": 2,
            "citation_coverage_pct": 60.0,
            "latency_s": 5.0,
            "cost_usd": 0.1,
            "retry_count": 0,
        },
    ]

    def test_aggregates_per_arm(self):
        result = aggregate(self.ROWS, "citewise")
        assert result["runs"] == 2
        assert result["total_claims"] == 6
        assert result["unsupported_claim_rate"] == 0.25
        assert result["total_cost_usd"] == 1.0

    def test_errored_runs_are_excluded(self):
        rows = [*self.ROWS, {"arm": "citewise", "error": "boom"}]
        assert aggregate(rows, "citewise")["runs"] == 2

    def test_retry_distribution_only_for_citewise(self):
        assert aggregate(self.ROWS, "citewise")["retry_distribution"] == {"1": 1, "2": 1}
        assert aggregate(self.ROWS, "baseline")["retry_distribution"] is None

    def test_no_runs_is_handled(self):
        assert aggregate([], "citewise") == {"arm": "citewise", "runs": 0}

    def test_retry_distribution_counts_rounds(self):
        assert retry_distribution([{"retry_count": 0}, {"retry_count": 0}, {"retry_count": 2}]) == {
            "0": 2,
            "2": 1,
        }


class TestMarkdownTable:
    def test_renders_both_arms(self):
        table = markdown_table([aggregate(TestAggregate.ROWS, a) for a in ("citewise", "baseline")])
        assert "| citewise |" in table
        assert "| baseline |" in table
        assert "Retry rounds used (citewise)" in table

    def test_empty_arm_renders_without_crashing(self):
        assert "| citewise | 0 |" in markdown_table([aggregate([], "citewise")])


class TestLabelingSheet:
    SAMPLES = [
        {
            "topic_id": "t06",
            "arm": "citewise",
            "evidence": [{"id": "e1", "snippet": "the snippet text"}],
            "draft": {
                "claims": [
                    {
                        "id": "c1",
                        "text": "a claim",
                        "evidence_ids": ["e1"],
                        "verdict": "SUPPORTED",
                        "verdict_reason": "stated directly",
                    },
                    {
                        "id": "c2",
                        "text": "another",
                        "evidence_ids": ["e9"],
                        "verdict": "UNSUPPORTED",
                        "verdict_reason": "not addressed",
                    },
                ]
            },
        }
    ]

    def test_human_verdict_column_is_left_blank(self):
        """The user fills this in by hand; we build the tooling only."""
        rows = build_labeling_rows(self.SAMPLES)
        assert all(row["human_verdict"] == "" for row in rows)

    def test_carries_claim_evidence_and_verifier_verdict(self):
        rows = build_labeling_rows(self.SAMPLES)
        assert rows[0]["claim"] == "a claim"
        assert "the snippet text" in rows[0]["cited_evidence"]
        assert rows[0]["verifier_verdict"] == "SUPPORTED"

    def test_claim_citing_missing_evidence_is_marked(self):
        rows = build_labeling_rows(self.SAMPLES)
        assert rows[1]["cited_evidence"] == "(no cited evidence in pool)"

    def test_sampling_is_capped_and_deterministic(self):
        big = [
            {
                **self.SAMPLES[0],
                "draft": {
                    "claims": [
                        {
                            "id": f"c{i}",
                            "text": f"claim {i}",
                            "evidence_ids": ["e1"],
                            "verdict": "SUPPORTED",
                            "verdict_reason": "r",
                        }
                        for i in range(100)
                    ]
                },
            }
        ]
        first = build_labeling_rows(big, target=60)
        assert len(first) == 60
        assert [r["claim_id"] for r in first] == [
            r["claim_id"] for r in build_labeling_rows(big, target=60)
        ]

    def test_written_sheet_has_the_expected_columns(self, tmp_path):
        path = write_labeling_sheet(build_labeling_rows(self.SAMPLES), tmp_path / "sheet.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == LABELING_COLUMNS
            assert len(list(reader)) == 2


# ---------------------------------------------------------------------------
# score_labels.py against a synthetic filled sheet
# ---------------------------------------------------------------------------


def write_sheet(tmp_path: Path, pairs: list[tuple[str, str]], name="sheet.csv") -> Path:
    """pairs are (verifier_verdict, human_verdict); "" means unlabelled."""
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABELING_COLUMNS)
        writer.writeheader()
        for i, (verifier, human) in enumerate(pairs):
            writer.writerow(
                {
                    "topic_id": "t06",
                    "arm": "citewise",
                    "claim_id": f"c{i}",
                    "claim": f"claim {i}",
                    "cited_evidence": "e",
                    "verifier_verdict": verifier,
                    "verifier_reason": "r",
                    "human_verdict": human,
                }
            )
    return path


class TestReadLabeledRows:
    def test_skips_blank_human_verdicts(self, tmp_path):
        """A blank human verdict must not be counted as agreement."""
        path = write_sheet(tmp_path, [("SUPPORTED", "SUPPORTED"), ("SUPPORTED", "")])
        assert len(read_labeled_rows(path)) == 1

    def test_verdicts_are_case_insensitive(self, tmp_path):
        path = write_sheet(tmp_path, [("SUPPORTED", " supported ")])
        assert read_labeled_rows(path)[0]["human"] == "SUPPORTED"

    def test_unfilled_sheet_raises(self, tmp_path):
        path = write_sheet(tmp_path, [("SUPPORTED", ""), ("UNSUPPORTED", "")])
        with pytest.raises(LabelingSheetError, match="no filled human_verdict"):
            read_labeled_rows(path)

    def test_bad_verdict_raises_with_the_line_number(self, tmp_path):
        path = write_sheet(tmp_path, [("SUPPORTED", "MAYBE")])
        with pytest.raises(LabelingSheetError, match="line 2"):
            read_labeled_rows(path)

    def test_missing_column_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("claim,verifier_verdict\na,SUPPORTED\n", encoding="utf-8")
        with pytest.raises(LabelingSheetError, match="human_verdict"):
            read_labeled_rows(path)


class TestScoreLabels:
    def test_perfect_agreement(self, tmp_path):
        path = write_sheet(
            tmp_path,
            [
                ("SUPPORTED", "SUPPORTED"),
                ("UNSUPPORTED", "UNSUPPORTED"),
                ("CONTRADICTED", "CONTRADICTED"),
            ],
        )
        result = score(read_labeled_rows(path))
        assert result["exact_agreement"] == 1.0
        assert result["flagging"]["precision"] == 1.0
        assert result["flagging"]["recall"] == 1.0

    def test_flagging_precision_and_recall(self, tmp_path):
        # verifier flags 3 (c1,c2,c4); humans reject 3 (c1,c2,c5).
        # TP=2 (c1,c2), FP=1 (c4), FN=1 (c5)
        path = write_sheet(
            tmp_path,
            [
                ("UNSUPPORTED", "UNSUPPORTED"),  # TP
                ("CONTRADICTED", "CONTRADICTED"),  # TP
                ("SUPPORTED", "SUPPORTED"),  # TN
                ("UNSUPPORTED", "SUPPORTED"),  # FP
                ("SUPPORTED", "UNSUPPORTED"),  # FN
            ],
        )
        flagging = score(read_labeled_rows(path))["flagging"]
        assert flagging["true_positives"] == 2
        assert flagging["false_positives"] == 1
        assert flagging["false_negatives"] == 1
        assert flagging["precision"] == pytest.approx(2 / 3, abs=1e-4)
        assert flagging["recall"] == pytest.approx(2 / 3, abs=1e-4)

    def test_verifier_too_lenient_shows_low_recall(self, tmp_path):
        path = write_sheet(
            tmp_path,
            [
                ("SUPPORTED", "UNSUPPORTED"),
                ("SUPPORTED", "UNSUPPORTED"),
                ("SUPPORTED", "SUPPORTED"),
            ],
        )
        flagging = score(read_labeled_rows(path))["flagging"]
        assert flagging["recall"] == 0.0
        assert flagging["precision"] is None, "nothing was flagged, so precision is undefined"

    def test_unsupported_vs_contradicted_mixup_is_not_exact_agreement(self, tmp_path):
        """Both are 'flagged', so flagging is perfect while exact agreement is not."""
        path = write_sheet(tmp_path, [("UNSUPPORTED", "CONTRADICTED")])
        result = score(read_labeled_rows(path))
        assert result["exact_agreement"] == 0.0
        assert result["flagging"]["precision"] == 1.0

    def test_per_class_and_confusion_are_reported(self, tmp_path):
        path = write_sheet(tmp_path, [("SUPPORTED", "SUPPORTED"), ("UNSUPPORTED", "SUPPORTED")])
        result = score(read_labeled_rows(path))
        assert result["per_class"]["SUPPORTED"]["true_positives"] == 1
        assert result["confusion"]["human=SUPPORTED|verifier=UNSUPPORTED"] == 1

    def test_report_renders(self, tmp_path):
        path = write_sheet(tmp_path, [("SUPPORTED", "SUPPORTED"), ("UNSUPPORTED", "UNSUPPORTED")])
        report = format_report(score(read_labeled_rows(path)))
        assert "Verifier vs human labels" in report
        assert "Precision:" in report
        assert "| SUPPORTED |" in report
