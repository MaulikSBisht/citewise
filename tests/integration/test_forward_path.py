"""Phase 2 done-condition: drive the graph from topic to ReportDraft on fakes only."""

from citewise.config import Config
from citewise.graph import run
from citewise.llm import StructuredLLM
from citewise.search import EvidenceSearcher
from citewise.state import ReportDraft
from tests.fixtures import FakeAnthropic, FakeTavily, canned, partitioned


def drive(anthropic_responses, tavily_payloads=None, **cfg):
    """`planner_ok` yields 4 sub-questions, so the rich pool is split 4 ways."""
    config = Config(**cfg)
    llm = StructuredLLM(client=FakeAnthropic(anthropic_responses), config=config)
    searcher = EvidenceSearcher(
        client=FakeTavily(payloads=tavily_payloads or partitioned("rich", 4)), config=config
    )
    return run("Does remote work increase developer productivity?", llm, searcher, config)


class TestHappyPath:
    def test_topic_to_report_draft(self):
        final = drive([canned("planner_ok"), canned("writer_ok")])

        assert final["aborted_reason"] is None
        assert isinstance(final["draft"], ReportDraft)
        assert len(final["draft"].claims) == 3
        assert len(final["sub_questions"]) == 4
        assert len(final["evidence"]) >= 6

    def test_every_claim_cites_evidence_that_exists(self):
        final = drive([canned("planner_ok"), canned("writer_ok")])
        pool = {c.id for c in final["evidence"]}
        for claim in final["draft"].claims:
            assert claim.evidence_ids
            assert set(claim.evidence_ids) <= pool

    def test_trace_records_each_node_in_order(self):
        final = drive([canned("planner_ok"), canned("writer_ok")])
        assert [t["node"] for t in final["trace"]] == ["planner", "searcher", "writer"]

    def test_retry_count_untouched_on_the_forward_path(self):
        assert drive([canned("planner_ok"), canned("writer_ok")])["retry_count"] == 0


class TestRejection:
    def test_draft_citing_nonexistent_evidence_id_is_rejected(self):
        """Phase 2 done-condition: the graph must not carry such a draft forward."""
        final = drive([canned("planner_ok"), canned("writer_unknown_evidence_id")])

        assert final["draft"] is None
        assert "e99" in final["aborted_reason"]


class TestShortCircuit:
    def test_thin_evidence_never_reaches_the_writer(self):
        """Rule 6 — and the writer's canned response must go unconsumed."""
        config = Config()
        client = FakeAnthropic([canned("planner_ok"), canned("writer_ok")])
        llm = StructuredLLM(client=client, config=config)
        searcher = EvidenceSearcher(
            client=FakeTavily(payloads=partitioned("thin", 4)), config=config
        )

        final = run("topic", llm, searcher, config)

        assert final["draft"] is None
        assert "thin evidence" in final["aborted_reason"]
        assert client.call_count == 1, "writer must not have been called"
        assert [t["node"] for t in final["trace"]] == ["planner", "searcher"]
