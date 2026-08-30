"""Phase 3 done-condition.

Two things must be true: a deliberately hallucinated claim is caught, fed back,
and either fixed or flagged; and the loop provably terminates at MAX_RETRIES.
"""

from citewise.config import Config
from citewise.graph import load_run, run, save_run
from citewise.llm import StructuredLLM
from citewise.search import EvidenceSearcher
from tests.fixtures import FakeAnthropic, FakeTavily, canned, partitioned


def drive(anthropic_responses, **cfg):
    config = Config(**cfg)
    llm = StructuredLLM(client=FakeAnthropic(anthropic_responses), config=config)
    searcher = EvidenceSearcher(client=FakeTavily(payloads=partitioned("rich", 4)), config=config)
    final = run("Does remote work increase developer productivity?", llm, searcher, config)
    return final, llm.client


class TestHallucinationCaught:
    """`writer_hallucinated` c2 says remote work 'doubles productivity across every
    industry' while citing a single-study chunk. The verifier must reject it."""

    RESPONSES = [
        canned("planner_ok"),
        canned("writer_hallucinated"),
        canned("verifier_supported"),  # c1
        canned("verifier_unsupported"),  # c2 — the hallucination
        canned("revision_c2"),  # writer revises only c2
        canned("verifier_supported"),  # revised c2
    ]

    def test_loop_catches_fixes_and_finishes_clean(self):
        final, _ = drive(self.RESPONSES)

        assert final["aborted_reason"] is None
        assert final["retry_count"] == 1, "exactly one loop back to the writer"
        assert [c.verdict for c in final["draft"].claims] == ["SUPPORTED", "SUPPORTED"]

    def test_the_hallucinated_text_is_gone_from_the_report(self):
        final, _ = drive(self.RESPONSES)
        assert "double developer productivity" not in final["final_report"]
        assert "62% of surveyed remote developers" in final["final_report"]

    def test_the_failure_and_its_reason_are_fed_back_to_the_writer(self):
        final, client = drive(self.RESPONSES)
        revision_prompt = client.calls[4]["messages"][0]["content"]

        assert "FAILING CLAIMS" in revision_prompt
        assert "c2" in revision_prompt
        assert "does not support a claim about every industry" in revision_prompt

    def test_the_supported_claim_is_frozen_not_regenerated(self):
        final, client = drive(self.RESPONSES)
        revision_prompt = client.calls[4]["messages"][0]["content"]

        assert "frozen — do not revise them" in revision_prompt
        assert final["draft"].claim_map()["c1"].text.startswith("A 2023 controlled study")

    def test_the_frozen_claim_is_not_reverified(self):
        """6 calls: planner, writer, 2 verifies, revision, 1 verify of the revised claim."""
        _, client = drive(self.RESPONSES)
        assert client.call_count == 6

    def test_trace_shows_the_round_trip(self):
        final, _ = drive(self.RESPONSES)
        assert [t["node"] for t in final["trace"]] == [
            "planner",
            "searcher",
            "writer",
            "verifier",
            "writer",
            "verifier",
            "finalizer",
        ]


class TestFlaggedRatherThanFixed:
    def test_claim_that_never_passes_ships_marked_unverified(self):
        """Never silently ship a failed claim — keep it, but mark it."""
        responses = [
            canned("planner_ok"),
            canned("writer_hallucinated"),
            canned("verifier_supported"),
            canned("verifier_unsupported"),
            canned("revision_c2"),
            canned("verifier_unsupported"),
            canned("revision_c2"),
            canned("verifier_unsupported"),
        ]
        final, _ = drive(responses)

        assert final["retry_count"] == 2
        assert final["draft"].claim_map()["c2"].verdict == "UNSUPPORTED"
        assert "[unverified]" in final["final_report"]
        assert "## Unverified claims" in final["final_report"]

    def test_contradicted_claim_is_dropped_from_the_report(self):
        responses = [
            canned("planner_ok"),
            canned("writer_hallucinated"),
            canned("verifier_supported"),
            canned("verifier_contradicted"),
            canned("revision_c2"),
            canned("verifier_contradicted"),
            canned("revision_c2"),
            canned("verifier_contradicted"),
        ]
        final, _ = drive(responses)

        assert [c.id for c in final["draft"].claims] == ["c1"]
        assert "62% of surveyed remote developers" not in final["final_report"]


class TestTermination:
    """An infinite loop here is a bug — these tests are the proof it terminates."""

    def _always_failing(self, rounds: int) -> list[str]:
        responses = [canned("planner_ok"), canned("writer_hallucinated")]
        for _ in range(rounds):
            responses += [canned("verifier_unsupported"), canned("verifier_unsupported")]
            responses.append(canned("revision_both"))
        responses += [canned("verifier_unsupported"), canned("verifier_unsupported")]
        return responses

    def test_stops_at_max_retries_of_two(self):
        final, client = drive(self._always_failing(rounds=2))

        assert final["retry_count"] == 2
        assert final["final_report"] is not None
        assert final["aborted_reason"] is None

    def test_writer_runs_exactly_one_plus_max_retries_times(self):
        final, _ = drive(self._always_failing(rounds=2))
        writer_rounds = [t for t in final["trace"] if t["node"] == "writer"]
        assert len(writer_rounds) == 3
        assert [t["mode"] for t in writer_rounds] == ["draft", "revision", "revision"]

    def test_max_retries_zero_never_loops(self):
        responses = [
            canned("planner_ok"),
            canned("writer_hallucinated"),
            canned("verifier_unsupported"),
            canned("verifier_unsupported"),
        ]
        final, client = drive(responses, max_retries=0)

        assert final["retry_count"] == 0
        assert client.call_count == 4, "no revision call may be made"
        assert final["final_report"] is not None

    def test_max_retries_one_loops_once(self):
        responses = [
            canned("planner_ok"),
            canned("writer_hallucinated"),
            canned("verifier_unsupported"),
            canned("verifier_unsupported"),
            canned("revision_both"),
            canned("verifier_unsupported"),
            canned("verifier_unsupported"),
        ]
        final, _ = drive(responses, max_retries=1)
        assert final["retry_count"] == 1


class TestPersistence:
    def test_run_is_written_as_json_with_the_full_trace(self, tmp_path):
        final, _ = drive(TestHallucinationCaught.RESPONSES)
        path = save_run(final, runs_dir=tmp_path)

        assert path.parent == tmp_path
        assert path.suffix == ".json"

        payload = load_run(path)
        assert payload["topic"].startswith("Does remote work")
        assert payload["retry_count"] == 1
        assert payload["final_report"].startswith("# ")
        assert [t["node"] for t in payload["trace"]] == [
            "planner",
            "searcher",
            "writer",
            "verifier",
            "writer",
            "verifier",
            "finalizer",
        ]
        assert len(payload["evidence"]) == 7
        assert payload["draft"]["claims"][0]["verdict"] == "SUPPORTED"
