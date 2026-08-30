"""Tavily wrapper tests: stable IDs, dedupe by URL, no network."""

from citewise.config import Config
from citewise.search import EvidenceSearcher
from tests.fixtures import FakeTavily, tavily_payload


def make_searcher(payloads=None, per_query=None, **cfg):
    client = FakeTavily(payloads=payloads, per_query=per_query)
    return EvidenceSearcher(client=client, config=Config(**cfg)), client


class TestIDAssignment:
    def test_ids_are_sequential_and_stable(self):
        searcher, _ = make_searcher(tavily_payload("rich"))
        chunks = searcher.search(["q1"])
        assert [c.id for c in chunks] == ["e1", "e2", "e3", "e4", "e5"]

    def test_ids_continue_across_sub_questions(self):
        searcher, _ = make_searcher(
            per_query={
                "q1": tavily_payload("thin"),
                "q2": {"results": tavily_payload("rich")["results"][:2]},
            }
        )
        chunks = searcher.search(["q1", "q2"])
        assert [c.id for c in chunks] == ["e1", "e2", "e3"]

    def test_chunk_carries_its_sub_question(self):
        searcher, _ = make_searcher(per_query={"first question": tavily_payload("thin")})
        chunks = searcher.search(["first question"])
        assert chunks[0].sub_question == "first question"


class TestDedupe:
    def test_duplicate_urls_within_one_query_collapse(self):
        searcher, _ = make_searcher(tavily_payload("duplicate_urls"))
        chunks = searcher.search(["q1"])
        assert len(chunks) == 2
        assert len({c.url for c in chunks}) == 2

    def test_duplicate_urls_across_queries_collapse(self):
        payload = {"results": tavily_payload("rich")["results"][:3]}
        searcher, _ = make_searcher(per_query={"q1": payload, "q2": payload})
        chunks = searcher.search(["q1", "q2"])
        assert len(chunks) == 3
        assert [c.id for c in chunks] == ["e1", "e2", "e3"]

    def test_first_occurrence_keeps_its_sub_question(self):
        payload = {"results": tavily_payload("rich")["results"][:1]}
        searcher, _ = make_searcher(per_query={"q1": payload, "q2": payload})
        chunks = searcher.search(["q1", "q2"])
        assert chunks[0].sub_question == "q1"


class TestLimits:
    def test_results_per_question_is_read_from_config(self):
        searcher, client = make_searcher(tavily_payload("rich"), results_per_question=2)
        chunks = searcher.search(["q1"])
        assert len(chunks) == 2
        assert client.calls[0]["max_results"] == 2

    def test_every_sub_question_is_queried(self):
        searcher, client = make_searcher(tavily_payload("thin"))
        searcher.search(["a", "b", "c"])
        assert client.queries == ["a", "b", "c"]


class TestDegenerate:
    def test_empty_results_yield_no_chunks(self):
        searcher, _ = make_searcher(tavily_payload("empty"))
        assert searcher.search(["q1"]) == []

    def test_results_missing_fields_are_skipped_not_crashed(self):
        searcher, _ = make_searcher(
            {"results": [{"url": "", "title": "no url", "content": "x"}, {"title": "no url key"}]}
        )
        assert searcher.search(["q1"]) == []

    def test_blank_content_is_skipped(self):
        searcher, _ = make_searcher(
            {"results": [{"url": "https://a.example", "title": "t", "content": "   "}]}
        )
        assert searcher.search(["q1"]) == []
