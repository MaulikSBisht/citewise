"""Unit tests for the forward-path nodes: planner, searcher, writer."""

import pytest

from citewise.config import Config
from citewise.llm import StructuredLLM
from citewise.nodes.planner import make_planner_node
from citewise.nodes.searcher import make_searcher_node
from citewise.nodes.writer import build_writer_prompt, make_writer_node
from citewise.search import EvidenceSearcher
from citewise.state import EvidenceChunk, initial_state
from tests.fixtures import FakeAnthropic, FakeTavily, canned, partitioned, tavily_payload


def evidence_pool(n: int = 7) -> list[EvidenceChunk]:
    results = tavily_payload("rich")["results"][:n]
    return [
        EvidenceChunk(
            id=f"e{i + 1}",
            url=r["url"],
            title=r["title"],
            snippet=r["content"],
            sub_question="q",
        )
        for i, r in enumerate(results)
    ]


class TestPlanner:
    def test_returns_sub_questions(self):
        node = make_planner_node(
            StructuredLLM(client=FakeAnthropic([canned("planner_ok")])), Config()
        )
        out = node(initial_state("remote work"))
        assert len(out["sub_questions"]) == 4
        assert out.get("aborted_reason") is None

    def test_truncates_to_max_sub_questions(self):
        """MAX_SUB_QUESTIONS is a cost control — enforce it, don't trust the prompt."""
        node = make_planner_node(
            StructuredLLM(client=FakeAnthropic([canned("planner_too_many")])),
            Config(max_sub_questions=5),
        )
        assert len(node(initial_state("t"))["sub_questions"]) == 5

    def test_uses_the_planner_model_from_config(self):
        client = FakeAnthropic([canned("planner_ok")])
        node = make_planner_node(StructuredLLM(client=client), Config())
        node(initial_state("t"))
        assert client.calls[0]["model"] == "claude-haiku-4-5"

    def test_blank_questions_are_dropped(self):
        client = FakeAnthropic(['{"sub_questions": ["  ", "", "a real question"]}'])
        node = make_planner_node(StructuredLLM(client=client), Config())
        assert node(initial_state("t"))["sub_questions"] == ["a real question"]

    def test_all_blank_aborts(self):
        client = FakeAnthropic(['{"sub_questions": ["  ", ""]}'])
        node = make_planner_node(StructuredLLM(client=client), Config())
        assert "no usable sub-questions" in node(initial_state("t"))["aborted_reason"]

    def test_writes_a_trace_entry(self):
        node = make_planner_node(
            StructuredLLM(client=FakeAnthropic([canned("planner_ok")])), Config()
        )
        trace = node(initial_state("t"))["trace"]
        assert trace[0]["node"] == "planner"


class TestSearcher:
    def _node(self, payload_name, n_queries=1, **cfg):
        cfg.setdefault("min_evidence_chunks", 6)
        config = Config(**cfg)
        searcher = EvidenceSearcher(
            client=FakeTavily(payloads=partitioned(payload_name, n_queries)), config=config
        )
        return make_searcher_node(searcher, config)

    def test_returns_evidence_when_pool_is_rich_enough(self):
        state = initial_state("t") | {"sub_questions": ["q1", "q2"]}
        out = self._node("rich", n_queries=2, results_per_question=5)(state)
        assert len(out["evidence"]) == 7
        assert out.get("aborted_reason") is None

    def test_aborts_below_min_evidence_chunks(self):
        """Rule 6: fail loudly on thin evidence rather than letting the writer paper over it."""
        state = initial_state("t") | {"sub_questions": ["q1"]}
        out = self._node("thin")(state)
        assert "thin evidence" in out["aborted_reason"]
        assert "MIN_EVIDENCE_CHUNKS threshold of 6" in out["aborted_reason"]

    def test_empty_results_abort(self):
        state = initial_state("t") | {"sub_questions": ["q1"]}
        assert self._node("empty")(state)["aborted_reason"] is not None

    def test_threshold_is_read_from_config(self):
        state = initial_state("t") | {"sub_questions": ["q1"]}
        out = self._node("thin", min_evidence_chunks=1)(state)
        assert out.get("aborted_reason") is None

    def test_makes_no_llm_call(self):
        """The searcher node has no LLM dependency at all — nothing to assert against."""
        state = initial_state("t") | {"sub_questions": ["q1", "q2"]}
        out = self._node("rich", n_queries=2, results_per_question=5)(state)
        assert [t["node"] for t in out["trace"]] == ["searcher"]


class TestWriter:
    def _state(self, n=7):
        return initial_state("remote work") | {"evidence": evidence_pool(n)}

    def test_produces_a_draft(self):
        node = make_writer_node(
            StructuredLLM(client=FakeAnthropic([canned("writer_ok")])), Config()
        )
        draft = node(self._state())["draft"]
        assert draft.title == "Remote Work and Developer Productivity"
        assert len(draft.claims) == 3

    def test_claims_start_pending(self):
        node = make_writer_node(
            StructuredLLM(client=FakeAnthropic([canned("writer_ok")])), Config()
        )
        draft = node(self._state())["draft"]
        assert all(c.verdict == "PENDING" for c in draft.claims)

    def test_rejects_draft_citing_unknown_evidence_id(self):
        """Phase 2 done-condition."""
        node = make_writer_node(
            StructuredLLM(client=FakeAnthropic([canned("writer_unknown_evidence_id")])), Config()
        )
        out = node(self._state())
        assert out["draft"] is None
        assert "e99" in out["aborted_reason"]

    def test_uses_the_writer_model_from_config(self):
        client = FakeAnthropic([canned("writer_ok")])
        make_writer_node(StructuredLLM(client=client), Config())(self._state())
        assert client.calls[0]["model"] == "claude-sonnet-4-6"

    def test_empty_evidence_ids_are_rejected_by_the_schema(self):
        """Rule 5 reaches the writer through the LLM wrapper's validation retry."""
        from citewise.llm import StructuredOutputError

        client = FakeAnthropic(
            [canned("schema_violation_empty_evidence"), canned("schema_violation_empty_evidence")]
        )
        node = make_writer_node(StructuredLLM(client=client), Config())
        with pytest.raises(StructuredOutputError):
            node(self._state())


class TestWriterPrompt:
    def test_prompt_lists_every_valid_evidence_id(self):
        prompt = build_writer_prompt("t", evidence_pool(7))
        assert "Valid evidence IDs: e1, e2, e3, e4, e5, e6, e7" in prompt

    def test_prompt_includes_snippet_text(self):
        prompt = build_writer_prompt("t", evidence_pool(1))
        assert "8% more code per week" in prompt

    def test_prompt_carries_the_topic(self):
        assert "Topic: remote work" in build_writer_prompt("remote work", evidence_pool(1))
