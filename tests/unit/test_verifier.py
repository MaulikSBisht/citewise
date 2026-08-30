"""Verifier tests, with the isolation guarantee (rule 4) as the headline case."""

from citewise.config import Config
from citewise.llm import StructuredLLM
from citewise.nodes.verifier import (
    VERIFIER_SYSTEM,
    build_verifier_prompt,
    failing_claims,
    make_verifier_node,
)
from citewise.state import Claim, ReportDraft, Section, initial_state
from tests.fixtures import FakeAnthropic, canned
from tests.unit.test_nodes_forward import evidence_pool


def draft_with(*claims: Claim) -> ReportDraft:
    return ReportDraft(
        title="Remote Work and Developer Productivity",
        sections=[Section(heading="Findings", claim_ids=[c.id for c in claims])],
        claims=list(claims),
    )


def claim(cid="c1", text="A claim.", evidence_ids=("e1",), verdict="PENDING"):
    return Claim(id=cid, text=text, evidence_ids=list(evidence_ids), verdict=verdict)


class TestIsolation:
    """Rule 4: the verifier sees a claim and evidence. Nothing else."""

    def test_topic_never_reaches_the_verifier(self):
        """Drive the real node with a sentinel topic and inspect every call it made."""
        sentinel = "ZZQX-SENTINEL-TOPIC-FRAMING"
        client = FakeAnthropic([canned("verifier_supported"), canned("verifier_supported")])
        node = make_verifier_node(StructuredLLM(client=client), Config())
        state = initial_state(sentinel) | {
            "draft": draft_with(claim("c1"), claim("c2")),
            "evidence": evidence_pool(7),
        }
        node(state)

        assert client.call_count == 2
        for call in client.calls:
            assert sentinel not in call["system"]
            assert sentinel not in call["messages"][0]["content"]

    def test_prompt_excludes_other_claims(self):
        target = claim("c1", "The claim under test.")
        other = claim("c2", "A completely different assertion about onboarding.")
        draft_with(target, other)
        prompt = build_verifier_prompt(target, evidence_pool(7))
        assert "completely different assertion" not in prompt

    def test_sibling_claims_never_reach_the_verifier(self):
        """The node has the whole draft in hand — prove it does not pass siblings through."""
        client = FakeAnthropic([canned("verifier_supported"), canned("verifier_supported")])
        node = make_verifier_node(StructuredLLM(client=client), Config())
        state = initial_state("t") | {
            "draft": draft_with(
                claim("c1", "ALPHA-DISTINCTIVE-CLAIM-TEXT"),
                claim("c2", "BETA-DISTINCTIVE-CLAIM-TEXT"),
            ),
            "evidence": evidence_pool(7),
        }
        node(state)

        first, second = (c["messages"][0]["content"] for c in client.calls)
        assert "ALPHA-DISTINCTIVE-CLAIM-TEXT" in first
        assert "BETA-DISTINCTIVE-CLAIM-TEXT" not in first
        assert "BETA-DISTINCTIVE-CLAIM-TEXT" in second
        assert "ALPHA-DISTINCTIVE-CLAIM-TEXT" not in second

    def test_prompt_excludes_the_draft_title(self):
        prompt = build_verifier_prompt(claim(), evidence_pool(7))
        assert "Remote Work and Developer Productivity" not in prompt

    def test_prompt_contains_the_claim_and_its_cited_evidence(self):
        prompt = build_verifier_prompt(claim(evidence_ids=("e1",)), evidence_pool(7))
        assert "A claim." in prompt
        assert "8% more code per week" in prompt

    def test_uncited_evidence_is_labelled_as_not_support(self):
        prompt = build_verifier_prompt(claim(evidence_ids=("e1",)), evidence_pool(7))
        cited_at = prompt.index("EVIDENCE CITED FOR THIS CLAIM")
        others_at = prompt.index("OTHER AVAILABLE EVIDENCE")
        assert cited_at < others_at
        assert "never as support" in prompt

    def test_wider_pool_is_present_for_contradiction_detection(self):
        prompt = build_verifier_prompt(claim(evidence_ids=("e1",)), evidence_pool(7))
        assert "[e6]" in prompt and "pooled effect on productivity was small" in prompt

    def test_system_prompt_forbids_using_outside_knowledge(self):
        assert "knowledge of your own" in VERIFIER_SYSTEM


class TestVerdicts:
    def _run(self, responses, *claims):
        client = FakeAnthropic(responses)
        node = make_verifier_node(StructuredLLM(client=client), Config())
        state = initial_state("t") | {
            "draft": draft_with(*claims),
            "evidence": evidence_pool(7),
        }
        return node(state), client

    def test_records_verdict_and_reason(self):
        out, _ = self._run([canned("verifier_supported")], claim())
        verified = out["draft"].claims[0]
        assert verified.verdict == "SUPPORTED"
        assert "8% figure" in verified.verdict_reason

    def test_one_call_per_claim(self):
        out, client = self._run(
            [canned("verifier_supported"), canned("verifier_unsupported")],
            claim("c1"),
            claim("c2"),
        )
        assert client.call_count == 2
        assert [c.verdict for c in out["draft"].claims] == ["SUPPORTED", "UNSUPPORTED"]

    def test_each_call_gets_a_fresh_message_list(self):
        """No claim's verdict may influence another's."""
        _, client = self._run(
            [canned("verifier_supported"), canned("verifier_unsupported")],
            claim("c1", "First claim."),
            claim("c2", "Second claim."),
        )
        second_prompt = client.calls[1]["messages"][0]["content"]
        assert "First claim." not in second_prompt
        assert all(len(call["messages"]) == 1 for call in client.calls)

    def test_contradicted_verdict_is_recorded(self):
        out, _ = self._run([canned("verifier_contradicted")], claim())
        assert out["draft"].claims[0].verdict == "CONTRADICTED"

    def test_uses_the_verifier_model_from_config(self):
        _, client = self._run([canned("verifier_supported")], claim())
        assert client.calls[0]["model"] == "claude-sonnet-4-6"


class TestFrozenClaims:
    def test_already_supported_claims_are_not_reverified(self):
        client = FakeAnthropic([canned("verifier_unsupported")])
        node = make_verifier_node(StructuredLLM(client=client), Config())
        state = initial_state("t") | {
            "draft": draft_with(claim("c1", verdict="SUPPORTED"), claim("c2")),
            "evidence": evidence_pool(7),
        }
        out = node(state)
        assert client.call_count == 1, "the frozen claim must not cost a second call"
        assert [c.verdict for c in out["draft"].claims] == ["SUPPORTED", "UNSUPPORTED"]

    def test_no_draft_is_a_no_op(self):
        client = FakeAnthropic([])
        node = make_verifier_node(StructuredLLM(client=client), Config())
        out = node(initial_state("t"))
        assert client.call_count == 0
        assert "skipped" in out["trace"][0]


class TestFailingClaims:
    def test_selects_everything_not_supported(self):
        draft = draft_with(
            claim("c1", verdict="SUPPORTED"),
            claim("c2", verdict="UNSUPPORTED"),
            claim("c3", verdict="CONTRADICTED"),
            claim("c4", verdict="PENDING"),
        )
        assert [c.id for c in failing_claims(draft)] == ["c2", "c3", "c4"]
