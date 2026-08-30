"""LLM wrapper tests, including the JSON-repair path.

Phase 1 done-condition: the repair path is covered by a test, and no test
touches the network.
"""

import pytest
from pydantic import BaseModel, Field

from citewise.llm import StructuredLLM, StructuredOutputError, to_strict_json_schema
from tests.fixtures import FakeAnthropic, FakeMessage, canned


class Plan(BaseModel):
    sub_questions: list[str]


class Draft(BaseModel):
    title: str
    claims: list[str] = Field(min_length=1)


def make_llm(responses) -> tuple[StructuredLLM, FakeAnthropic]:
    client = FakeAnthropic(responses)
    return StructuredLLM(client=client), client


class TestHappyPath:
    def test_parses_valid_json_in_one_call(self):
        llm, client = make_llm([canned("planner_ok")])
        result = llm.complete_structured("sys", "user", Plan, "claude-haiku-4-5")
        assert isinstance(result, Plan)
        assert len(result.sub_questions) == 4
        assert client.call_count == 1

    def test_passes_model_and_schema_to_the_api(self):
        llm, client = make_llm([canned("planner_ok")])
        llm.complete_structured("sys prompt", "user prompt", Plan, "claude-haiku-4-5")
        call = client.calls[0]
        assert call["model"] == "claude-haiku-4-5"
        assert call["system"] == "sys prompt"
        assert call["messages"][0]["content"] == "user prompt"
        fmt = call["output_config"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["schema"]["properties"]["sub_questions"]["type"] == "array"


class TestToleratedWrappers:
    """Fenced or prose-wrapped JSON is recovered without spending a repair call."""

    def test_strips_code_fence(self):
        llm, client = make_llm([canned("malformed_fenced")])
        result = llm.complete_structured("s", "u", Plan, "m")
        assert len(result.sub_questions) == 3
        assert client.call_count == 1

    def test_extracts_json_from_surrounding_prose(self):
        llm, client = make_llm([canned("malformed_prose_wrapped")])
        result = llm.complete_structured("s", "u", Plan, "m")
        assert len(result.sub_questions) == 3
        assert client.call_count == 1


class TestRepairPath:
    def test_retries_once_on_unparseable_json_then_succeeds(self):
        llm, client = make_llm([canned("malformed_truncated"), canned("planner_ok")])
        result = llm.complete_structured("s", "u", Plan, "m")
        assert len(result.sub_questions) == 4
        assert client.call_count == 2

    def test_retries_once_on_schema_violation_then_succeeds(self):
        bad = '{"title": "R", "claims": []}'  # violates min_length=1
        good = '{"title": "R", "claims": ["a claim"]}'
        llm, client = make_llm([bad, good])
        result = llm.complete_structured("s", "u", Draft, "m")
        assert result.claims == ["a claim"]
        assert client.call_count == 2

    def test_repair_call_feeds_the_error_back_to_the_model(self):
        llm, client = make_llm([canned("malformed_truncated"), canned("planner_ok")])
        llm.complete_structured("s", "u", Plan, "m")

        repair_messages = client.calls[1]["messages"]
        assert [m["role"] for m in repair_messages] == ["user", "assistant", "user"]
        assert repair_messages[1]["content"] == canned("malformed_truncated")
        feedback = repair_messages[2]["content"]
        assert "did not parse" in feedback or "did not validate" in feedback
        assert "Return only" in feedback

    def test_schema_error_text_reaches_the_repair_prompt(self):
        llm, client = make_llm(['{"title": "R", "claims": []}', '{"title": "R", "claims": ["x"]}'])
        llm.complete_structured("s", "u", Draft, "m")
        feedback = client.calls[1]["messages"][2]["content"]
        assert "claims" in feedback

    def test_raises_after_a_second_failure(self):
        llm, client = make_llm([canned("malformed_truncated"), canned("malformed_truncated")])
        with pytest.raises(StructuredOutputError) as exc:
            llm.complete_structured("s", "u", Plan, "m")
        assert client.call_count == 2, "must retry exactly once, then raise"
        assert "after 2 attempt" in str(exc.value)

    def test_raises_when_second_attempt_violates_schema(self):
        llm, client = make_llm(['{"title": "R", "claims": []}', '{"title": "R", "claims": []}'])
        with pytest.raises(StructuredOutputError):
            llm.complete_structured("s", "u", Draft, "m")
        assert client.call_count == 2


class TestDegenerateResponses:
    def test_empty_content_is_a_failure(self):
        llm, client = make_llm([FakeMessage(None), canned("planner_ok")])
        result = llm.complete_structured("s", "u", Plan, "m")
        assert len(result.sub_questions) == 4
        assert client.call_count == 2

    def test_refusal_raises_without_retrying(self):
        llm, client = make_llm([FakeMessage("", stop_reason="refusal")])
        with pytest.raises(StructuredOutputError, match="refus"):
            llm.complete_structured("s", "u", Plan, "m")
        assert client.call_count == 1, "a refusal is not repairable — do not spend a retry"


class TestStrictJSONSchema:
    def test_objects_get_additional_properties_false(self):
        schema = to_strict_json_schema(Plan)
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["sub_questions"]

    def test_nested_defs_are_made_strict(self):
        class Inner(BaseModel):
            a: str

        class Outer(BaseModel):
            inner: Inner
            items: list[Inner]

        schema = to_strict_json_schema(Outer)
        inner = schema["$defs"]["Inner"]
        assert inner["additionalProperties"] is False
        assert inner["required"] == ["a"]
        assert set(schema["required"]) == {"inner", "items"}
