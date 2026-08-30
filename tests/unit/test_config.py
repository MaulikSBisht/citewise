"""Config tests. Section 5: every limit is a cost or safety control, so it must
be readable from config rather than hardcoded at a call site."""

import pytest

from citewise.config import Config, load_config


class TestDefaults:
    def test_section_5_defaults(self):
        c = Config()
        assert c.max_sub_questions == 5
        assert c.results_per_question == 5
        assert c.min_evidence_chunks == 6
        assert c.max_retries == 2
        assert c.planner_model == "claude-haiku-4-5"
        assert c.writer_model == "claude-sonnet-4-6"
        assert c.verifier_model == "claude-sonnet-4-6"

    def test_keys_default_to_none(self):
        c = Config()
        assert c.anthropic_api_key is None
        assert c.tavily_api_key is None


class TestEnvOverrides:
    def test_limits_read_from_env(self, monkeypatch):
        monkeypatch.setenv("CITEWISE_MAX_RETRIES", "4")
        monkeypatch.setenv("CITEWISE_MIN_EVIDENCE_CHUNKS", "2")
        c = load_config(use_dotenv=False)
        assert c.max_retries == 4
        assert c.min_evidence_chunks == 2

    def test_api_keys_read_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
        c = load_config(use_dotenv=False)
        assert c.anthropic_api_key == "sk-test"
        assert c.tavily_api_key == "tvly-test"

    def test_models_read_from_env(self, monkeypatch):
        monkeypatch.setenv("CITEWISE_WRITER_MODEL", "claude-opus-5")
        assert load_config(use_dotenv=False).writer_model == "claude-opus-5"


class TestValidation:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_sub_questions", 0),
            ("results_per_question", 0),
            ("min_evidence_chunks", 0),
            ("max_retries", -1),
        ],
    )
    def test_nonsensical_limits_rejected(self, field, value):
        with pytest.raises(ValueError):
            Config(**{field: value})

    def test_max_retries_zero_is_allowed(self):
        """Zero retries is a valid (if strict) cost control — one writer pass only."""
        assert Config(max_retries=0).max_retries == 0

    def test_require_keys_raises_when_absent(self):
        c = Config(anthropic_api_key=None, tavily_api_key=None)
        with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
            c.require_keys()

    def test_require_keys_passes_when_present(self):
        Config(anthropic_api_key="a", tavily_api_key="t").require_keys()
