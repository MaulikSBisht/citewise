"""Limits, model names, and credentials.

Every value in section 5 of CLAUDE.md is a cost or safety control. Read them
from a `Config` instance; never hardcode one at a call site.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"


def runs_dir() -> Path:
    """Where runs are written and replayed from. Overridable so the UI and its
    tests can point at a fixture directory."""
    override = os.environ.get("CITEWISE_RUNS_DIR")
    return Path(override) if override else RUNS_DIR


def list_runs() -> list[Path]:
    """Saved runs, newest first."""
    directory = runs_dir()
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"), reverse=True)


class Config(BaseModel):
    max_sub_questions: int = Field(default=5, ge=1)
    results_per_question: int = Field(default=5, ge=1)
    min_evidence_chunks: int = Field(default=6, ge=1)
    max_retries: int = Field(default=2, ge=0)

    planner_model: str = "claude-haiku-4-5"
    writer_model: str = "claude-sonnet-4-6"
    verifier_model: str = "claude-sonnet-4-6"

    max_tokens: int = 16000

    anthropic_api_key: str | None = None
    tavily_api_key: str | None = None

    def require_keys(self) -> None:
        """Fail loudly before spending time on a run that cannot finish."""
        missing = [
            name
            for name, value in (
                ("ANTHROPIC_API_KEY", self.anthropic_api_key),
                ("TAVILY_API_KEY", self.tavily_api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                f"Missing required credential(s): {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def load_config(use_dotenv: bool = True) -> Config:
    """Build a Config from the environment, optionally loading `.env` first.

    Tests pass `use_dotenv=False` so a developer's real `.env` cannot leak into
    a test run. `CITEWISE_DISABLE_DOTENV=1` does the same for code that calls
    `load_config()` with no arguments — the Streamlit app, whose no-key
    behaviour has to be testable on a machine that does have keys.
    """
    if use_dotenv and os.environ.get("CITEWISE_DISABLE_DOTENV") != "1":
        load_dotenv(PROJECT_ROOT / ".env")

    return Config(
        max_sub_questions=_int_env("CITEWISE_MAX_SUB_QUESTIONS", 5),
        results_per_question=_int_env("CITEWISE_RESULTS_PER_QUESTION", 5),
        min_evidence_chunks=_int_env("CITEWISE_MIN_EVIDENCE_CHUNKS", 6),
        max_retries=_int_env("CITEWISE_MAX_RETRIES", 2),
        planner_model=os.environ.get("CITEWISE_PLANNER_MODEL", "claude-haiku-4-5"),
        writer_model=os.environ.get("CITEWISE_WRITER_MODEL", "claude-sonnet-4-6"),
        verifier_model=os.environ.get("CITEWISE_VERIFIER_MODEL", "claude-sonnet-4-6"),
        max_tokens=_int_env("CITEWISE_MAX_TOKENS", 16000),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        tavily_api_key=os.environ.get("TAVILY_API_KEY") or None,
    )
