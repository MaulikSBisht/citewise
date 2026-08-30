"""Phase 4 done-condition: a saved run renders end to end in the UI with no API
key present.

Uses Streamlit's headless AppTest so this is a real render of the real app, not
a test of extracted helpers.
"""

import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
SAVED_RUN = Path(__file__).parent.parent / "fixtures" / "saved_run.json"


@pytest.fixture
def no_keys(monkeypatch, tmp_path):
    """Strip every credential and point the app at a runs dir holding one run."""
    for var in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copy(SAVED_RUN, runs / "20260101T000000000000.json")
    monkeypatch.setenv("CITEWISE_RUNS_DIR", str(runs))
    # The app calls load_config(), which reads the developer's real .env.
    monkeypatch.setenv("CITEWISE_DISABLE_DOTENV", "1")
    return runs


def run_app() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=30)
    app.run()
    return app


class TestReplayWithoutKeys:
    def test_app_renders_without_exception(self, no_keys):
        app = run_app()
        assert not app.exception

    def test_defaults_to_replay_mode_when_no_keys(self, no_keys):
        app = run_app()
        assert app.sidebar.radio[0].value == "Replay saved run"

    def test_saved_run_is_offered_and_selected(self, no_keys):
        app = run_app()
        assert app.selectbox[0].value.name == "20260101T000000000000.json"

    def test_report_is_rendered(self, no_keys):
        app = run_app()
        markdown = " ".join(m.value for m in app.markdown)
        assert "Remote Work and Developer Productivity" in markdown
        assert "8% more code per week" in markdown
        assert "https://example.org/developer-survey-2024" in markdown

    def test_topic_and_retry_rounds_are_shown(self, no_keys):
        app = run_app()
        markdown = " ".join(m.value for m in app.markdown)
        assert "Does remote work increase developer productivity?" in markdown
        assert "Retry rounds:" in markdown

    def test_step_trace_shows_every_node_including_the_retry_round(self, no_keys):
        app = run_app()
        labels = [s.label for s in app.status]
        assert any("Planning sub-questions" in label for label in labels)
        assert any("Searching for evidence" in label for label in labels)
        assert any("Verifying claims (blind)" in label for label in labels)
        assert any("revision round 1" in label for label in labels)
        assert any("Finalising report" in label for label in labels)

    def test_verification_table_has_the_required_columns(self, no_keys):
        app = run_app()
        frames = app.dataframe
        assert len(frames) == 1
        columns = list(frames[0].value.columns)
        assert columns == ["Claim", "Verdict", "Reason", "Sources"]

    def test_verification_table_lists_every_claim_with_a_verdict(self, no_keys):
        app = run_app()
        table = app.dataframe[0].value
        assert len(table) == 2
        assert all("SUPPORTED" in v for v in table["Verdict"])


class TestNewRunGuard:
    def test_new_run_mode_warns_and_disables_the_button_without_keys(self, no_keys):
        app = run_app()
        app.sidebar.radio[0].set_value("New run").run()

        assert not app.exception
        assert any("No API keys found" in w.value for w in app.warning)
        assert app.button[0].disabled


class TestEmptyRunsDirectory:
    def test_no_saved_runs_is_handled(self, monkeypatch, tmp_path):
        for var in ("ANTHROPIC_API_KEY", "TAVILY_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CITEWISE_RUNS_DIR", str(tmp_path / "empty"))
        monkeypatch.setenv("CITEWISE_DISABLE_DOTENV", "1")

        app = run_app()
        assert not app.exception
        assert any("No saved runs yet" in i.value for i in app.info)
