"""Graph wiring: planner -> searcher -> writer -> verifier -> {writer | finalizer}.

Any node may set `aborted_reason`; the short-circuit edges send the run straight
to END when it does, so a thin-evidence run never reaches the writer (rule 6).

Loop termination: `retry_count` is incremented in exactly one place — the writer,
on re-entry — and `route_after_verify` refuses to loop once it reaches
MAX_RETRIES. That gives at most 1 + MAX_RETRIES writer passes.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from citewise.config import Config, load_config
from citewise.config import runs_dir as config_runs_dir
from citewise.llm import StructuredLLM
from citewise.nodes.finalizer import make_finalizer_node
from citewise.nodes.planner import make_planner_node
from citewise.nodes.searcher import make_searcher_node
from citewise.nodes.verifier import failing_claims, make_verifier_node
from citewise.nodes.writer import make_writer_node
from citewise.search import EvidenceSearcher
from citewise.state import ResearchState, initial_state


def _continue_or_abort(next_node: str):
    """Route to `next_node` unless a previous node aborted the run."""

    def route(state: ResearchState) -> str:
        return END if state.get("aborted_reason") else next_node

    return route


def make_verify_router(config: Config):
    """Loop back to the writer only while there is something to fix and budget to fix it."""

    def route_after_verify(state: ResearchState) -> str:
        if state.get("aborted_reason") or state["draft"] is None:
            return "finalizer"
        if failing_claims(state["draft"]) and state["retry_count"] < config.max_retries:
            return "writer"
        return "finalizer"

    return route_after_verify


def build_graph(
    llm: StructuredLLM,
    searcher: EvidenceSearcher,
    config: Config | None = None,
) -> Any:
    """Compile the graph. Dependencies are injected so tests run on fakes."""
    config = config or Config()

    graph = StateGraph(ResearchState)
    graph.add_node("planner", make_planner_node(llm, config))
    graph.add_node("searcher", make_searcher_node(searcher, config))
    graph.add_node("writer", make_writer_node(llm, config))
    graph.add_node("verifier", make_verifier_node(llm, config))
    graph.add_node("finalizer", make_finalizer_node(config))

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner", _continue_or_abort("searcher"), {"searcher": "searcher", END: END}
    )
    graph.add_conditional_edges(
        "searcher", _continue_or_abort("writer"), {"writer": "writer", END: END}
    )
    graph.add_conditional_edges(
        "writer", _continue_or_abort("verifier"), {"verifier": "verifier", END: END}
    )
    graph.add_conditional_edges(
        "verifier",
        make_verify_router(config),
        {"writer": "writer", "finalizer": "finalizer"},
    )
    graph.add_edge("finalizer", END)

    return graph.compile()


def state_to_dict(state: ResearchState) -> dict:
    """Serialise a finished run, including the full trace."""
    draft = state.get("draft")
    return {
        "topic": state["topic"],
        "sub_questions": state["sub_questions"],
        "evidence": [c.model_dump() for c in state["evidence"]],
        "draft": draft.model_dump() if draft is not None else None,
        "retry_count": state["retry_count"],
        "final_report": state["final_report"],
        "trace": state["trace"],
        "aborted_reason": state["aborted_reason"],
    }


def save_run(state: ResearchState, runs_dir: Path | None = None) -> Path:
    """Persist a run as `runs/<timestamp>.json` so eval and demos can replay it."""
    runs_dir = runs_dir or config_runs_dir()
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    path = runs_dir / f"{stamp}.json"
    path.write_text(
        json.dumps(state_to_dict(state), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def load_run(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(
    topic: str,
    llm: StructuredLLM,
    searcher: EvidenceSearcher,
    config: Config | None = None,
    persist: bool = False,
    runs_dir: Path | None = None,
) -> ResearchState:
    """Drive the graph from a topic to a final state.

    `persist` is opt-in so the test suite does not litter `runs/`.
    """
    config = config or Config()
    final = build_graph(llm, searcher, config).invoke(initial_state(topic))
    if persist:
        save_run(final, runs_dir)
    return final


def stream_run(
    topic: str,
    llm: StructuredLLM,
    searcher: EvidenceSearcher,
    config: Config | None = None,
) -> Iterator[ResearchState]:
    """Yield a full state snapshot after each graph step.

    The UI needs to show progress as it happens; which node just ran is readable
    from the last `trace` entry, so no separate event channel is needed.
    """
    config = config or Config()
    yield from build_graph(llm, searcher, config).stream(initial_state(topic), stream_mode="values")


def build_default_graph() -> Any:
    """Production wiring: real Anthropic and Tavily clients from the environment."""
    config = load_config()
    config.require_keys()
    return build_graph(StructuredLLM(config=config), EvidenceSearcher(config=config), config)
