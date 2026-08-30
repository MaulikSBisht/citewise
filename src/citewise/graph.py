"""Graph wiring: planner -> searcher -> writer.

Phase 2 is the forward path only. The verification loop is added in Phase 3.

Any node may set `aborted_reason`; the short-circuit edges below send the run
straight to END when it does, so a thin-evidence run never reaches the writer
(rule 6).
"""

from typing import Any

from langgraph.graph import END, StateGraph

from citewise.config import Config, load_config
from citewise.llm import StructuredLLM
from citewise.nodes.planner import make_planner_node
from citewise.nodes.searcher import make_searcher_node
from citewise.nodes.writer import make_writer_node
from citewise.search import EvidenceSearcher
from citewise.state import ResearchState, initial_state


def _continue_or_abort(next_node: str):
    """Route to `next_node` unless a previous node aborted the run."""

    def route(state: ResearchState) -> str:
        return END if state.get("aborted_reason") else next_node

    return route


def build_graph(
    llm: StructuredLLM,
    searcher: EvidenceSearcher,
    config: Config | None = None,
) -> Any:
    """Compile the forward-path graph. Dependencies are injected so tests run on fakes."""
    config = config or Config()

    graph = StateGraph(ResearchState)
    graph.add_node("planner", make_planner_node(llm, config))
    graph.add_node("searcher", make_searcher_node(searcher, config))
    graph.add_node("writer", make_writer_node(llm, config))

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner", _continue_or_abort("searcher"), {"searcher": "searcher", END: END}
    )
    graph.add_conditional_edges(
        "searcher", _continue_or_abort("writer"), {"writer": "writer", END: END}
    )
    graph.add_edge("writer", END)

    return graph.compile()


def run(topic: str, llm: StructuredLLM, searcher: EvidenceSearcher, config: Config | None = None):
    """Drive the graph from a topic to a final state."""
    config = config or Config()
    return build_graph(llm, searcher, config).invoke(initial_state(topic))


def build_default_graph() -> Any:
    """Production wiring: real Anthropic and Tavily clients from the environment."""
    config = load_config()
    config.require_keys()
    return build_graph(StructuredLLM(config=config), EvidenceSearcher(config=config), config)
