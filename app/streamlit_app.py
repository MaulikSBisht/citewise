"""citewise demo UI.

This is a demo surface, not the project. It reads from `src/citewise` and holds
no reasoning of its own — if something here starts making decisions about claims
or evidence, it belongs in `src/`.

Replay mode renders a saved run from `runs/` and needs no API key, which is what
makes the demo showable without spending anything.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from citewise.config import list_runs, load_config  # noqa: E402
from citewise.graph import load_run, save_run, stream_run  # noqa: E402
from citewise.llm import StructuredLLM  # noqa: E402
from citewise.search import EvidenceSearcher  # noqa: E402
from citewise.state import EvidenceChunk, ReportDraft  # noqa: E402

STEP_LABELS = {
    "planner": "Planning sub-questions",
    "searcher": "Searching for evidence",
    "writer": "Writing claims",
    "verifier": "Verifying claims (blind)",
    "finalizer": "Finalising report",
}

VERDICT_ICON = {
    "SUPPORTED": "✅",
    "UNSUPPORTED": "⚠️",
    "CONTRADICTED": "❌",
    "PENDING": "…",
}

st.set_page_config(page_title="citewise", page_icon="🔍", layout="wide")


def describe_step(entry: dict) -> str:
    """One-line summary of a trace entry, including which retry round it was."""
    node = entry.get("node", "?")
    label = STEP_LABELS.get(node, node)

    if node == "planner":
        return f"{label} — {len(entry.get('sub_questions', []))} sub-questions"
    if node == "searcher":
        return f"{label} — {entry.get('chunks_retrieved', 0)} chunks"
    if node == "writer":
        if entry.get("mode") == "revision":
            return (
                f"{label} — revision round {entry.get('round', '?')}, "
                f"revised {len(entry.get('revised', []))}, "
                f"froze {len(entry.get('frozen', []))}"
            )
        return f"{label} — {entry.get('claims', 0)} claims in {entry.get('sections', 0)} sections"
    if node == "verifier":
        results = entry.get("results", [])
        supported = sum(1 for r in results if r["verdict"] == "SUPPORTED")
        return f"{label} — round {entry.get('round', 0)}: {supported}/{len(results)} supported"
    if node == "finalizer":
        return (
            f"{label} — shipped {entry.get('claims_shipped', 0)}, "
            f"dropped {len(entry.get('dropped_contradicted', []))}, "
            f"flagged {len(entry.get('kept_unverified', []))}"
        )
    return label


def render_trace(trace: list[dict]) -> None:
    for entry in trace:
        with st.status(describe_step(entry), state="complete"):
            if entry.get("node") == "planner":
                for question in entry.get("sub_questions", []):
                    st.write(f"- {question}")
            elif entry.get("node") == "verifier":
                for result in entry.get("results", []):
                    icon = VERDICT_ICON.get(result["verdict"], "")
                    st.write(f"{icon} **{result['claim_id']}** — {result['reason']}")
            else:
                st.json(entry, expanded=False)


def verification_rows(draft: ReportDraft, evidence: list[EvidenceChunk]) -> list[dict]:
    by_id = {c.id: c for c in evidence}
    rows = []
    for claim in draft.claims:
        sources = "\n".join(
            f"[{eid}] {by_id[eid].url}" for eid in claim.evidence_ids if eid in by_id
        )
        rows.append(
            {
                "Claim": claim.text,
                "Verdict": f"{VERDICT_ICON.get(claim.verdict, '')} {claim.verdict}",
                "Reason": claim.verdict_reason or "—",
                "Sources": sources,
            }
        )
    return rows


def show_results(
    draft: ReportDraft | None,
    evidence: list[EvidenceChunk],
    final_report: str | None,
    aborted_reason: str | None,
) -> None:
    if aborted_reason:
        st.error(f"Run aborted: {aborted_reason}")

    if final_report:
        st.subheader("Report")
        st.markdown(final_report)

    if draft is not None and draft.claims:
        with st.expander("Verification detail", expanded=False):
            st.dataframe(
                verification_rows(draft, evidence),
                width="stretch",
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("🔍 citewise")
st.caption("Research reports as claims bound to evidence, with every claim blind-verified.")

config = load_config()
has_keys = bool(config.anthropic_api_key and config.tavily_api_key)
saved_runs = list_runs()

mode = st.sidebar.radio(
    "Mode",
    ["New run", "Replay saved run"],
    index=0 if has_keys else 1,
    help="Replay needs no API key.",
)

st.sidebar.markdown("---")
st.sidebar.write(f"**Retries allowed:** {config.max_retries}")
st.sidebar.write(f"**Min evidence chunks:** {config.min_evidence_chunks}")
st.sidebar.write(f"**Writer / verifier:** `{config.writer_model}`")
st.sidebar.write(f"**Planner:** `{config.planner_model}`")

if mode == "New run":
    if not has_keys:
        st.warning(
            "No API keys found. Copy `.env.example` to `.env` and fill in "
            "`ANTHROPIC_API_KEY` and `TAVILY_API_KEY`, or switch to **Replay saved run** "
            "in the sidebar to browse an existing run without spending anything."
        )

    topic = st.text_input(
        "Research topic",
        placeholder="Does remote work increase developer productivity?",
    )
    if st.button("Run", type="primary", disabled=not (has_keys and topic.strip())):
        llm = StructuredLLM(config=config)
        searcher = EvidenceSearcher(config=config)

        seen = 0
        final_state = None
        progress = st.container()
        for snapshot in stream_run(topic.strip(), llm, searcher, config):
            final_state = snapshot
            trace = snapshot.get("trace", [])
            with progress:
                for entry in trace[seen:]:
                    with st.status(describe_step(entry), state="complete"):
                        st.json(entry, expanded=False)
            seen = len(trace)

        if final_state is not None:
            save_run(final_state)
            show_results(
                final_state.get("draft"),
                final_state.get("evidence", []),
                final_state.get("final_report"),
                final_state.get("aborted_reason"),
            )

else:
    if not saved_runs:
        st.info("No saved runs yet. Runs are written to `runs/` as they complete.")
    else:
        choice = st.selectbox("Saved run", saved_runs, format_func=lambda p: p.name)
        payload = load_run(choice)

        st.write(f"**Topic:** {payload['topic']}")
        st.write(f"**Retry rounds:** {payload['retry_count']}")

        render_trace(payload.get("trace", []))

        draft = (
            ReportDraft.model_validate(payload["draft"])
            if payload.get("draft") is not None
            else None
        )
        evidence = [EvidenceChunk.model_validate(c) for c in payload.get("evidence", [])]
        show_results(draft, evidence, payload.get("final_report"), payload.get("aborted_reason"))
