"""Run both arms over every topic and record the comparison.

THIS SCRIPT MAKES REAL, PAID API CALLS. It is the only thing in the repo that
does. Everything in `tests/` runs on fakes.

Per topic it runs:
  citewise  — the full graph, with the verification loop
  baseline  — the same evidence, one writer call, no loop

Both arms are scored by the same blind verifier, so the comparison isolates the
loop rather than the verifier.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baseline import run_baseline  # noqa: E402
from citewise.config import Config, load_config  # noqa: E402
from citewise.graph import run, save_run  # noqa: E402
from citewise.llm import StructuredLLM  # noqa: E402
from citewise.nodes.verifier import make_verifier_node  # noqa: E402
from citewise.search import EvidenceSearcher  # noqa: E402
from citewise.state import EvidenceChunk, ReportDraft, initial_state  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
TOPICS_FILE = EVAL_DIR / "topics.yaml"
LABELING_SHEET = EVAL_DIR / "labeling_sheet.csv"

# USD per 1M tokens (input, output).
PRICING = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}

LABELING_COLUMNS = [
    "topic_id",
    "arm",
    "claim_id",
    "claim",
    "cited_evidence",
    "verifier_verdict",
    "verifier_reason",
    "human_verdict",
]


# ---------------------------------------------------------------------------
# Metrics (pure — unit tested without touching the API)
# ---------------------------------------------------------------------------


def cost_usd(usage: list[dict]) -> float:
    """Price a list of per-call usage records. Unknown models price at zero."""
    total = 0.0
    for record in usage:
        rate_in, rate_out = PRICING.get(record["model"], (0.0, 0.0))
        total += record["input_tokens"] / 1_000_000 * rate_in
        total += record["output_tokens"] / 1_000_000 * rate_out
    return round(total, 6)


def score_draft(draft: ReportDraft | None, evidence: list[EvidenceChunk]) -> dict:
    """Verdict and citation metrics for one arm's draft."""
    if draft is None or not draft.claims:
        return {
            "claims": 0,
            "unsupported_claims": 0,
            "contradicted_claims": 0,
            "unsupported_claim_rate": None,
            "contradicted_claim_rate": None,
            "citation_coverage_pct": None,
        }

    pool = {c.id for c in evidence}
    total = len(draft.claims)
    unsupported = sum(1 for c in draft.claims if c.verdict == "UNSUPPORTED")
    contradicted = sum(1 for c in draft.claims if c.verdict == "CONTRADICTED")
    # A claim counts as cited only if at least one of its IDs is really in the
    # pool — an invented ID is not a citation.
    cited = sum(1 for c in draft.claims if any(e in pool for e in c.evidence_ids))

    return {
        "claims": total,
        "unsupported_claims": unsupported,
        "contradicted_claims": contradicted,
        "unsupported_claim_rate": round(unsupported / total, 4),
        "contradicted_claim_rate": round(contradicted / total, 4),
        "citation_coverage_pct": round(100 * cited / total, 2),
    }


def retry_distribution(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("retry_count", 0))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: list[float]) -> float | None:
    usable = [v for v in values if v is not None]
    return round(sum(usable) / len(usable), 4) if usable else None


def aggregate(rows: list[dict], arm: str) -> dict:
    arm_rows = [r for r in rows if r["arm"] == arm and r.get("error") is None]
    if not arm_rows:
        return {"arm": arm, "runs": 0}
    return {
        "arm": arm,
        "runs": len(arm_rows),
        "total_claims": sum(r["claims"] for r in arm_rows),
        "unsupported_claim_rate": _mean([r["unsupported_claim_rate"] for r in arm_rows]),
        "contradicted_claims": sum(r["contradicted_claims"] for r in arm_rows),
        "citation_coverage_pct": _mean([r["citation_coverage_pct"] for r in arm_rows]),
        "mean_latency_s": _mean([r["latency_s"] for r in arm_rows]),
        "total_cost_usd": round(sum(r["cost_usd"] for r in arm_rows), 4),
        "retry_distribution": retry_distribution(arm_rows) if arm == "citewise" else None,
    }


def _fmt(value, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def markdown_table(summary: list[dict]) -> str:
    header = (
        "| Arm | Runs | Claims | Unsupported rate | Contradicted | "
        "Citation coverage | Mean latency | Cost |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    lines = [header]
    for row in summary:
        if not row.get("runs"):
            lines.append(f"| {row['arm']} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {row['arm']} | {row['runs']} | {row['total_claims']} | "
            f"{_fmt(row['unsupported_claim_rate'])} | {row['contradicted_claims']} | "
            f"{_fmt(row['citation_coverage_pct'], '%')} | "
            f"{_fmt(row['mean_latency_s'], 's')} | ${row['total_cost_usd']} |"
        )

    citewise = next((r for r in summary if r["arm"] == "citewise"), None)
    if citewise and citewise.get("retry_distribution"):
        lines.append("")
        lines.append(
            "Retry rounds used (citewise): "
            + ", ".join(
                f"{k} round(s): {v} run(s)" for k, v in citewise["retry_distribution"].items()
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_topics(path: Path = TOPICS_FILE) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["topics"]


def verify_once(draft: ReportDraft, evidence: list[EvidenceChunk], llm, config) -> ReportDraft:
    """Score the baseline with the same blind verifier, one pass, no feedback."""
    node = make_verifier_node(llm, config)
    state = initial_state("") | {"draft": draft, "evidence": evidence}
    return node(state)["draft"]


def evaluate_topic(entry: dict, config: Config, runs_dir: Path | None) -> list[dict]:
    topic = entry["topic"]
    rows: list[dict] = []

    # --- citewise arm ---
    llm = StructuredLLM(config=config)
    searcher = EvidenceSearcher(config=config)
    started = time.perf_counter()
    try:
        final = run(topic, llm, searcher, config)
        latency = time.perf_counter() - started
        save_run(final, runs_dir)
        row = {
            "topic_id": entry["id"],
            "kind": entry["kind"],
            "topic": topic,
            "arm": "citewise",
            "retry_count": final["retry_count"],
            "aborted_reason": final["aborted_reason"],
            "latency_s": round(latency, 2),
            "cost_usd": cost_usd(llm.usage),
            "tokens": llm.usage_totals(),
            "error": None,
            **score_draft(final["draft"], final["evidence"]),
        }
        rows.append(row)
        evidence = final["evidence"]
    except Exception as exc:  # noqa: BLE001 — one topic failing must not kill the eval
        rows.append(
            {
                "topic_id": entry["id"],
                "kind": entry["kind"],
                "topic": topic,
                "arm": "citewise",
                "error": str(exc),
                "latency_s": None,
                "cost_usd": cost_usd(llm.usage),
                "retry_count": 0,
                **score_draft(None, []),
            }
        )
        return rows

    if not evidence:
        return rows

    # --- baseline arm, over the identical evidence pool ---
    baseline_llm = StructuredLLM(config=config)
    started = time.perf_counter()
    try:
        draft = run_baseline(topic, evidence, baseline_llm, config)
        verified = verify_once(draft, evidence, baseline_llm, config)
        latency = time.perf_counter() - started
        rows.append(
            {
                "topic_id": entry["id"],
                "kind": entry["kind"],
                "topic": topic,
                "arm": "baseline",
                "retry_count": 0,
                "aborted_reason": None,
                "latency_s": round(latency, 2),
                "cost_usd": cost_usd(baseline_llm.usage),
                "tokens": baseline_llm.usage_totals(),
                "error": None,
                "draft": verified.model_dump(),
                **score_draft(verified, evidence),
            }
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            {
                "topic_id": entry["id"],
                "kind": entry["kind"],
                "topic": topic,
                "arm": "baseline",
                "error": str(exc),
                "latency_s": None,
                "cost_usd": cost_usd(baseline_llm.usage),
                "retry_count": 0,
                **score_draft(None, []),
            }
        )

    return rows


# ---------------------------------------------------------------------------
# Labeling sheet
# ---------------------------------------------------------------------------


def build_labeling_rows(samples: list[dict], target: int = 60, seed: int = 0) -> list[dict]:
    """Sample claims for human labelling, leaving `human_verdict` blank.

    `samples` entries are {topic_id, arm, draft, evidence}. Sampling is seeded so
    regenerating the sheet does not reshuffle rows a human has already filled in.
    """
    pool: list[dict] = []
    for sample in samples:
        by_id = {c["id"]: c for c in sample["evidence"]}
        for claim in sample["draft"]["claims"]:
            cited = "\n".join(
                f"[{eid}] {by_id[eid]['snippet']}" for eid in claim["evidence_ids"] if eid in by_id
            )
            pool.append(
                {
                    "topic_id": sample["topic_id"],
                    "arm": sample["arm"],
                    "claim_id": claim["id"],
                    "claim": claim["text"],
                    "cited_evidence": cited or "(no cited evidence in pool)",
                    "verifier_verdict": claim["verdict"],
                    "verifier_reason": claim.get("verdict_reason") or "",
                    "human_verdict": "",
                }
            )

    if len(pool) > target:
        pool = random.Random(seed).sample(pool, target)
    return pool


def write_labeling_sheet(rows: list[dict], path: Path = LABELING_SHEET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABELING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=Path, default=TOPICS_FILE)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N topics")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    config = load_config()
    try:
        config.require_keys()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("run_eval.py is the only script that calls the real API.", file=sys.stderr)
        return 1

    topics = load_topics(args.topics)[: args.limit]
    print(f"Running {len(topics)} topic(s) x 2 arms against the live API. This costs money.")

    rows: list[dict] = []
    labeling_samples: list[dict] = []
    for entry in topics:
        print(f"  {entry['id']}: {entry['topic']}")
        topic_rows = evaluate_topic(entry, config, None)
        rows.extend(topic_rows)
        if entry.get("labeling_sample"):
            for row in topic_rows:
                draft = row.pop("draft", None)
                if draft:
                    labeling_samples.append(
                        {
                            "topic_id": entry["id"],
                            "arm": row["arm"],
                            "draft": draft,
                            "evidence": [],
                        }
                    )

    summary = [aggregate(rows, "citewise"), aggregate(rows, "baseline")]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(
        json.dumps({"runs": rows, "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    table = markdown_table(summary)
    (args.out / "results.md").write_text(table + "\n", encoding="utf-8")

    if labeling_samples:
        sheet = write_labeling_sheet(build_labeling_rows(labeling_samples))
        print(f"Wrote labeling sheet: {sheet}")

    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
