"""Score the verifier against human labels.

Reads a filled `labeling_sheet.csv` — the `human_verdict` column completed by
hand — and reports how well the verifier agrees.

The headline numbers treat the verifier as a detector of bad claims: a claim is
"flagged" when the verdict is anything other than SUPPORTED. That framing is the
one that matters, because the cost of missing a bad claim (it ships) is not the
same as the cost of flagging a good one (it gets rewritten).

Makes no API calls.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

VERDICTS = ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED")
DEFAULT_SHEET = Path(__file__).resolve().parent / "labeling_sheet.csv"


class LabelingSheetError(ValueError):
    """The sheet is unusable — malformed, unlabelled, or holding a bad verdict."""


def _normalise(value: str) -> str:
    return (value or "").strip().upper()


def read_labeled_rows(path: Path) -> list[dict]:
    """Return only the rows a human has actually labelled.

    Unlabelled rows are skipped rather than counted as agreement — treating a
    blank as a match would inflate every number on the sheet.
    """
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise LabelingSheetError(f"{path} has no rows")

    missing = {"verifier_verdict", "human_verdict"} - set(rows[0])
    if missing:
        raise LabelingSheetError(f"{path} is missing column(s): {', '.join(sorted(missing))}")

    labeled = []
    for i, row in enumerate(rows, start=2):
        human = _normalise(row["human_verdict"])
        if not human:
            continue
        verifier = _normalise(row["verifier_verdict"])
        for name, value in (("human_verdict", human), ("verifier_verdict", verifier)):
            if value not in VERDICTS:
                raise LabelingSheetError(
                    f"{path} line {i}: {name} is {value!r}; expected one of {', '.join(VERDICTS)}"
                )
        labeled.append({"verifier": verifier, "human": human})

    if not labeled:
        raise LabelingSheetError(
            f"{path} has no filled human_verdict values — nothing to score yet"
        )
    return labeled


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }


def score(labeled: list[dict]) -> dict:
    total = len(labeled)
    exact = sum(1 for r in labeled if r["verifier"] == r["human"])

    # Headline: positive class = "flagged", i.e. not SUPPORTED.
    tp = sum(1 for r in labeled if r["verifier"] != "SUPPORTED" and r["human"] != "SUPPORTED")
    fp = sum(1 for r in labeled if r["verifier"] != "SUPPORTED" and r["human"] == "SUPPORTED")
    fn = sum(1 for r in labeled if r["verifier"] == "SUPPORTED" and r["human"] != "SUPPORTED")

    per_class = {}
    for verdict in VERDICTS:
        per_class[verdict] = _prf(
            tp=sum(1 for r in labeled if r["verifier"] == verdict and r["human"] == verdict),
            fp=sum(1 for r in labeled if r["verifier"] == verdict and r["human"] != verdict),
            fn=sum(1 for r in labeled if r["verifier"] != verdict and r["human"] == verdict),
        )

    confusion = Counter((r["human"], r["verifier"]) for r in labeled)

    return {
        "labeled_claims": total,
        "exact_agreement": round(exact / total, 4),
        "flagging": _prf(tp, fp, fn),
        "per_class": per_class,
        "confusion": {f"human={h}|verifier={v}": n for (h, v), n in sorted(confusion.items())},
    }


def format_report(result: dict) -> str:
    flag = result["flagging"]
    lines = [
        "# Verifier vs human labels",
        "",
        f"Labeled claims: {result['labeled_claims']}",
        f"Exact verdict agreement: {result['exact_agreement']:.1%}",
        "",
        "## Catching bad claims (positive = not SUPPORTED)",
        "",
        f"- Precision: {_pct(flag['precision'])}  "
        f"(of claims the verifier flagged, how many a human also rejected)",
        f"- Recall:    {_pct(flag['recall'])}  "
        f"(of claims a human rejected, how many the verifier caught)",
        f"- F1:        {_pct(flag['f1'])}",
        f"- TP {flag['true_positives']} / FP {flag['false_positives']} "
        f"/ FN {flag['false_negatives']}",
        "",
        "## Per verdict",
        "",
        "| Verdict | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for verdict, stats in result["per_class"].items():
        lines.append(
            f"| {verdict} | {_pct(stats['precision'])} | {_pct(stats['recall'])} | "
            f"{_pct(stats['f1'])} | {stats['true_positives']} | "
            f"{stats['false_positives']} | {stats['false_negatives']} |"
        )
    lines += ["", "## Confusion", ""]
    lines += [f"- {key}: {count}" for key, count in result["confusion"].items()]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", nargs="?", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args()

    if not args.sheet.exists():
        print(f"error: {args.sheet} does not exist. Generate it with run_eval.py.", file=sys.stderr)
        return 1

    try:
        result = score(read_labeled_rows(args.sheet))
    except LabelingSheetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
