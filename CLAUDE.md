# CLAUDE.md — citewise build instructions

You are building **citewise**, a multi-agent research report writer with a
claim-verification loop. Read this entire file before writing any code.

---

## 0. What this project is

Given a research topic, citewise:

1. plans sub-questions,
2. searches the web for evidence,
3. writes a report as **structured claims bound to evidence**, not prose,
4. **blind-verifies every claim** against the evidence that supposedly supports it,
5. loops back to the writer with the specific failures, up to twice,
6. renders the surviving claims to a cited markdown report.

The verification loop is the entire point of the project. Everything else is
scaffolding around it. Do not simplify it away.

---

## 1. Hard rules

These are non-negotiable. Violating any of them means the build is wrong even
if the tests pass.

1. **Write tests before implementation** for every module. Then run them, and
   fix until green. Do not declare a phase done on unrun tests.
2. **No real API calls in `tests/`.** Anthropic and Tavily are fully mocked with
   canned fixtures. Real keys are used only by `eval/run_eval.py`.
3. **The writer never emits prose.** It emits a `ReportDraft` object. Markdown
   exists only in `render.py`, at the very end.
4. **The verifier is blind.** It receives a claim and evidence chunks. It never
   receives the draft, the writer's reasoning, the other claims, or the framing
   of the topic. This isolation is what makes the verification meaningful —
   preserve it exactly.
5. **Every claim carries `evidence_ids`.** A claim with an empty `evidence_ids`
   list is invalid and must be rejected at the schema level, not tolerated.
6. **Fail loudly on thin evidence.** If retrieval returns too little to work
   with, abort the run with a reason. Never let the writer paper over a gap.
7. `make validate` (lint + full test suite) is the done gate for every phase.

---

## 2. Phases

Each phase has a done-condition. Do not start the next phase until the current
one's condition holds. There is no time budget — correctness of the gate is
what matters.

### Phase 0 — Repo hygiene

Do this first, before anything else.

- Create `.gitignore` containing at minimum: `.env`, `__pycache__/`, `*.pyc`,
  `runs/`, `eval/results/`, `.venv/`, `.pytest_cache/`.
- **The user's `.env` already exists with live API keys and is not yet ignored.**
  Getting it into `.gitignore` before the first commit is the priority.
- Create `.env.example` with empty placeholder keys (`ANTHROPIC_API_KEY=`,
  `TAVILY_API_KEY=`) and commit that instead.
- Create `requirements.txt`, `Makefile` (targets: `install`, `test`, `lint`,
  `validate`, `run`, `eval`), and the directory skeleton from section 3.
- `git init` and make the first commit only after `.gitignore` is in place.

**Done when:** `git status` never shows `.env`, and `make install` succeeds.

### Phase 1 — Contracts and fakes

Build the data model and the test doubles before any node logic.

- `src/citewise/state.py` — all pydantic schemas (section 4).
- `src/citewise/config.py` — limits and model names (section 5), loading `.env`
  via `python-dotenv`.
- `src/citewise/llm.py` — Anthropic wrapper exposing
  `complete_structured(system, user, schema, model) -> BaseModel`. On a JSON
  parse or schema-validation failure, retry **once** with the validation error
  fed back to the model, then raise.
- `src/citewise/search.py` — Tavily wrapper returning `list[EvidenceChunk]`
  with stable IDs (`e1`, `e2`, ...).
- `tests/fixtures/` — canned Anthropic and Tavily responses, including at least
  one malformed JSON response to exercise the repair path.
- Both wrappers must be injectable so tests can swap in fakes. Constructor
  injection or a module-level factory both fine; pick one and be consistent.

**Done when:** schema tests pass, the JSON-repair path is covered by a test,
and no test touches the network.

### Phase 2 — Forward path

- `nodes/planner.py` — topic → 3–5 sub-questions. Haiku. Structured output.
- `nodes/searcher.py` — no LLM. For each sub-question, call Tavily, dedupe by
  URL, assign IDs, return chunks. Enforce `MIN_EVIDENCE_CHUNKS`; below that,
  set `aborted_reason` and short-circuit the graph.
- `nodes/writer.py` — evidence → `ReportDraft`. Sonnet. The prompt must state
  that every claim needs at least one evidence ID drawn from the supplied pool,
  and that claims must be atomic — one assertion each, not compound sentences.
  Compound claims are the main thing that makes verification mushy, so push on
  this in the prompt and reject drafts whose claims cite unknown IDs.
- `graph.py` — wire planner → searcher → writer, no loop yet.

**Done when:** an integration test drives the graph from topic to `ReportDraft`
using only fakes, and a draft citing a nonexistent evidence ID is rejected.

### Phase 3 — The verification loop

This is the core. Take your time here.

- `nodes/verifier.py` — for each claim, build a prompt containing **only** the
  claim text and the full text of its cited evidence chunks, plus the wider
  evidence pool for contradiction detection. Return
  `SUPPORTED | UNSUPPORTED | CONTRADICTED` with a one-sentence reason.
  Verify claims independently — one call per claim, or batched, but never in a
  way that lets one claim's verdict influence another's.
- `graph.py` — conditional edge after the verifier:
  - any claim not `SUPPORTED` **and** `retry_count < MAX_RETRIES` → back to
    `writer`, with the failing claims and their reasons as feedback;
  - otherwise → `finalizer`.
  - Increment `retry_count` on every loop. An infinite loop here is a bug, and
    a test must prove it terminates.
- On re-entry, the writer revises **only** the failing claims. Supported claims
  are frozen and passed through untouched — do not regenerate the whole draft.
- `nodes/finalizer.py` — drop claims still `CONTRADICTED`; keep claims still
  `UNSUPPORTED` but mark them explicitly as unverified in the output. Never
  silently ship a failed claim.
- `render.py` — `ReportDraft` → markdown, with numbered citations resolving to
  source URLs and a footer listing unverified claims, if any.
- Persist every run as JSON in `runs/<timestamp>.json`, including the full
  trace. Eval and demos read from these instead of re-calling the API.

**Done when:** a test with a deliberately hallucinated claim in the fixture
shows the loop catching it, feeding it back, and either fixing or flagging it —
and a test proves the loop terminates at `MAX_RETRIES`.

### Phase 4 — Streamlit

- `app/streamlit_app.py`: topic input, run button, live step trace via
  `st.status` containers (planner → searcher → writer → verifier, showing retry
  rounds), final rendered report, and an expandable verification table with
  columns claim / verdict / reason / sources.
- The UI is a demo surface, not the project. Keep it one file. Do not let it
  grow logic that belongs in `src/`.

**Done when:** a saved run from `runs/` renders end to end in the UI with no
API key present.

### Phase 5 — Eval

- `eval/topics.yaml` — 10 topics, mixed: roughly half clean factual subjects,
  half contested or ambiguous ones where sources disagree. The contested ones
  are where the verifier earns its keep.
- `eval/baseline.py` — same evidence, one Sonnet call, "write a cited report",
  no loop. This is what citewise is measured against.
- `eval/run_eval.py` — run both arms over all topics and record per run:
  unsupported claim rate, citation coverage %, contradicted claim count, retry
  distribution, wall-clock latency, token cost.
- Write `eval/results/results.json` plus a markdown summary table.
- **Verifier ground truth:** produce `eval/labeling_sheet.csv` — roughly 60
  claims sampled across 3 topics, with columns for the claim, its evidence, the
  verifier's verdict, and a blank human verdict column. Add a script that reads
  the filled sheet and computes the verifier's precision and recall against the
  human labels. The user fills this in by hand; you build the tooling and leave
  the human column empty.

**Done when:** the results table is generated from real runs, and the labeling
sheet + scoring script exist and are tested against a small synthetic filled
sheet.

### Phase 6 — Documentation

- `README.md`: what it is, the architecture diagram (mermaid), quickstart, a
  sample report excerpt, and the eval table with real numbers.
- Lead the README with the verification loop and the eval results. That is the
  interesting part; retrieval and the UI are commodity.

---

## 3. Layout

```
citewise/
  src/citewise/
    __init__.py  state.py  config.py  llm.py  search.py  render.py  graph.py
    nodes/  __init__.py planner.py searcher.py writer.py verifier.py finalizer.py
  app/streamlit_app.py
  eval/  topics.yaml  baseline.py  run_eval.py  score_labels.py  results/
  tests/  fixtures/  unit/  integration/
  runs/
  .env  .env.example  .gitignore  requirements.txt  Makefile  README.md
```

Keep every file under roughly 200 lines. If a node file is growing past that,
the prompt construction probably wants to move into its own module.

---

## 4. Schemas (`state.py`)

```python
class EvidenceChunk(BaseModel):
    id: str                 # "e1", "e2", ...
    url: str
    title: str
    snippet: str
    sub_question: str

class Claim(BaseModel):
    id: str                 # "c1", "c2", ...
    text: str               # atomic — one assertion
    evidence_ids: list[str] # must be non-empty
    verdict: Literal["PENDING", "SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
    verdict_reason: str | None

class Section(BaseModel):
    heading: str
    claim_ids: list[str]

class ReportDraft(BaseModel):
    title: str
    sections: list[Section]
    claims: list[Claim]

class ResearchState(TypedDict):
    topic: str
    sub_questions: list[str]
    evidence: list[EvidenceChunk]
    draft: ReportDraft | None
    retry_count: int
    final_report: str | None
    trace: list[dict]
    aborted_reason: str | None
```

Validate at the boundary: `evidence_ids` non-empty, every ID present in the
evidence pool, every `claim_id` in a section resolving to a real claim.

---

## 5. Config defaults

```
MAX_SUB_QUESTIONS   = 5
RESULTS_PER_QUESTION = 5
MIN_EVIDENCE_CHUNKS = 6
MAX_RETRIES         = 2
PLANNER_MODEL       = claude-haiku-4-5
WRITER_MODEL        = claude-sonnet-4-6
VERIFIER_MODEL      = claude-sonnet-4-6
```

Every one of these is a cost or safety control. Read them from config, never
hardcode them at a call site.

---

## 6. Test tiers

- **unit** — schemas, render, config, the JSON-repair path, prompt builders.
- **integration** — full graph runs on fakes: happy path, hallucinated claim
  caught, thin-evidence abort, retry exhaustion.
- **eval-regression** — replay a saved run from `runs/` and assert the rendered
  output is stable.

Run `make validate` before declaring any phase complete. If something fails,
fix it rather than adjusting the test to match the bug.
