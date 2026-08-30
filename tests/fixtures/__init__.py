"""Test doubles and canned payloads.

Rule 2: no test touches the network. Every Anthropic and Tavily interaction in
`tests/` goes through the fakes defined here, driven by the canned JSON
payloads next to this file.
"""

import json
from pathlib import Path

FIXTURE_DIR = Path(__file__).parent

ANTHROPIC = json.loads((FIXTURE_DIR / "anthropic_responses.json").read_text(encoding="utf-8"))
TAVILY = json.loads((FIXTURE_DIR / "tavily_responses.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Anthropic fake
# --------------------------------------------------------------------------


class FakeTextBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class FakeUsage:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class FakeMessage:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.content = [FakeTextBlock(text)] if text is not None else []
        self.stop_reason = stop_reason
        self.usage = FakeUsage()
        self.model = "fake-model"


class _FakeMessages:
    def __init__(self, owner: "FakeAnthropic"):
        self._owner = owner

    def create(self, **kwargs):
        self._owner.calls.append(kwargs)
        if not self._owner.queue:
            raise AssertionError(
                f"FakeAnthropic ran out of queued responses after {len(self._owner.calls)} call(s)"
            )
        item = self._owner.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return FakeMessage(item)
        return item


class FakeAnthropic:
    """Stands in for `anthropic.Anthropic`.

    Queue entries may be a raw string (becomes the response text), a
    `FakeMessage`, or an exception instance to raise.
    """

    def __init__(self, responses):
        self.queue = list(responses)
        self.calls: list[dict] = []
        self.messages = _FakeMessages(self)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def canned(name: str) -> str:
    """Look up a canned Anthropic response body by fixture name."""
    if name not in ANTHROPIC:
        raise KeyError(f"unknown Anthropic fixture {name!r}")
    return ANTHROPIC[name]


# --------------------------------------------------------------------------
# Tavily fake
# --------------------------------------------------------------------------


class FakeTavily:
    """Stands in for `tavily.TavilyClient`.

    `payloads` is either a single payload dict reused for every query, or a list
    consumed one query at a time.
    """

    def __init__(self, payloads=None, per_query: dict | None = None):
        self._payloads = payloads
        self._per_query = per_query or {}
        self.queries: list[str] = []
        self.calls: list[dict] = []

    def search(self, query: str, max_results: int = 5, **kwargs) -> dict:
        self.queries.append(query)
        self.calls.append({"query": query, "max_results": max_results, **kwargs})
        if self._per_query:
            payload = self._per_query.get(query, {"results": []})
        elif isinstance(self._payloads, list):
            payload = self._payloads.pop(0) if self._payloads else {"results": []}
        else:
            payload = self._payloads or {"results": []}
        return {"results": payload["results"][:max_results]}


def tavily_payload(name: str) -> dict:
    if name not in TAVILY:
        raise KeyError(f"unknown Tavily fixture {name!r}")
    return TAVILY[name]


def partitioned(name: str, n_queries: int) -> list[dict]:
    """Split a canned payload round-robin across `n_queries` payloads.

    Distinct sub-questions retrieve mostly distinct sources in reality. Handing
    every query the same payload instead makes the searcher's URL dedupe eat all
    but the first query's results, which is correct behaviour but a poor stand-in
    for a real run.
    """
    results = TAVILY[name]["results"]
    buckets: list[list[dict]] = [[] for _ in range(n_queries)]
    for i, result in enumerate(results):
        buckets[i % n_queries].append(result)
    return [{"results": bucket} for bucket in buckets]
