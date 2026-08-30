"""Tavily wrapper producing `EvidenceChunk`s with stable IDs.

No LLM involved. IDs are assigned in retrieval order (`e1`, `e2`, ...) and are
the only handle the writer and verifier ever use to refer to a source, so they
must stay stable for the life of a run.

The client is constructor-injected, matching `llm.py`.
"""

from typing import Any

from citewise.config import Config
from citewise.state import EvidenceChunk


class EvidenceSearcher:
    """Injectable Tavily wrapper.

    `client` is anything exposing `search(query, max_results=...) -> {"results": [...]}`;
    production passes a `tavily.TavilyClient`, tests pass `FakeTavily`.
    """

    def __init__(self, client: Any = None, config: Config | None = None):
        self.config = config or Config()
        self.client = client if client is not None else self._build_client()

    def _build_client(self) -> Any:
        from tavily import TavilyClient

        if not self.config.tavily_api_key:
            raise RuntimeError(
                "Missing required credential: TAVILY_API_KEY. "
                "Copy .env.example to .env and fill it in."
            )
        return TavilyClient(api_key=self.config.tavily_api_key)

    def search(self, sub_questions: list[str]) -> list[EvidenceChunk]:
        """Query every sub-question, dedupe by URL, and assign sequential IDs.

        Deduping is global across sub-questions: the same source surfacing under
        two questions is one chunk, keeping the sub-question that found it first.
        """
        chunks: list[EvidenceChunk] = []
        seen_urls: set[str] = set()

        for sub_question in sub_questions:
            response = self.client.search(
                query=sub_question, max_results=self.config.results_per_question
            )
            for result in response.get("results", []):
                url = (result.get("url") or "").strip()
                snippet = (result.get("content") or "").strip()
                if not url or not snippet or url in seen_urls:
                    continue
                seen_urls.add(url)
                chunks.append(
                    EvidenceChunk(
                        id=f"e{len(chunks) + 1}",
                        url=url,
                        title=(result.get("title") or url).strip(),
                        snippet=snippet,
                        sub_question=sub_question,
                    )
                )

        return chunks
