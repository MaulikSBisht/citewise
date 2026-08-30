"""Sub-questions -> evidence chunks. No LLM.

Rule 6: if retrieval returns too little to work with, abort the run with a
reason. The writer never gets the chance to paper over a gap.
"""

from citewise.config import Config
from citewise.search import EvidenceSearcher
from citewise.state import ResearchState


def make_searcher_node(searcher: EvidenceSearcher, config: Config):
    def search(state: ResearchState) -> dict:
        evidence = searcher.search(state["sub_questions"])

        trace = [
            {
                "node": "searcher",
                "queries": state["sub_questions"],
                "chunks_retrieved": len(evidence),
            }
        ]

        if len(evidence) < config.min_evidence_chunks:
            return {
                "evidence": evidence,
                "aborted_reason": (
                    f"thin evidence: retrieved {len(evidence)} usable chunk(s) across "
                    f"{len(state['sub_questions'])} sub-question(s), below the "
                    f"MIN_EVIDENCE_CHUNKS threshold of {config.min_evidence_chunks}"
                ),
                "trace": trace,
            }

        return {"evidence": evidence, "trace": trace}

    return search
