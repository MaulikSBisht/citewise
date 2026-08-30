"""Topic -> sub-questions. Haiku, structured output.

The sub-questions are what the searcher turns into queries, so they need to be
independently searchable rather than facets of one broad question.
"""

from pydantic import BaseModel, Field

from citewise.config import Config
from citewise.llm import StructuredLLM
from citewise.state import ResearchState

MIN_SUB_QUESTIONS = 3

PLANNER_SYSTEM = """You are a research planner. Given a topic, you break it into \
independently searchable sub-questions.

Rules:
- Produce between {min_q} and {max_q} sub-questions.
- Each must be answerable from web sources on its own, without the others.
- Each must be a specific factual question, not a theme or a heading.
- Together they should cover the topic, including any points on which credible \
sources are likely to disagree.
- Do not ask for opinions, predictions, or recommendations."""


class PlannerOutput(BaseModel):
    sub_questions: list[str] = Field(min_length=1)


def make_planner_node(llm: StructuredLLM, config: Config):
    """Build the planner node. Dependencies are injected, matching the wrappers."""

    def planner(state: ResearchState) -> dict:
        result = llm.complete_structured(
            system=PLANNER_SYSTEM.format(min_q=MIN_SUB_QUESTIONS, max_q=config.max_sub_questions),
            user=f"Topic: {state['topic']}",
            schema=PlannerOutput,
            model=config.planner_model,
        )

        questions = [q.strip() for q in result.sub_questions if q and q.strip()]
        # MAX_SUB_QUESTIONS is a cost control: each extra question is another
        # Tavily call and more evidence for the writer to read. Enforce it here
        # rather than trusting the prompt.
        questions = questions[: config.max_sub_questions]

        if not questions:
            return {
                "sub_questions": [],
                "aborted_reason": "planner returned no usable sub-questions",
                "trace": [{"node": "planner", "sub_questions": []}],
            }

        return {
            "sub_questions": questions,
            "trace": [{"node": "planner", "sub_questions": questions}],
        }

    return planner
