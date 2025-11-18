"""Lightweight wrapper graph for exposing Deep Research over MCP."""

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda

from open_deep_research.deep_researcher import deep_researcher


class DeepResearchInput(BaseModel):
    """Schema for MCP callers."""

    topic: str = Field(
        ...,
        description="Describe the subject you want fully researched. Provide enough detail for high-quality retrieval.",
    )


class DeepResearchOutput(BaseModel):
    """Structured output returned to MCP callers."""

    report: str = Field(..., description="Final research report in markdown.")
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="Full message trace (including intermediate reasoning) for auditing.",
    )


def _prepare_input(data: DeepResearchInput) -> dict:
    """Convert topic string into the AgentInputState expected by the main graph."""
    topic = data.topic.strip()
    if not topic:
        raise ValueError("The 'topic' argument must be a non-empty string.")
    return {"messages": [HumanMessage(content=topic)]}


def _format_output(result: dict) -> DeepResearchOutput:
    """Select the relevant fields for MCP consumers."""
    return DeepResearchOutput(
        report=result.get("final_report", ""),
        messages=result.get("messages", []),
    )


deep_research_topic_graph = (
    RunnableLambda(_prepare_input)
    | deep_researcher
    | RunnableLambda(_format_output)
).with_types(input_type=DeepResearchInput, output_type=DeepResearchOutput)
