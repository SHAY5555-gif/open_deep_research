"""Graph wrapper that exposes Deep Research as a single-topic MCP tool."""

from typing import List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from open_deep_research.configuration import Configuration
from open_deep_research.deep_researcher import AgentInputState, deep_researcher


class TopicInput(BaseModel):
    """Schema exposed to MCP callers."""

    topic: str = Field(
        ...,
        description="Describe the subject you want fully researched. Provide enough detail for high-quality retrieval.",
    )


class TopicState(TypedDict):
    """Internal graph state used by the wrapper graph."""

    topic: str
    report: str
    messages: List[BaseMessage]


class TopicOutput(BaseModel):
    """Structured response returned to Smithery clients."""

    report: str
    messages: List[BaseMessage]


topic_graph_builder = StateGraph(
    TopicState,
    input=TopicInput,
    output=TopicOutput,
    config_schema=Configuration,
)


async def run_deep_research(state: TopicState, config: RunnableConfig) -> TopicState:
    """Invoke the original Deep Research graph using the supplied topic."""

    topic = state["topic"].strip()
    if not topic:
        raise ValueError("The 'topic' argument must be a non-empty string.")

    result = await deep_researcher.ainvoke(
        AgentInputState(messages=[HumanMessage(content=topic)]),
        config=config,
    )
    return {
        "topic": topic,
        "report": result.get("final_report", ""),
        "messages": result.get("messages", []),
    }


topic_graph_builder.add_node("run_research", run_deep_research)
topic_graph_builder.add_edge(START, "run_research")
topic_graph_builder.add_edge("run_research", END)

deep_research_topic_graph = topic_graph_builder.compile()
