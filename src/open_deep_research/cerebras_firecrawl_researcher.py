"""
Cerebras-powered Deep Research Agent with Firecrawl SDK.

This agent uses Cerebras DIRECTLY (not via OpenRouter) for ULTRA-FAST inference
combined with Firecrawl SDK for comprehensive web scraping and content extraction.

Designed to run as a local MCP server via STDIO transport.
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from firecrawl import AsyncFirecrawl
from typing import Literal

from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)

logger = logging.getLogger(__name__)


# ============================================
# Cerebras Direct Model Initialization
# Using OpenAI-compatible API for tool calling support
# ============================================

from langchain_openai import ChatOpenAI


def get_cerebras_model(
    max_tokens: int = None,
    temperature: float = 0.7,
):
    """Initialize Cerebras using OpenAI-compatible API for tool calling support.

    Cerebras provides an OpenAI-compatible endpoint that supports tool calling,
    which ChatCerebras doesn't support natively.

    Args:
        max_tokens: Maximum tokens for model output
        temperature: Temperature for generation (default: 0.7)

    Returns:
        ChatOpenAI instance configured for Cerebras
    """
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError(
            "CEREBRAS_API_KEY environment variable not set. "
            "Get your API key from https://cloud.cerebras.ai"
        )

    model_name = os.getenv("CEREBRAS_MODEL", "zai-glm-4.6")

    model_params = {
        "model": model_name,
        "api_key": api_key,
        "base_url": "https://api.cerebras.ai/v1",  # Cerebras OpenAI-compatible endpoint
        "temperature": temperature,
    }

    if max_tokens:
        model_params["max_tokens"] = max_tokens

    logger.info(f"Initializing Cerebras (OpenAI-compatible) with model: {model_name}")

    return ChatOpenAI(**model_params)


# ============================================
# FireCrawl SDK Tools
# ============================================

_firecrawl_client = None


def get_firecrawl_client() -> AsyncFirecrawl:
    """Get or initialize FireCrawl SDK client."""
    global _firecrawl_client

    if _firecrawl_client is None:
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError(
                "FIRECRAWL_API_KEY environment variable not set. "
                "Get your API key from https://firecrawl.dev"
            )
        _firecrawl_client = AsyncFirecrawl(api_key=api_key)

    return _firecrawl_client


@tool
async def firecrawl_scrape_url(
    url: str,
    formats: Optional[List[str]] = None
) -> str:
    """Scrape a single webpage and extract content.

    Args:
        url: The URL to scrape
        formats: Optional list of formats to extract. Options: 'markdown', 'html', 'rawHtml', 'screenshot', 'links'
                Default: ['markdown']

    Returns:
        JSON string with scraped content
    """
    try:
        client = get_firecrawl_client()

        if formats is None:
            formats = ['markdown']

        result = await client.scrape_url(
            url=url,
            params={
                'formats': formats,
                'onlyMainContent': True
            }
        )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }, ensure_ascii=False)


@tool
async def firecrawl_search_web(
    query: str,
    max_results: int = 5
) -> str:
    """Search the web and get content from results.

    Args:
        query: Search query
        max_results: Maximum number of results (default: 5)

    Returns:
        JSON string with search results and their content
    """
    try:
        client = get_firecrawl_client()

        results = await client.search(
            query=query,
            limit=max_results,
            scrape_options={
                "formats": ["markdown"]
            }
        )

        return json.dumps(results, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }, ensure_ascii=False)


@tool
async def firecrawl_extract_structured(
    url: str,
    prompt: str,
    schema: Optional[str] = None
) -> str:
    """Extract structured data from a webpage using AI.

    Args:
        url: The URL to extract data from
        prompt: Natural language description of what data to extract
        schema: Optional JSON schema as string

    Returns:
        JSON string with extracted structured data
    """
    try:
        client = get_firecrawl_client()

        schema_dict = None
        if schema:
            try:
                schema_dict = json.loads(schema)
            except json.JSONDecodeError:
                return json.dumps({
                    'success': False,
                    'error': 'Invalid JSON schema provided'
                }, ensure_ascii=False)

        result = await client.extract(
            urls=[url],
            prompt=prompt,
            schema=schema_dict
        )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }, ensure_ascii=False)


@tool
async def firecrawl_crawl_website(
    url: str,
    max_pages: int = 10,
    include_paths: Optional[List[str]] = None,
    exclude_paths: Optional[List[str]] = None
) -> str:
    """Crawl a website and extract content from multiple pages.

    Args:
        url: The starting URL to crawl
        max_pages: Maximum number of pages to crawl (default: 10)
        include_paths: Optional list of paths to include (e.g., ['/blog/*'])
        exclude_paths: Optional list of paths to exclude (e.g., ['/admin/*'])

    Returns:
        JSON string with crawled content
    """
    try:
        client = get_firecrawl_client()

        crawl_params = {
            'limit': max_pages,
            'scrapeOptions': {
                'formats': ['markdown'],
                'onlyMainContent': True
            }
        }

        if include_paths:
            crawl_params['includePaths'] = include_paths
        if exclude_paths:
            crawl_params['excludePaths'] = exclude_paths

        result = await client.crawl_url(url=url, params=crawl_params)

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }, ensure_ascii=False)


@tool
async def firecrawl_map_website(
    url: str,
    search_query: Optional[str] = None,
    limit: int = 100
) -> str:
    """Map a website to discover all its pages/URLs.

    Args:
        url: The website URL to map
        search_query: Optional search query to filter results
        limit: Maximum number of URLs to return (default: 100)

    Returns:
        JSON string with list of discovered URLs
    """
    try:
        client = get_firecrawl_client()

        map_params = {'limit': limit}
        if search_query:
            map_params['search'] = search_query

        result = await client.map_url(url=url, params=map_params)

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({
            'success': False,
            'error': f'{type(e).__name__}: {str(e)}'
        }, ensure_ascii=False)


# ============================================
# Get All Tools for Cerebras Researcher
# ============================================

def get_cerebras_firecrawl_tools(config: RunnableConfig):
    """Get all tools available for the Cerebras Firecrawl researcher.

    Includes: FireCrawl SDK tools, think_tool, ResearchComplete
    """
    from langchain_core.tools import tool as tool_decorator

    tools = [tool_decorator(ResearchComplete), think_tool]

    # Add FireCrawl tools if API key is available
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if firecrawl_key:
        tools.extend([
            firecrawl_scrape_url,
            firecrawl_search_web,
            firecrawl_extract_structured,
            firecrawl_crawl_website,
            firecrawl_map_website,
        ])
    else:
        raise ValueError(
            "FIRECRAWL_API_KEY not set. This researcher requires Firecrawl API access."
        )

    return tools


# ============================================
# Agent Nodes - Following Original Template
# ============================================

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear."""

    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")

    messages = state["messages"]

    base_model = get_cerebras_model(
        max_tokens=configurable.research_model_max_tokens,
    )

    clarification_model = (
        base_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    if response.need_clarification:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief."""

    configurable = Configuration.from_runnable_config(config)

    base_model = get_cerebras_model(
        max_tokens=configurable.research_model_max_tokens,
    )

    research_model = (
        base_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])

    # Firecrawl-specific context
    firecrawl_context = """

You have access to powerful web research tools via Firecrawl SDK:
- firecrawl_search_web: Search the web and get content from results
- firecrawl_scrape_url: Scrape individual webpages for detailed content
- firecrawl_extract_structured: Extract structured data from pages using AI
- firecrawl_crawl_website: Crawl multiple pages from a website
- firecrawl_map_website: Discover all URLs on a website

Use these tools strategically to gather comprehensive research data.
"""

    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    ) + firecrawl_context

    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers."""

    configurable = Configuration.from_runnable_config(config)

    base_model = get_cerebras_model(
        max_tokens=configurable.research_model_max_tokens,
    )

    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    research_model = (
        base_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )


async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor."""

    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # Check exit conditions
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )

    # Process tool calls
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}

    # Handle think_tool calls
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]

    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))

    # Handle ConductResearch calls
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]

    if conduct_research_calls:
        try:
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]

            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config)
                for tool_call in allowed_conduct_research_calls
            ]

            tool_results = await asyncio.gather(*research_tasks)

            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Exceeded max concurrent research units ({configurable.max_concurrent_research_units})",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))

            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", []))
                for observation in tool_results
            ])

            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]

        except Exception as e:
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )

    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    )


# Build Supervisor Subgraph
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research with Firecrawl tools."""

    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # Get all available research tools
    tools = get_cerebras_firecrawl_tools(config)

    base_model = get_cerebras_model(
        max_tokens=configurable.research_model_max_tokens,
    )

    firecrawl_info = """

## Web Research Tools (Firecrawl SDK)
You have access to powerful web research tools:

- **firecrawl_search_web**: Search the web for information on any topic
- **firecrawl_scrape_url**: Get detailed content from a specific webpage
- **firecrawl_extract_structured**: Extract specific structured data from pages
- **firecrawl_crawl_website**: Crawl multiple pages from a website
- **firecrawl_map_website**: Discover all URLs on a website

Strategy:
1. Start with broad searches to find relevant sources
2. Scrape promising URLs for detailed content
3. Use extraction for structured data when needed
4. Cross-reference information from multiple sources
"""

    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or firecrawl_info,
        date=get_today_str()
    )

    research_model = (
        base_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)

    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )


async def execute_tool_safely(tool_obj, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool_obj.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher."""

    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]

    has_tool_calls = bool(most_recent_message.tool_calls)

    if not has_tool_calls:
        return Command(goto="compress_research")

    # Get all tools
    tools = get_cerebras_firecrawl_tools(config)
    tools_by_name = {
        t.name if hasattr(t, "name") else t.get("name", "unknown"): t
        for t in tools
    }

    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        )
        for observation, tool_call in zip(observations, tool_calls)
    ]

    # Check exit conditions
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if exceeded_iterations or research_complete_called:
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )

    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )


async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise summary."""

    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = get_cerebras_model(
        max_tokens=configurable.compression_model_max_tokens,
    )

    researcher_messages = state.get("researcher_messages", [])
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

    synthesis_attempts = 0
    max_attempts = 3

    while synthesis_attempts < max_attempts:
        try:
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages

            response = await synthesizer_model.ainvoke(messages)

            raw_notes_content = "\n".join([
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])

            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }

        except Exception as e:
            synthesis_attempts += 1

            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

            continue

    raw_notes_content = "\n".join([
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])

    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }


# Build Researcher Subgraph
researcher_builder = StateGraph(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report."""

    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    configurable = Configuration.from_runnable_config(config)
    writer_model = get_cerebras_model(
        max_tokens=configurable.final_report_model_max_tokens,
    )

    max_retries = 3
    current_retry = 0
    findings_token_limit = None

    while current_retry <= max_retries:
        try:
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )

            final_report = await writer_model.ainvoke([
                HumanMessage(content=final_report_prompt)
            ])

            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state
            }

        except Exception as e:
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1

                if current_retry == 1:
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    findings_token_limit = model_token_limit * 4
                else:
                    findings_token_limit = int(findings_token_limit * 0.9)

                findings = findings[:findings_token_limit]
                continue
            else:
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }

    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }


# ============================================
# Build Main Graph
# ============================================

cerebras_firecrawl_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

# Add nodes
cerebras_firecrawl_researcher_builder.add_node("clarify_with_user", clarify_with_user)
cerebras_firecrawl_researcher_builder.add_node("write_research_brief", write_research_brief)
cerebras_firecrawl_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
cerebras_firecrawl_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add edges
cerebras_firecrawl_researcher_builder.add_edge(START, "clarify_with_user")
cerebras_firecrawl_researcher_builder.add_edge("research_supervisor", "final_report_generation")
cerebras_firecrawl_researcher_builder.add_edge("final_report_generation", END)

# Compile the main graph
cerebras_firecrawl_researcher = cerebras_firecrawl_researcher_builder.compile()


# ============================================
# Simple API for direct invocation
# ============================================

async def run_research(topic: str, config: Optional[dict] = None) -> dict:
    """
    Run deep research on a topic.

    Args:
        topic: The research topic/question
        config: Optional configuration overrides

    Returns:
        Dictionary with 'report' and 'messages' keys
    """
    runnable_config = RunnableConfig(configurable=config or {})

    result = await cerebras_firecrawl_researcher.ainvoke(
        AgentInputState(messages=[HumanMessage(content=topic)]),
        config=runnable_config,
    )

    return {
        "report": result.get("final_report", ""),
        "messages": result.get("messages", []),
    }
