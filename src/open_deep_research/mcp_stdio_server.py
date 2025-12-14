#!/usr/bin/env python3
"""
MCP STDIO Server for Cerebras + Firecrawl Deep Research Agent.

This server exposes the deep research capabilities via the Model Context Protocol (MCP)
using STDIO transport for local execution with NPX or direct Python invocation.

Usage:
    python -m open_deep_research.mcp_stdio_server

Or via the installed command:
    cerebras-research-mcp
"""

import asyncio
import logging
import os
import sys
from typing import Optional

# Suppress stdout logging to avoid corrupting MCP STDIO transport
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr  # CRITICAL: Log to stderr, not stdout!
)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    logger.error("MCP package not installed. Install with: pip install mcp")
    sys.exit(1)


# Initialize MCP server
app = Server("cerebras-firecrawl-research")


def get_default_model() -> str:
    """Get the default Cerebras model from environment or use fallback."""
    return os.getenv("CEREBRAS_MODEL", "cerebras:zai-glm-4.6")


@app.list_tools()
async def list_tools():
    """List available research tools."""
    return [
        Tool(
            name="deep_research",
            description="""Conduct comprehensive deep research on any topic using Cerebras AI and Firecrawl web scraping.

This tool performs multi-step research:
1. Analyzes the research question
2. Creates a research plan
3. Searches and scrapes multiple web sources using Firecrawl
4. Synthesizes findings into a comprehensive report

Best for: Complex research questions, market analysis, technical investigations,
literature reviews, competitive analysis, and any topic requiring thorough web research.

Returns a detailed research report with citations.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The research topic or question. Be specific and detailed for best results."
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Maximum research iterations (default: 3)",
                        "default": 3
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="quick_search",
            description="""Perform a quick web search using Firecrawl without full deep research.

Use this for simple factual lookups or when you need quick information
without the full research workflow.

Returns search results with content summaries.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="scrape_url",
            description="""Scrape content from a specific URL using Firecrawl.

Extracts the main content from a webpage in markdown format.
Use this when you know the exact URL you need content from.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to scrape"
                    }
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="extract_data",
            description="""Extract structured data from a webpage using AI.

Uses Firecrawl's AI extraction to pull specific data points from a page.
Provide a natural language prompt describing what data you need.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to extract data from"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Natural language description of what data to extract"
                    }
                },
                "required": ["url", "prompt"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute a research tool."""
    logger.info(f"Tool called: {name} with arguments: {arguments}")

    try:
        if name == "deep_research":
            return await handle_deep_research(arguments)
        elif name == "quick_search":
            return await handle_quick_search(arguments)
        elif name == "scrape_url":
            return await handle_scrape_url(arguments)
        elif name == "extract_data":
            return await handle_extract_data(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def handle_deep_research(arguments: dict) -> list:
    """Handle deep research request."""
    topic = arguments.get("topic", "")
    max_iterations = arguments.get("max_iterations", 3)

    if not topic:
        return [TextContent(type="text", text="Error: 'topic' is required")]

    # Import the researcher
    from open_deep_research.cerebras_firecrawl_researcher import run_research

    # Prepare configuration
    config = {
        "research_model": get_default_model(),
        "max_researcher_iterations": max_iterations,
        "allow_clarification": False,  # No clarification in MCP context
    }

    logger.info(f"Starting deep research on: {topic}")

    result = await run_research(topic, config)

    report = result.get("report", "No report generated")

    return [TextContent(type="text", text=report)]


async def handle_quick_search(arguments: dict) -> list:
    """Handle quick search request."""
    query = arguments.get("query", "")
    max_results = arguments.get("max_results", 5)

    if not query:
        return [TextContent(type="text", text="Error: 'query' is required")]

    # Import Firecrawl tool
    from open_deep_research.cerebras_firecrawl_researcher import firecrawl_search_web

    logger.info(f"Quick search for: {query}")

    result = await firecrawl_search_web.ainvoke({
        "query": query,
        "max_results": max_results
    })

    return [TextContent(type="text", text=result)]


async def handle_scrape_url(arguments: dict) -> list:
    """Handle URL scraping request."""
    url = arguments.get("url", "")

    if not url:
        return [TextContent(type="text", text="Error: 'url' is required")]

    # Import Firecrawl tool
    from open_deep_research.cerebras_firecrawl_researcher import firecrawl_scrape_url

    logger.info(f"Scraping URL: {url}")

    result = await firecrawl_scrape_url.ainvoke({
        "url": url,
        "formats": ["markdown"]
    })

    return [TextContent(type="text", text=result)]


async def handle_extract_data(arguments: dict) -> list:
    """Handle data extraction request."""
    url = arguments.get("url", "")
    prompt = arguments.get("prompt", "")

    if not url:
        return [TextContent(type="text", text="Error: 'url' is required")]
    if not prompt:
        return [TextContent(type="text", text="Error: 'prompt' is required")]

    # Import Firecrawl tool
    from open_deep_research.cerebras_firecrawl_researcher import firecrawl_extract_structured

    logger.info(f"Extracting data from {url}: {prompt}")

    result = await firecrawl_extract_structured.ainvoke({
        "url": url,
        "prompt": prompt
    })

    return [TextContent(type="text", text=result)]


async def main():
    """Main entry point for STDIO MCP server."""
    logger.info("Starting Cerebras-Firecrawl Research MCP Server (STDIO)")

    # Validate required environment variables
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_key:
        logger.warning("FIRECRAWL_API_KEY not set - some tools may not work")

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    if not openrouter_key and not cerebras_key:
        logger.warning("No API key found for Cerebras (OPENROUTER_API_KEY or CEREBRAS_API_KEY)")

    # Run the STDIO server
    async with stdio_server() as (read_stream, write_stream):
        logger.info("STDIO streams connected, running server...")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


def run():
    """Entry point for console script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run()
