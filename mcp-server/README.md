# Cerebras + Firecrawl Deep Research MCP Server

An MCP (Model Context Protocol) server that provides deep research capabilities using Cerebras AI (DIRECT) for ultra-fast inference and Firecrawl for comprehensive web scraping.

## Features

- **Deep Research**: Comprehensive multi-step research on any topic
- **Quick Search**: Fast web searches using Firecrawl
- **URL Scraping**: Extract content from specific webpages
- **Data Extraction**: AI-powered structured data extraction
- **Cerebras Direct**: Uses Cerebras API directly (not via OpenRouter)

## Installation

### Option 1: Local NPX (from repository)

```bash
cd /path/to/open_deep_research/mcp-server
npm start
```

### Option 2: Python Direct

```bash
cd /path/to/open_deep_research
pip install -e .
cerebras-research-mcp
```

### Option 3: Python Module

```bash
cd /path/to/open_deep_research
python -m open_deep_research.mcp_stdio_server
```

## Configuration

### Required Environment Variables

```bash
# Firecrawl API key (required for web research)
FIRECRAWL_API_KEY=fc-your-key

# Cerebras API key (DIRECT - not OpenRouter)
CEREBRAS_API_KEY=your-cerebras-key

# Model name (optional, default: zai-glm-4.6)
CEREBRAS_MODEL=zai-glm-4.6
```

### Claude Desktop Configuration (Local Path)

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

```json
{
  "mcpServers": {
    "cerebras-research": {
      "command": "node",
      "args": ["C:/projects/Deep research/open_deep_research/mcp-server/index.js"],
      "env": {
        "FIRECRAWL_API_KEY": "fc-your-key",
        "CEREBRAS_API_KEY": "your-cerebras-key",
        "CEREBRAS_MODEL": "zai-glm-4.6"
      }
    }
  }
}
```

### Claude Code Configuration (Local Path)

Add to your Claude Code settings (`~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "cerebras-research": {
      "command": "node",
      "args": ["C:/projects/Deep research/open_deep_research/mcp-server/index.js"],
      "env": {
        "FIRECRAWL_API_KEY": "fc-your-key",
        "CEREBRAS_API_KEY": "your-cerebras-key",
        "CEREBRAS_MODEL": "zai-glm-4.6"
      }
    }
  }
}
```

### Alternative: Python Direct (without Node.js wrapper)

```json
{
  "mcpServers": {
    "cerebras-research": {
      "command": "python",
      "args": ["-m", "open_deep_research.mcp_stdio_server"],
      "cwd": "C:/projects/Deep research/open_deep_research",
      "env": {
        "PYTHONPATH": "C:/projects/Deep research/open_deep_research/src",
        "FIRECRAWL_API_KEY": "fc-your-key",
        "CEREBRAS_API_KEY": "your-cerebras-key"
      }
    }
  }
}
```

## Available Tools

### `deep_research`

Conduct comprehensive deep research on any topic.

**Parameters:**
- `topic` (required): The research topic or question
- `max_iterations` (optional): Maximum research iterations (default: 3)

**Example:**
```
Research the current state of quantum computing in 2024,
including major breakthroughs, key players, and future outlook.
```

### `quick_search`

Perform a quick web search without full research workflow.

**Parameters:**
- `query` (required): The search query
- `max_results` (optional): Maximum results (default: 5)

### `scrape_url`

Scrape content from a specific URL.

**Parameters:**
- `url` (required): The URL to scrape

### `extract_data`

Extract structured data from a webpage using AI.

**Parameters:**
- `url` (required): The URL to extract from
- `prompt` (required): Description of what data to extract

## How It Works

1. **Research Planning**: The supervisor agent analyzes your research question and creates a research plan
2. **Parallel Research**: Multiple researcher agents work in parallel to gather information
3. **Web Scraping**: Firecrawl SDK is used to search, scrape, and extract content from the web
4. **Synthesis**: Findings are compressed and synthesized
5. **Report Generation**: A final comprehensive report is generated with citations

## Model Options

### Direct Cerebras API (This Implementation)

This server uses Cerebras API directly for ultra-fast inference:

```bash
CEREBRAS_API_KEY=your-cerebras-key
CEREBRAS_MODEL=zai-glm-4.6
```

Get your API key from: https://cloud.cerebras.ai

### Available Cerebras Models

- `zai-glm-4.6` (default) - Best for deep research, tool calling, reasoning
  - Context: 64K (Free) / 131K (Paid)
  - Max Output: 40K tokens
  - Speed: ~1000 tokens/sec
- `llama-3.3-70b` - Alternative high quality model
- `llama3.1-8b` - Faster, lighter

## Troubleshooting

### "Python not found"

Make sure Python 3.10+ is installed and in your PATH:
```bash
python3 --version
```

### "FIRECRAWL_API_KEY not set"

Get your API key from [https://firecrawl.dev](https://firecrawl.dev)

### "CEREBRAS_API_KEY not set"

Get your API key from [https://cloud.cerebras.ai](https://cloud.cerebras.ai)

### "Module not found" errors

Make sure you've installed the package:
```bash
cd /path/to/open_deep_research
pip install -e .
```

## License

MIT
