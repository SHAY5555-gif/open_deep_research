"""Configuration management for the Open Deep Research system."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    BRIGHTDATA = "brightdata"
    FIRECRAWL = "firecrawl"
    PERPLEXITY = "perplexity"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""

    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""


class FilesystemConfig(BaseModel):
    """Configuration for local filesystem access."""

    enabled: bool = Field(
        default=False,
        description="Enable filesystem access for reading local code files"
    )
    allowed_paths: Optional[List[str]] = Field(
        default=None,
        description="List of directory paths the agent can access (e.g., ['/path/to/project'])"
    )
    """List of allowed directory paths for filesystem access"""

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits."
            }
        }
    )
    # Research Configuration - Using FireCrawl as default for Cerebras
    search_api: SearchAPI = Field(
        default=SearchAPI.FIRECRAWL,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "firecrawl",
                "description": "Search API to use for research. FireCrawl is configured as default provider. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "FireCrawl", "value": SearchAPI.FIRECRAWL.value},
                    {"label": "BrightData", "value": SearchAPI.BRIGHTDATA.value},
                    {"label": "Perplexity (Fast & Focused)", "value": SearchAPI.PERPLEXITY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=6,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 6,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=10,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 10,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration - Using Cerebras GLM-4.6 directly
    summarization_model: str = Field(
        default="cerebras:zai-glm-4.6",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "cerebras:zai-glm-4.6",
                "description": "Model for summarizing research results from search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    max_content_length: int = Field(
        default=50000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 50000,
                "min": 1000,
                "max": 200000,
                "description": "Maximum character length for webpage content before summarization"
            }
        }
    )
    research_model: str = Field(
        default="cerebras:zai-glm-4.6",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "cerebras:zai-glm-4.6",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=40000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 40000,
                "description": "Maximum output tokens for research model (Cerebras GLM-4.6 max output: 40K)"
            }
        }
    )
    compression_model: str = Field(
        default="cerebras:zai-glm-4.6",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "cerebras:zai-glm-4.6",
                "description": "Model for compressing research findings from sub-agents."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=16384,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 16384,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="cerebras:zai-glm-4.6",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "cerebras:zai-glm-4.6",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=40000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 40000,
                "description": "Maximum output tokens for final report model (Cerebras GLM-4.6 max output: 40K)"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )
    # Filesystem configuration for local code access
    filesystem_config: Optional[FilesystemConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "object",
                "description": "Configuration for local filesystem access. Enable to allow the agent to read and analyze local code files."
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig.

        Priority order: configurable (API request) > environment variable > default
        """
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())

        # Build values with correct priority: configurable > env > default
        values: dict[str, Any] = {}
        for field_name in field_names:
            # First try configurable (from API request)
            config_value = configurable.get(field_name)
            if config_value is not None:
                values[field_name] = config_value
            else:
                # Fall back to environment variable
                env_value = os.environ.get(field_name.upper())
                if env_value is not None:
                    values[field_name] = env_value

        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        
        arbitrary_types_allowed = True