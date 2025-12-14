"""
Cerebras-powered Deep Research Agent with GitHub MCP and FireCrawl SDK.

This agent uses the EXACT same structure as the original deep_researcher.py template,
with Cerebras GLM-4.6 via OpenRouter for ULTRA-FAST inference and additional tools:
- GitHub MCP integration for repository/code access and bug investigation
- FireCrawl SDK for web scraping and content extraction

Workflow: clarify_with_user -> write_research_brief -> research_supervisor -> final_report_generation
"""

import asyncio
import json
import os
from datetime import timedelta
from typing import List, Literal, Optional

# Force load .env file to override system environment variables
# This is critical because GITHUB_TOKEN may be set at system level with an old/different value
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
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    init_model_with_openrouter,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)


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


# ============================================
# GitHub Direct API Integration (PyGithub)
# Replaces MCP stdio transport which has ephemeral session issues on Windows
# ============================================

import logging
from github import Github, GithubException
from base64 import b64decode

logger = logging.getLogger(__name__)

_github_client = None


def get_github_client() -> Optional[Github]:
    """Get or initialize GitHub client using PyGithub."""
    global _github_client

    if _github_client is None:
        from dotenv import dotenv_values
        env_values = dotenv_values(find_dotenv(usecwd=True))
        github_token = env_values.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")

        if not github_token:
            logger.warning("GITHUB_TOKEN not set - GitHub tools will not be available")
            return None

        logger.info(f"Initializing PyGithub client with token: {github_token[:15]}...")
        _github_client = Github(github_token)

    return _github_client


def get_default_repo_name() -> Optional[str]:
    """Extract owner/repo from GITHUB_REPO env var."""
    from dotenv import dotenv_values
    env_values = dotenv_values(find_dotenv(usecwd=True))
    github_repo = env_values.get("GITHUB_REPO") or os.getenv("GITHUB_REPO", "")

    if not github_repo:
        return None

    # Parse URL: https://github.com/owner/repo -> owner/repo
    if "github.com/" in github_repo:
        parts = github_repo.split("github.com/")[-1].strip("/")
        return parts

    return github_repo


@tool
def github_list_issues(
    repo: Optional[str] = None,
    state: str = "open",
    max_results: int = 10
) -> str:
    """List issues from a GitHub repository.

    Args:
        repo: Repository in format 'owner/repo'. If not provided, uses GITHUB_REPO env var.
        state: Filter by issue state: 'open', 'closed', or 'all'. Default: 'open'
        max_results: Maximum number of issues to return. Default: 10

    Returns:
        JSON string with list of issues
    """
    try:
        client = get_github_client()
        if not client:
            return json.dumps({"error": "GitHub client not initialized. Set GITHUB_TOKEN."})

        repo_name = repo or get_default_repo_name()
        if not repo_name:
            return json.dumps({"error": "No repository specified. Set GITHUB_REPO or provide repo parameter."})

        repository = client.get_repo(repo_name)
        issues = repository.get_issues(state=state)

        result = []
        for i, issue in enumerate(issues):
            if i >= max_results:
                break
            result.append({
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "url": issue.html_url,
                "created_at": issue.created_at.isoformat() if issue.created_at else None,
                "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
                "labels": [label.name for label in issue.labels],
                "assignees": [a.login for a in issue.assignees],
                "body_preview": (issue.body or "")[:500] + "..." if issue.body and len(issue.body) > 500 else issue.body
            })

        return json.dumps({"issues": result, "total_count": len(result)}, ensure_ascii=False, indent=2)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API error: {e.data.get('message', str(e))}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@tool
def github_get_issue(
    issue_number: int,
    repo: Optional[str] = None
) -> str:
    """Get detailed information about a specific GitHub issue.

    Args:
        issue_number: The issue number
        repo: Repository in format 'owner/repo'. If not provided, uses GITHUB_REPO env var.

    Returns:
        JSON string with full issue details including comments
    """
    try:
        client = get_github_client()
        if not client:
            return json.dumps({"error": "GitHub client not initialized. Set GITHUB_TOKEN."})

        repo_name = repo or get_default_repo_name()
        if not repo_name:
            return json.dumps({"error": "No repository specified. Set GITHUB_REPO or provide repo parameter."})

        repository = client.get_repo(repo_name)
        issue = repository.get_issue(number=issue_number)

        # Get comments
        comments = []
        for comment in issue.get_comments():
            comments.append({
                "author": comment.user.login,
                "created_at": comment.created_at.isoformat() if comment.created_at else None,
                "body": comment.body
            })

        result = {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "url": issue.html_url,
            "created_at": issue.created_at.isoformat() if issue.created_at else None,
            "updated_at": issue.updated_at.isoformat() if issue.updated_at else None,
            "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
            "labels": [label.name for label in issue.labels],
            "assignees": [a.login for a in issue.assignees],
            "author": issue.user.login if issue.user else None,
            "body": issue.body,
            "comments_count": issue.comments,
            "comments": comments
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API error: {e.data.get('message', str(e))}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@tool
def github_get_file_contents(
    path: str,
    repo: Optional[str] = None,
    ref: Optional[str] = None
) -> str:
    """Read a file from a GitHub repository.

    Args:
        path: Path to the file in the repository (e.g., 'src/main.py')
        repo: Repository in format 'owner/repo'. If not provided, uses GITHUB_REPO env var.
        ref: Branch name, tag, or commit SHA. Default: default branch.

    Returns:
        JSON string with file content
    """
    try:
        client = get_github_client()
        if not client:
            return json.dumps({"error": "GitHub client not initialized. Set GITHUB_TOKEN."})

        repo_name = repo or get_default_repo_name()
        if not repo_name:
            return json.dumps({"error": "No repository specified. Set GITHUB_REPO or provide repo parameter."})

        repository = client.get_repo(repo_name)
        file_content = repository.get_contents(path, ref=ref) if ref else repository.get_contents(path)

        if isinstance(file_content, list):
            # It's a directory
            return json.dumps({
                "type": "directory",
                "path": path,
                "contents": [{"name": f.name, "type": f.type, "path": f.path} for f in file_content]
            }, ensure_ascii=False, indent=2)

        # Decode file content
        try:
            content = b64decode(file_content.content).decode('utf-8')
        except UnicodeDecodeError:
            content = "[Binary file - content not displayed]"

        result = {
            "type": "file",
            "path": file_content.path,
            "name": file_content.name,
            "size": file_content.size,
            "sha": file_content.sha,
            "content": content
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API error: {e.data.get('message', str(e))}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@tool
def github_search_code(
    query: str,
    repo: Optional[str] = None,
    max_results: int = 10
) -> str:
    """Search for code in a GitHub repository.

    Args:
        query: Search query (supports GitHub code search syntax)
        repo: Repository in format 'owner/repo'. If not provided, uses GITHUB_REPO env var.
        max_results: Maximum number of results. Default: 10

    Returns:
        JSON string with search results
    """
    try:
        client = get_github_client()
        if not client:
            return json.dumps({"error": "GitHub client not initialized. Set GITHUB_TOKEN."})

        repo_name = repo or get_default_repo_name()
        if not repo_name:
            return json.dumps({"error": "No repository specified. Set GITHUB_REPO or provide repo parameter."})

        # Add repo qualifier to query
        full_query = f"{query} repo:{repo_name}"

        code_results = client.search_code(full_query)

        results = []
        for i, item in enumerate(code_results):
            if i >= max_results:
                break
            results.append({
                "name": item.name,
                "path": item.path,
                "sha": item.sha,
                "url": item.html_url,
                "repository": item.repository.full_name
            })

        return json.dumps({"results": results, "total_count": min(code_results.totalCount, max_results)}, ensure_ascii=False, indent=2)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API error: {e.data.get('message', str(e))}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@tool
def github_list_pull_requests(
    repo: Optional[str] = None,
    state: str = "open",
    max_results: int = 10
) -> str:
    """List pull requests from a GitHub repository.

    Args:
        repo: Repository in format 'owner/repo'. If not provided, uses GITHUB_REPO env var.
        state: Filter by PR state: 'open', 'closed', or 'all'. Default: 'open'
        max_results: Maximum number of PRs to return. Default: 10

    Returns:
        JSON string with list of pull requests
    """
    try:
        client = get_github_client()
        if not client:
            return json.dumps({"error": "GitHub client not initialized. Set GITHUB_TOKEN."})

        repo_name = repo or get_default_repo_name()
        if not repo_name:
            return json.dumps({"error": "No repository specified. Set GITHUB_REPO or provide repo parameter."})

        repository = client.get_repo(repo_name)
        pulls = repository.get_pulls(state=state)

        result = []
        for i, pr in enumerate(pulls):
            if i >= max_results:
                break
            result.append({
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "url": pr.html_url,
                "created_at": pr.created_at.isoformat() if pr.created_at else None,
                "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
                "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
                "author": pr.user.login if pr.user else None,
                "head_branch": pr.head.ref,
                "base_branch": pr.base.ref,
                "body_preview": (pr.body or "")[:500] + "..." if pr.body and len(pr.body) > 500 else pr.body
            })

        return json.dumps({"pull_requests": result, "total_count": len(result)}, ensure_ascii=False, indent=2)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API error: {e.data.get('message', str(e))}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@tool
def github_get_pull_request(
    pr_number: int,
    repo: Optional[str] = None
) -> str:
    """Get detailed information about a specific pull request.

    Args:
        pr_number: The pull request number
        repo: Repository in format 'owner/repo'. If not provided, uses GITHUB_REPO env var.

    Returns:
        JSON string with full PR details
    """
    try:
        client = get_github_client()
        if not client:
            return json.dumps({"error": "GitHub client not initialized. Set GITHUB_TOKEN."})

        repo_name = repo or get_default_repo_name()
        if not repo_name:
            return json.dumps({"error": "No repository specified. Set GITHUB_REPO or provide repo parameter."})

        repository = client.get_repo(repo_name)
        pr = repository.get_pull(number=pr_number)

        # Get changed files
        files = []
        for f in pr.get_files():
            files.append({
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes
            })

        result = {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "url": pr.html_url,
            "created_at": pr.created_at.isoformat() if pr.created_at else None,
            "updated_at": pr.updated_at.isoformat() if pr.updated_at else None,
            "merged_at": pr.merged_at.isoformat() if pr.merged_at else None,
            "closed_at": pr.closed_at.isoformat() if pr.closed_at else None,
            "author": pr.user.login if pr.user else None,
            "head_branch": pr.head.ref,
            "base_branch": pr.base.ref,
            "body": pr.body,
            "mergeable": pr.mergeable,
            "merged": pr.merged,
            "additions": pr.additions,
            "deletions": pr.deletions,
            "changed_files_count": pr.changed_files,
            "files": files[:20]  # Limit to first 20 files
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API error: {e.data.get('message', str(e))}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@tool
def github_list_repo_contents(
    path: str = "",
    repo: Optional[str] = None,
    ref: Optional[str] = None
) -> str:
    """List contents of a directory in a GitHub repository.

    Args:
        path: Path to directory (empty string for root). Default: root
        repo: Repository in format 'owner/repo'. If not provided, uses GITHUB_REPO env var.
        ref: Branch name, tag, or commit SHA. Default: default branch.

    Returns:
        JSON string with directory contents
    """
    try:
        client = get_github_client()
        if not client:
            return json.dumps({"error": "GitHub client not initialized. Set GITHUB_TOKEN."})

        repo_name = repo or get_default_repo_name()
        if not repo_name:
            return json.dumps({"error": "No repository specified. Set GITHUB_REPO or provide repo parameter."})

        repository = client.get_repo(repo_name)
        contents = repository.get_contents(path, ref=ref) if ref else repository.get_contents(path)

        if not isinstance(contents, list):
            contents = [contents]

        result = []
        for item in contents:
            result.append({
                "name": item.name,
                "path": item.path,
                "type": item.type,
                "size": item.size if item.type == "file" else None,
                "sha": item.sha
            })

        return json.dumps({"path": path or "/", "contents": result}, ensure_ascii=False, indent=2)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API error: {e.data.get('message', str(e))}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@tool
def github_get_repo_info(
    repo: Optional[str] = None
) -> str:
    """Get information about a GitHub repository.

    Args:
        repo: Repository in format 'owner/repo'. If not provided, uses GITHUB_REPO env var.

    Returns:
        JSON string with repository information
    """
    try:
        client = get_github_client()
        if not client:
            return json.dumps({"error": "GitHub client not initialized. Set GITHUB_TOKEN."})

        repo_name = repo or get_default_repo_name()
        if not repo_name:
            return json.dumps({"error": "No repository specified. Set GITHUB_REPO or provide repo parameter."})

        repository = client.get_repo(repo_name)

        result = {
            "name": repository.name,
            "full_name": repository.full_name,
            "description": repository.description,
            "url": repository.html_url,
            "private": repository.private,
            "default_branch": repository.default_branch,
            "language": repository.language,
            "languages": dict(repository.get_languages()),
            "stars": repository.stargazers_count,
            "forks": repository.forks_count,
            "open_issues": repository.open_issues_count,
            "created_at": repository.created_at.isoformat() if repository.created_at else None,
            "updated_at": repository.updated_at.isoformat() if repository.updated_at else None,
            "topics": repository.get_topics()
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    except GithubException as e:
        return json.dumps({"error": f"GitHub API error: {e.data.get('message', str(e))}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


def get_github_tools() -> List:
    """Get all GitHub tools using PyGithub (direct API calls).

    This replaces the MCP-based tools which had ephemeral session issues on Windows.
    """
    from dotenv import dotenv_values
    env_values = dotenv_values(find_dotenv(usecwd=True))
    github_token = env_values.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")

    if not github_token:
        logger.warning("GITHUB_TOKEN not set - GitHub tools will not be available")
        return []

    logger.info(f"GitHub tools enabled with token: {github_token[:15]}...")

    return [
        github_list_issues,
        github_get_issue,
        github_get_file_contents,
        github_search_code,
        github_list_pull_requests,
        github_get_pull_request,
        github_list_repo_contents,
        github_get_repo_info,
    ]


# ============================================
# Get All Tools for Cerebras Researcher
# ============================================

def get_cerebras_researcher_tools(config: RunnableConfig):
    """Get all tools available for the Cerebras researcher.

    Includes: FireCrawl SDK tools, GitHub PyGithub tools, think_tool, ResearchComplete
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
        ])

    # Add GitHub tools using PyGithub (direct API, no MCP)
    github_tools = get_github_tools()
    tools.extend(github_tools)

    return tools


# ============================================
# Agent Nodes - Following Original Template
# ============================================

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.

    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")

    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]

    base_model = init_model_with_openrouter(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        tags=["langsmith:nostream"]
    )

    clarification_model = (
        base_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    # Step 4: Route based on clarification analysis
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
    """Transform user messages into a structured research brief and initialize supervisor."""

    configurable = Configuration.from_runnable_config(config)

    base_model = init_model_with_openrouter(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        tags=["langsmith:nostream"]
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

    # Add GitHub context if available
    github_repo = os.getenv("GITHUB_REPO", "")
    github_context = ""
    if github_repo:
        github_context = f"\n\nYou have access to the GitHub repository: {github_repo}. Use GitHub MCP tools to investigate code, issues, and pull requests."

    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    ) + github_context

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

    base_model = init_model_with_openrouter(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        tags=["langsmith:nostream"]
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
    """Individual researcher that conducts focused research with FireCrawl and GitHub tools."""

    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    # Get all available research tools
    tools = get_cerebras_researcher_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure FireCrawl API key "
            "and/or GitHub token."
        )

    base_model = init_model_with_openrouter(
        model=configurable.research_model,
        max_tokens=configurable.research_model_max_tokens,
        api_key=get_api_key_for_model(configurable.research_model, config),
        tags=["langsmith:nostream"]
    )

    # Build GitHub context for system prompt
    github_repo = os.getenv("GITHUB_REPO", "")
    github_info = ""
    if github_repo:
        github_info = f"""

## GitHub Repository Access
You have access to the GitHub repository: **{github_repo}**

Available GitHub tools for investigating code and bugs:
- get_file_contents: Read files from the repository
- search_code: Search for code patterns
- list_issues: List repository issues
- get_issue: Get issue details
- list_pull_requests: List PRs
- get_pull_request: Get PR details

Use these tools to investigate bugs, understand code structure, and find relevant context.
"""

    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or github_info,
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
    tools = get_cerebras_researcher_tools(config)
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
    synthesizer_model = init_model_with_openrouter(
        model=configurable.compression_model,
        max_tokens=configurable.compression_model_max_tokens,
        api_key=get_api_key_for_model(configurable.compression_model, config),
        tags=["langsmith:nostream"]
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
    writer_model = init_model_with_openrouter(
        model=configurable.final_report_model,
        max_tokens=configurable.final_report_model_max_tokens,
        api_key=get_api_key_for_model(configurable.final_report_model, config),
        tags=["langsmith:nostream"]
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
# Build Main Graph - Following Original Template
# ============================================

grok_github_researcher_builder = StateGraph(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration
)

# Add nodes - SAME as original deep_researcher.py
grok_github_researcher_builder.add_node("clarify_with_user", clarify_with_user)
grok_github_researcher_builder.add_node("write_research_brief", write_research_brief)
grok_github_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
grok_github_researcher_builder.add_node("final_report_generation", final_report_generation)

# Add edges - SAME as original deep_researcher.py
grok_github_researcher_builder.add_edge(START, "clarify_with_user")  # CRITICAL: Start with clarify!
grok_github_researcher_builder.add_edge("research_supervisor", "final_report_generation")
grok_github_researcher_builder.add_edge("final_report_generation", END)

# Compile the main graph
grok_github_researcher = grok_github_researcher_builder.compile()
