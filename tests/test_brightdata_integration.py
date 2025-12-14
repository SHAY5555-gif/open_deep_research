"""Integration test for Bright Data MCP with full tool loading."""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from open_deep_research.utils import get_all_tools
from open_deep_research.configuration import Configuration, SearchAPI

async def test_brightdata_integration():
    """Test that Bright Data MCP tools are loaded in get_all_tools."""

    print("=" * 80)
    print("BRIGHT DATA MCP INTEGRATION TEST")
    print("=" * 80)

    # Check token
    token = os.getenv("BRIGHTDATA_API_TOKEN")
    if not token:
        print("ERROR: BRIGHTDATA_API_TOKEN not found in environment")
        return False

    print(f"Token found: {token[:20]}...")

    # Create config with BRIGHTDATA search_api
    config = {
        "configurable": {
            "search_api": SearchAPI.BRIGHTDATA.value,
        }
    }

    print("\n[TEST] Loading all tools (including MCP)...")

    try:
        # Load all tools - this should include Bright Data MCP tools
        tools = await get_all_tools(config)

        print(f"\n[OK] Successfully loaded {len(tools)} total tools")

        # Find Bright Data MCP tools
        mcp_tools = [t for t in tools if hasattr(t, 'name') and any(keyword in t.name for keyword in ['search_engine', 'scrape'])]

        print(f"\n[OK] Found {len(mcp_tools)} Bright Data MCP tools")

        if mcp_tools:
            print("\nBright Data MCP tools:")
            for tool in mcp_tools:
                print(f"  - {tool.name}: {tool.description[:100] if tool.description else 'No description'}...")
        else:
            print("\n[WARN] No Bright Data MCP tools found")

        # List all tools
        print(f"\nAll loaded tools ({len(tools)}):")
        for tool in tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.get('name', 'Unknown')
            print(f"  - {tool_name}")

        return len(mcp_tools) > 0

    except Exception as e:
        print(f"\n[ERROR] Failed to load tools: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_brightdata_integration())

    print("\n" + "=" * 80)
    if success:
        print("[PASSED] Integration test completed - Bright Data MCP tools loaded")
    else:
        print("[FAILED] Integration test failed - No Bright Data MCP tools found")
    print("=" * 80)
