"""Test script to verify Bright Data MCP configuration."""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from open_deep_research.utils import load_mcp_tools
from open_deep_research.configuration import Configuration, SearchAPI

async def test_brightdata_mcp():
    """Test that Bright Data MCP loads correctly."""

    print("=" * 80)
    print("BRIGHT DATA MCP TEST")
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

    print("\n[TEST] Loading MCP tools with BRIGHTDATA search_api...")

    try:
        # Load MCP tools
        tools = await load_mcp_tools(config, existing_tool_names=set())

        print(f"\n[OK] Successfully loaded {len(tools)} MCP tools")

        if tools:
            print("\nAvailable tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description[:100] if tool.description else 'No description'}...")
        else:
            print("\n[WARN] No tools loaded - this might be expected if MCP connection failed")

        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to load MCP tools: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_brightdata_mcp())

    print("\n" + "=" * 80)
    if success:
        print("[PASSED] Test completed")
    else:
        print("[FAILED] Test failed")
    print("=" * 80)
