"""Direct test for Cerebras BrightData Researcher - no server required."""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(usecwd=True), override=True)

async def main():
    print("=" * 80)
    print("CEREBRAS BRIGHTDATA RESEARCHER - DIRECT TEST")
    print("=" * 80)

    # Check required API keys
    cerebras_key = os.getenv("CEREBRAS_API_KEY")
    brightdata_token = os.getenv("BRIGHTDATA_API_TOKEN")

    print(f"\nCEREBRAS_API_KEY: {'SET' if cerebras_key else 'NOT SET'}")
    print(f"BRIGHTDATA_API_TOKEN: {'SET' if brightdata_token else 'NOT SET'}")

    if not cerebras_key:
        print("\n[ERROR] CEREBRAS_API_KEY not set!")
        return

    if not brightdata_token:
        print("\n[ERROR] BRIGHTDATA_API_TOKEN not set!")
        return

    print("\n[OK] All API keys are set. Importing agent...")

    # Import the agent
    from open_deep_research.cerebras_brightdata_researcher import (
        cerebras_brightdata_researcher,
        run_research
    )
    from langchain_core.messages import HumanMessage

    print("[OK] Agent imported successfully!")

    # Test query
    test_query = "What are the latest developments in quantum computing in 2024?"

    print(f"\n[TEST] Running research query:")
    print(f"  Query: {test_query}")
    print("\n" + "-" * 80)
    print("RUNNING RESEARCH (this may take a few minutes)...")
    print("-" * 80 + "\n")

    try:
        result = await run_research(test_query)

        print("\n" + "=" * 80)
        print("RESEARCH COMPLETE!")
        print("=" * 80)

        report = result.get("report", "No report generated")
        print(f"\n{report[:2000]}...")  # Print first 2000 chars

        if len(report) > 2000:
            print(f"\n[...truncated, full report is {len(report)} characters]")

        print("\n[SUCCESS] Research completed!")

    except Exception as e:
        print(f"\n[ERROR] Research failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
