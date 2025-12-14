"""Test the LangGraph API to verify OpenRouter/Cerebras configuration."""

import requests
import json
import time
from datetime import datetime

def test_research_agent():
    """Test the research agent with a simple query."""

    base_url = "http://127.0.0.1:2024"

    # Create a new thread
    print("Creating a new thread...")
    thread_response = requests.post(
        f"{base_url}/threads",
        json={"metadata": {"test": "openrouter_cerebras"}}
    )

    if thread_response.status_code != 200:
        print(f"Failed to create thread: {thread_response.text}")
        return

    thread_data = thread_response.json()
    thread_id = thread_data["thread_id"]
    print(f"Created thread: {thread_id}")

    # Submit a research query
    query = "What are the latest developments in quantum computing?"
    print(f"\nSubmitting query: {query}")

    run_response = requests.post(
        f"{base_url}/threads/{thread_id}/runs/stream",
        json={
            "assistant_id": "Deep Researcher",
            "input": {
                "messages": [
                    {
                        "role": "human",
                        "content": query
                    }
                ]
            },
            "stream_mode": "updates"
        },
        stream=True
    )

    if run_response.status_code != 200:
        print(f"Failed to start run: {run_response.text}")
        return

    print("\nReceiving updates...")
    print("-" * 80)

    model_info_found = False
    last_node = None

    # Process streaming responses
    for line in run_response.iter_lines():
        if line:
            line_text = line.decode('utf-8')

            # Skip empty lines and metadata
            if not line_text.strip() or line_text.startswith('event:'):
                continue

            # Parse data lines
            if line_text.startswith('data:'):
                try:
                    data_str = line_text[5:].strip()
                    if data_str:
                        data = json.loads(data_str)

                        # Track which node is executing
                        if isinstance(data, list) and len(data) > 0:
                            for item in data:
                                if isinstance(item, dict):
                                    for key, value in item.items():
                                        if key not in ['__start__', '__end__']:
                                            last_node = key
                                            print(f"\n[NODE: {key}]")

                                            # Look for messages
                                            if isinstance(value, dict):
                                                if 'messages' in value:
                                                    messages = value['messages']
                                                    if messages:
                                                        for msg in messages:
                                                            if isinstance(msg, dict):
                                                                content = msg.get('content', '')
                                                                if content:
                                                                    print(f"Message: {content[:200]}...")

                                                # Look for final report
                                                if 'final_report' in value:
                                                    report = value['final_report']
                                                    print(f"\nFinal Report Preview:")
                                                    print(report[:500] + "...")

                except json.JSONDecodeError:
                    continue

    print("\n" + "-" * 80)
    print("\nTest completed!")

    # Get thread state to check configuration
    print("\nChecking thread state for model information...")
    state_response = requests.get(f"{base_url}/threads/{thread_id}/state")

    if state_response.status_code == 200:
        state_data = state_response.json()
        print(f"Thread state retrieved successfully")

        # Save full state for inspection
        with open("test_thread_state.json", "w") as f:
            json.dump(state_data, f, indent=2)
        print("Full state saved to test_thread_state.json")

    return thread_id

if __name__ == "__main__":
    print("=" * 80)
    print("Testing OpenRouter/Cerebras Configuration via LangGraph API")
    print("=" * 80)

    thread_id = test_research_agent()

    if thread_id:
        print(f"\nThread ID for inspection: {thread_id}")
        print(f"You can view logs in the server output")
