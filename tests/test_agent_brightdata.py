"""Test the agent with Bright Data MCP via LangGraph API."""

import asyncio
import aiohttp
import json

async def test_agent_with_brightdata():
    """Test that the agent uses Bright Data MCP tools."""

    print("=" * 80)
    print("AGENT BRIGHT DATA MCP TEST")
    print("=" * 80)

    # LangGraph server URL
    base_url = "http://127.0.0.1:2024"

    # Create a thread
    async with aiohttp.ClientSession() as session:
        # Step 1: Create a thread
        print("\n[1] Creating a new thread...")
        async with session.post(f"{base_url}/threads") as resp:
            if resp.status != 200:
                print(f"Failed to create thread: {resp.status}")
                return False

            thread_data = await resp.json()
            thread_id = thread_data["thread_id"]
            print(f"[OK] Thread created: {thread_id}")

        # Step 2: Send a message that requires web search
        print("\n[2] Sending a message to trigger Bright Data MCP search...")
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": "מה המזג באילת היום? חפש באינטרנט"
                }
            ]
        }

        async with session.post(
            f"{base_url}/threads/{thread_id}/runs/stream",
            json=message,
            headers={"Content-Type": "application/json"}
        ) as resp:
            if resp.status != 200:
                print(f"Failed to send message: {resp.status}")
                text = await resp.text()
                print(f"Response: {text}")
                return False

            print(f"[OK] Streaming response...")
            print("\n" + "=" * 80)
            print("AGENT OUTPUT:")
            print("=" * 80)

            # Stream the response
            async for line in resp.content:
                line_text = line.decode('utf-8').strip()
                if line_text.startswith('data: '):
                    data_text = line_text[6:]  # Remove "data: " prefix
                    if data_text and data_text != "[DONE]":
                        try:
                            data = json.loads(data_text)

                            # Print relevant events
                            if 'event' in data:
                                event_type = data['event']

                                # Tool calls - this is what we want to see!
                                if event_type == 'on_chat_model_stream':
                                    if 'data' in data and 'chunk' in data['data']:
                                        chunk = data['data']['chunk']
                                        if 'tool_calls' in chunk:
                                            for tool_call in chunk['tool_calls']:
                                                if 'name' in tool_call:
                                                    print(f"\n[TOOL CALL] {tool_call['name']}")
                                                    if 'args' in tool_call:
                                                        print(f"  Args: {tool_call['args']}")

                                # Messages
                                elif event_type == 'on_chain_end':
                                    if 'data' in data and 'output' in data['data']:
                                        output = data['data']['output']
                                        if isinstance(output, dict) and 'messages' in output:
                                            for msg in output['messages']:
                                                if isinstance(msg, dict) and 'content' in msg:
                                                    if msg.get('type') == 'ai':
                                                        print(f"\n[AI]: {msg['content']}")

                        except json.JSONDecodeError:
                            pass  # Skip invalid JSON

            print("\n" + "=" * 80)

        print("\n[3] Checking thread history for tool usage...")
        async with session.get(f"{base_url}/threads/{thread_id}/state") as resp:
            if resp.status == 200:
                state = await resp.json()
                messages = state.get('values', {}).get('messages', [])

                # Look for Bright Data MCP tool calls
                brightdata_tools_used = []
                for msg in messages:
                    if isinstance(msg, dict):
                        if msg.get('type') == 'ai' and 'tool_calls' in msg:
                            for tool_call in msg['tool_calls']:
                                tool_name = tool_call.get('name', '')
                                if any(bd_tool in tool_name for bd_tool in ['search_engine', 'scrape']):
                                    brightdata_tools_used.append(tool_name)
                                    print(f"  [MCP TOOL USED] {tool_name}")

                if brightdata_tools_used:
                    print(f"\n[SUCCESS] Bright Data MCP tools were used: {brightdata_tools_used}")
                    return True
                else:
                    print("\n[WARN] No Bright Data MCP tools detected in conversation")
                    return False

    return False

if __name__ == "__main__":
    success = asyncio.run(test_agent_with_brightdata())

    print("\n" + "=" * 80)
    if success:
        print("[PASSED] Agent successfully used Bright Data MCP!")
    else:
        print("[FAILED] Agent did not use Bright Data MCP tools")
    print("=" * 80)
