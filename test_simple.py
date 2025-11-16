"""Simple test to verify OpenRouter/Cerebras is working."""

import requests
import json

base_url = "http://127.0.0.1:2024"

# Create thread
thread_resp = requests.post(f"{base_url}/threads", json={})
thread_id = thread_resp.json()["thread_id"]
print(f"Created thread: {thread_id}")

# Submit simple query
print("Submitting query...")
run_resp = requests.post(
    f"{base_url}/threads/{thread_id}/runs",
    json={
        "assistant_id": "Deep Researcher",
        "input": {
            "messages": [{"role": "human", "content": "What is 2+2?"}]
        }
    }
)

print(f"Response status: {run_resp.status_code}")
if run_resp.status_code == 200:
    print("Request succeeded! Check server logs for model information.")
else:
    print(f"Error: {run_resp.text}")
