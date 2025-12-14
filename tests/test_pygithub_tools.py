"""Test the new PyGithub-based GitHub tools."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Force reload .env
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

from open_deep_research.grok_github_researcher import (
    get_github_client,
    get_default_repo_name,
    github_list_issues,
    github_get_issue,
    github_get_repo_info,
    github_list_repo_contents,
)

print("Testing PyGithub Tools")
print("=" * 50)

# Test 1: Get GitHub client
print("\n1. Testing GitHub client initialization...")
client = get_github_client()
if client:
    print(f"   SUCCESS: GitHub client initialized")
    user = client.get_user()
    print(f"   Authenticated as: {user.login}")
else:
    print("   FAILED: Could not initialize GitHub client")
    sys.exit(1)

# Test 2: Get default repo
print("\n2. Testing default repo extraction...")
repo_name = get_default_repo_name()
print(f"   Default repo: {repo_name}")

# Test 3: Get repo info
print("\n3. Testing github_get_repo_info...")
result = github_get_repo_info.invoke({})
print(f"   Result (first 500 chars): {result[:500]}...")

# Test 4: List issues
print("\n4. Testing github_list_issues...")
result = github_list_issues.invoke({"max_results": 5})
print(f"   Result (first 500 chars): {result[:500]}...")

# Test 5: List repo contents
print("\n5. Testing github_list_repo_contents...")
result = github_list_repo_contents.invoke({"path": ""})
print(f"   Result (first 500 chars): {result[:500]}...")

print("\n" + "=" * 50)
print("All tests completed successfully!")
