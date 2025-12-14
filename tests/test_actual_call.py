"""Test actual API call to verify Cerebras works."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from open_deep_research.utils import init_model_with_openrouter

def test_actual_call():
    """Test actual API call with Cerebras routing."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found")
        return False

    print("Testing actual API call with Cerebras-only routing...")
    print(f"API Key: {api_key[:15]}...")

    try:
        # Initialize model with Cerebras-only routing
        model = init_model_with_openrouter(
            model="openrouter:z-ai/glm-4.6",
            max_tokens=1000,
            api_key=api_key,
            provider="Cerebras"
        )

        print(f"Model initialized: {type(model).__name__}")
        print(f"Model name: {model.model_name}")
        print(f"API base: {model.openai_api_base}")
        print(f"Extra body: {model.extra_body}")

        # Make actual API call
        print("\nMaking API call...")
        response = model.invoke("Say hi!")

        print(f"\n[SUCCESS] Response received: {response.content}")
        return True

    except Exception as e:
        print(f"\n[FAILED] Error during API call: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_actual_call()
    sys.exit(0 if success else 1)
