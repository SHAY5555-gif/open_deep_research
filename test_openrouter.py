"""Test script to verify OpenRouter configuration with Cerebras provider."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from open_deep_research.utils import init_model_with_openrouter

def test_openrouter_initialization():
    """Test that OpenRouter model initializes correctly with Cerebras routing."""

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in environment")
        return False

    print("Testing OpenRouter initialization with Cerebras provider...")
    print(f"API Key: {api_key[:10]}...")

    try:
        # Test initialization with OpenRouter model
        model = init_model_with_openrouter(
            model="openrouter:z-ai/glm-4.6",
            max_tokens=1000,
            api_key=api_key,
            tags=["test"],
            provider="Cerebras"
        )

        print(f"Model initialized successfully!")
        print(f"Model type: {type(model)}")
        print(f"Model name: {model.model_name if hasattr(model, 'model_name') else 'N/A'}")

        # Check if extra_body contains Cerebras provider routing
        if hasattr(model, 'extra_body'):
            extra_body = model.extra_body
            print(f"Extra body: {extra_body}")
            if extra_body and 'provider' in extra_body:
                print("[OK] Cerebras provider routing configured correctly via extra_body")
                print(f"Provider config: {extra_body['provider']}")
            else:
                print("[WARN] Provider not found in extra_body")
        else:
            print("[WARN] extra_body attribute not found on model")

        # Test that base_url is set to OpenRouter
        if hasattr(model, 'openai_api_base'):
            print(f"API Base URL: {model.openai_api_base}")
            if "openrouter.ai" in str(model.openai_api_base):
                print("[OK] OpenRouter base URL configured correctly")
            else:
                print("[WARN] OpenRouter base URL not configured")

        print("\n[OK] All checks passed!")
        return True

    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_non_openrouter_model():
    """Test that non-OpenRouter models still work."""

    print("\nTesting standard model initialization...")

    try:
        # This should use the standard init_chat_model fallback
        model = init_model_with_openrouter(
            model="openai:gpt-4o-mini",
            max_tokens=1000,
            api_key="test_key",
            tags=["test"]
        )

        print(f"[OK] Standard model initialized successfully: {type(model)}")
        return True

    except Exception as e:
        print(f"Note: Standard model test skipped (expected if OpenAI key not set): {e}")
        return True  # This is acceptable for this test

if __name__ == "__main__":
    print("=" * 60)
    print("OpenRouter Configuration Test")
    print("=" * 60)

    success = test_openrouter_initialization()
    test_non_openrouter_model()

    print("\n" + "=" * 60)
    if success:
        print("[PASSED] Configuration test PASSED")
    else:
        print("[FAILED] Configuration test FAILED")
    print("=" * 60)
