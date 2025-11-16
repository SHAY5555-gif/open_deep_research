"""Final verification test for Cerebras-only OpenRouter configuration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
load_dotenv()

from open_deep_research.utils import init_model_with_openrouter

def verify_cerebras_config():
    """Verify that all configuration is correct for Cerebras-only routing."""

    print("=" * 70)
    print("CEREBRAS-ONLY CONFIGURATION VERIFICATION")
    print("=" * 70)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[FAIL] OPENROUTER_API_KEY not found in environment")
        return False

    print(f"[OK] OpenRouter API Key found: {api_key[:15]}...")

    # Test model initialization
    try:
        model = init_model_with_openrouter(
            model="openrouter:z-ai/glm-4.6",
            max_tokens=1000,
            api_key=api_key,
            tags=["verification-test"],
            provider="Cerebras"
        )

        print(f"[OK] Model initialized: {type(model).__name__}")

        # Verify configuration
        checks_passed = []
        checks_failed = []

        # Check 1: OpenRouter base URL
        if hasattr(model, 'openai_api_base'):
            if "openrouter.ai" in str(model.openai_api_base):
                checks_passed.append("OpenRouter base URL")
                print(f"[OK] Base URL: {model.openai_api_base}")
            else:
                checks_failed.append("OpenRouter base URL incorrect")
        else:
            checks_failed.append("openai_api_base not found")

        # Check 2: Cerebras provider routing
        if hasattr(model, 'extra_body'):
            extra_body = model.extra_body
            if extra_body and 'provider' in extra_body:
                provider_config = extra_body['provider']

                # Check for hard requirement (only + no fallbacks)
                if provider_config.get('only') == ['Cerebras']:
                    checks_passed.append("Cerebras-only routing")
                    print(f"[OK] Provider routing: only=['Cerebras']")
                else:
                    checks_failed.append("Provider 'only' field not set to Cerebras")

                if provider_config.get('allow_fallbacks') == False:
                    checks_passed.append("Fallbacks disabled")
                    print(f"[OK] Fallbacks: allow_fallbacks=False")
                else:
                    checks_failed.append("Fallbacks not disabled")
            else:
                checks_failed.append("Provider routing not configured in extra_body")
        else:
            checks_failed.append("extra_body not found on model")

        # Check 3: Model name
        if hasattr(model, 'model_name'):
            if model.model_name == "z-ai/glm-4.6":
                checks_passed.append("Model name")
                print(f"[OK] Model name: {model.model_name}")
            else:
                checks_failed.append(f"Model name incorrect: {model.model_name}")

        # Print summary
        print("\n" + "=" * 70)
        print("VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"Checks passed: {len(checks_passed)}")
        for check in checks_passed:
            print(f"  [OK] {check}")

        if checks_failed:
            print(f"\nChecks failed: {len(checks_failed)}")
            for check in checks_failed:
                print(f"  [FAIL] {check}")
            return False

        print("\n" + "=" * 70)
        print("[SUCCESS] All verifications passed!")
        print("Configuration: OpenRouter -> Cerebras (HARD requirement)")
        print("Model: z-ai/glm-4.6")
        print("Fallbacks: DISABLED")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"[FAIL] Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_cerebras_config()
    sys.exit(0 if success else 1)
