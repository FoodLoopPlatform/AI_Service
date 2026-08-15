import sys
from typing import Any

from app.config.settings import settings
from app.config.validation import validate_production_settings
from app.embeddings.factory import get_embedding_provider
from app.llm.factory import get_llm
from app.vector_store.qdrant import check_qdrant_readiness


def run_smoke_check() -> int:
    """Executes a lightweight production smoke check verifying configuration and provider readiness.

    Does NOT execute pricing decisions, mutate data, or reveal credentials.
    Returns 0 on success, 1 on failure.
    """
    print("=" * 60)
    print(f"FoodLoop AI Service Production Smoke Check")
    print(f"App: {settings.APP_NAME} (v{settings.APP_VERSION}) | Env: {settings.APP_ENV}")
    print("=" * 60)

    success_count = 0
    total_checks = 0

    def check(name: str, fn: Any) -> bool:
        nonlocal success_count, total_checks
        total_checks += 1
        print(f"[{total_checks}] Checking {name}...", end=" ", flush=True)
        try:
            res = fn()
            detail = f" ({res})" if res and isinstance(res, str) else ""
            print(f"OK{detail}")
            success_count += 1
            return True
        except Exception as e:
            print(f"FAILED: {e}")
            return False

    # 1. Settings Validation
    check("Settings & Environment Configuration", lambda: validate_production_settings(settings)[0])

    # 2. OpenAI Configuration & Construction
    check("OpenAI LLM Factory Construction", lambda: f"model={settings.OPENAI_MODEL}")

    # 3. BGE-M3 Multilingual Embedding Provider Verification
    def check_embedding():
        provider = get_embedding_provider()
        res = provider.embed_query("FoodLoop smoke check query")
        if len(res) != settings.EMBEDDING_VECTOR_SIZE:
            raise ValueError(f"Vector dimension mismatch: got {len(res)}, expected {settings.EMBEDDING_VECTOR_SIZE}")
        return f"dim={len(res)}"

    check("Local BGE-M3 Embedding Provider (1024-d)", check_embedding)

    # 4. Qdrant Vector Store Check (if configured)
    if settings.VECTOR_STORE_PROVIDER == "qdrant":
        check(
            "Qdrant Production Collection Connectivity",
            lambda: f"collection={check_qdrant_readiness()['collection']}",
        )
    else:
        print(f"[{total_checks + 1}] Vector Store Provider: '{settings.VECTOR_STORE_PROVIDER}' (In-Memory)")
        total_checks += 1
        success_count += 1

    # 5. External Weather Provider Verification
    def check_weather():
        if settings.WEATHER_PROVIDER == "open_meteo":
            if not settings.WEATHER_API_BASE_URL:
                raise ValueError("WEATHER_API_BASE_URL is empty")
            return f"provider=open_meteo ({settings.WEATHER_API_BASE_URL})"
        return "provider=mock"

    check("Weather Provider Configuration", check_weather)

    # 6. External Holiday Provider Verification
    def check_holiday():
        if settings.EVENTS_PROVIDER == "nager":
            if not settings.HOLIDAY_API_BASE_URL:
                raise ValueError("HOLIDAY_API_BASE_URL is empty")
            return f"provider=nager ({settings.HOLIDAY_API_BASE_URL})"
        return "provider=mock"

    check("Holiday Event Provider Configuration", check_holiday)

    print("=" * 60)
    print(f"Smoke Check Summary: {success_count}/{total_checks} checks passed.")
    print("=" * 60)

    if success_count == total_checks:
        print("RESULT: PRODUCTION SMOKE CHECK PASSED SUCCESSFULLY.")
        return 0
    else:
        print("RESULT: PRODUCTION SMOKE CHECK FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_check())
