import asyncio

import httpx

from src.llm_config import (
    get_keep_models_loaded,
    get_model,
    get_model_swap_url,
)


# ##################################################################
# model swap
# helper to ask a model-swap proxy (e.g. llama-swap) to hot-swap which
# model the LLM server is currently serving. no-op when keep-models-loaded
# is set or no swap url is configured.

# module state: last successfully loaded tier. guarded by an asyncio lock
# so concurrent ensure_model calls don't race.
_last_loaded_tier: str | None = None
_swap_lock = asyncio.Lock()

# polling parameters for "wait for ready" after a swap. most swap proxies
# return 200 only once the model is up, so a single POST is usually enough,
# but we keep a bounded polling fallback in case the proxy returns early.
_POLL_ATTEMPTS = 30
_POLL_INTERVAL_SECONDS = 1.0


async def ensure_model(tier: str) -> None:
    """Ensure the model server is serving the requested tier.

    No-op if KEEP_MODELS_LOADED is set or MODEL_SWAP_URL is unset.
    Otherwise POSTs {"model": <resolved_model>} to MODEL_SWAP_URL.
    Tracks last-loaded tier in module state to skip redundant swaps.
    """
    global _last_loaded_tier

    # cheap pre-check before grabbing the lock
    if get_keep_models_loaded():
        return
    swap_url = get_model_swap_url()
    if not swap_url:
        return

    async with _swap_lock:
        # re-check after acquiring the lock; another task may have swapped
        # to the same tier while we were waiting
        if _last_loaded_tier == tier:
            return

        model = get_model(tier)
        payload = {"model": model}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(swap_url, json=payload)
            if response.status_code == 200:
                _last_loaded_tier = tier
                return

            # not ready yet; poll the same endpoint until 200 or we exhaust
            # the budget. keep this simple; most proxies don't need it.
            for _ in range(_POLL_ATTEMPTS):
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                response = await client.post(swap_url, json=payload)
                if response.status_code == 200:
                    _last_loaded_tier = tier
                    return

            response.raise_for_status()
            # if we somehow got here without a 200 and without raising, fail loudly
            raise RuntimeError(
                f"model-swap proxy at {swap_url} did not become ready for tier {tier!r}"
            )


def _reset_for_tests() -> None:
    """Reset module state. Test-only helper."""
    global _last_loaded_tier, _swap_lock
    _last_loaded_tier = None
    _swap_lock = asyncio.Lock()
