import os
from dotenv import load_dotenv

# load environment variables from .env if it exists
load_dotenv()

# ##################################################################
# llm configuration
# centralized configuration for the LLM backend (llama.cpp / OpenAI-compatible)

# default base url for the llama.cpp server
DEFAULT_BASE_URL = "http://localhost:8080/v1"

# default model identifier (legacy single-model fallback)
DEFAULT_MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_XL"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# resolve the legacy single-model env var first so it can act as the default for
# both tiers when only LLAMACPP_MODEL is set
_legacy_model = os.environ.get("LLAMACPP_MODEL", "") or DEFAULT_MODEL


# global configuration - set via environment variables or programmatically
_config = {
    "base_url": os.environ.get("LLAMACPP_BASE_URL", DEFAULT_BASE_URL),
    # legacy single-model entry, preserved for backward compatibility
    "model": _legacy_model,
    # tier-aware models; default to the legacy single model if their env vars are unset
    "small_model": os.environ.get("LLAMACPP_SMALL_MODEL", "") or _legacy_model,
    "large_model": os.environ.get("LLAMACPP_LARGE_MODEL", "") or _legacy_model,
    # optional model-swap proxy (e.g., llama-swap)
    "model_swap_url": os.environ.get("MODEL_SWAP_URL", "") or None,
    # keep both models resident; if true, ensure_model is a no-op
    "keep_models_loaded": _env_bool("KEEP_MODELS_LOADED", default=False),
}


# ##################################################################
# configure
# set the llm backend configuration programmatically
def configure(
    base_url: str | None = None,
    model: str | None = None,
    small_model: str | None = None,
    large_model: str | None = None,
    model_swap_url: str | None = None,
    keep_models_loaded: bool | None = None,
) -> None:
    if base_url is not None:
        # ensure the base_url ends with /v1 for OpenAI compatibility
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        _config["base_url"] = base_url
    if model is not None:
        _config["model"] = model
    if small_model is not None:
        _config["small_model"] = small_model
    if large_model is not None:
        _config["large_model"] = large_model
    if model_swap_url is not None:
        _config["model_swap_url"] = model_swap_url or None
    if keep_models_loaded is not None:
        _config["keep_models_loaded"] = bool(keep_models_loaded)


# ##################################################################
# get base url
# return the configured base url
def get_base_url() -> str:
    return _config["base_url"]


# ##################################################################
# get model
# return the configured model name. accepts an optional tier ("small"|"large").
# default tier is "small" so legacy callers keep working.
def get_model(tier: str = "small") -> str:
    if tier == "small":
        return _config["small_model"]
    if tier == "large":
        return _config["large_model"]
    raise ValueError(f"unknown tier: {tier!r} (expected 'small' or 'large')")


# ##################################################################
# get model swap url
# return the configured model-swap proxy url (or None)
def get_model_swap_url() -> str | None:
    return _config["model_swap_url"]


# ##################################################################
# get keep models loaded
# return whether both models are kept resident
def get_keep_models_loaded() -> bool:
    return bool(_config["keep_models_loaded"])
