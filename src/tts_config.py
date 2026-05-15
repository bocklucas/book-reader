import os
from dotenv import load_dotenv

# load environment variables from .env if it exists
load_dotenv()

# ##################################################################
# tts configuration
# centralized configuration for the OmniVoice-FastAPI backend

# default base url for the OmniVoice-FastAPI server
DEFAULT_BASE_URL = "http://localhost:8880/v1"
DEFAULT_SPEED = 1  # User found 1.0 a touch too fast

# global configuration - set via environment variables or programmatically
_config = {
    "base_url": os.environ.get("OMNIVOICE_BASE_URL", DEFAULT_BASE_URL),
    "speed": float(os.environ.get("OMNIVOICE_SPEED", DEFAULT_SPEED)),
}

# ##################################################################
# configure
# set the tts backend configuration programmatically
def configure(base_url: str | None = None, speed: float | None = None) -> None:
    if base_url is not None:
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        _config["base_url"] = base_url
    if speed is not None:
        _config["speed"] = speed

# ##################################################################
# get base url
# return the configured base url
def get_base_url() -> str:
    return _config["base_url"]

# ##################################################################
# get speed
# return the configured speech speed
def get_speed() -> float:
    return _config["speed"]
