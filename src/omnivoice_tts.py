import asyncio
import logging
import time
from pathlib import Path

import httpx

from src.tts_config import get_base_url, get_speed

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 5

# ##################################################################
# sanitize instruct
# ensure only one attribute per category is present to avoid API conflicts
def sanitize_instruct(text: str) -> str:
    categories = {
        "gender": ["male", "female"],
        "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
        "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
        "style": ["whisper"],
        "accent": ["american accent", "australian accent", "british accent", "canadian accent", "chinese accent", "indian accent", "japanese accent", "korean accent", "portuguese accent", "russian accent"],
        "dialect": []
    }
    
    all_allowed = []
    keyword_to_category = {}
    for cat, keywords in categories.items():
        for k in keywords:
            all_allowed.append(k)
            keyword_to_category[k] = cat
            
    tags = [t.strip().lower() for t in text.split(",") if t.strip()]
    
    selected_tags = []
    seen_categories = set()
    
    for tag in tags:
        if tag in all_allowed:
            cat = keyword_to_category[tag]
            if cat not in seen_categories:
                selected_tags.append(tag)
                seen_categories.add(cat)
    
    return ", ".join(selected_tags) if selected_tags else "male, moderate pitch"


# ##################################################################
# submit request
# make a robust POST request to OmniVoice-FastAPI
def _submit_request(endpoint: str, data: dict, files: dict | None = None, timeout: float = 300.0) -> bytes:
    url = f"{get_base_url()}/{endpoint}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            with httpx.Client(timeout=timeout) as client:
                if files:
                    response = client.post(url, data=data, files=files)
                else:
                    response = client.post(url, data=data)
                response.raise_for_status()
                return response.content
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            err_details = ""
            if hasattr(e, "response") and e.response is not None:
                err_details = f" - Body: {e.response.text}"
            last_err = f"{e}{err_details}"
            wait = min(RETRY_BACKOFF_SEC * (attempt + 1), 60)
            log.warning("OmniVoice request to %s failed (attempt %d/%d): %s - retrying in %ds",
                        endpoint, attempt + 1, MAX_RETRIES, last_err, wait)
            time.sleep(wait)
    raise RuntimeError(f"OmniVoice request to {endpoint} failed after {MAX_RETRIES} attempts: {last_err}")


# ##################################################################
# submit request async
# make a robust async POST request to OmniVoice-FastAPI
async def _submit_request_async(endpoint: str, data: dict, files: dict | None = None, timeout: float = 300.0) -> bytes:
    url = f"{get_base_url()}/{endpoint}"
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if files:
                    # For AsyncClient, we need to handle files differently if we want to be fully async
                    # but httpx handles the byte-reading if we pass the file objects.
                    response = await client.post(url, data=data, files=files)
                else:
                    response = await client.post(url, data=data)
                response.raise_for_status()
                return response.content
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            err_details = ""
            if hasattr(e, "response") and e.response is not None:
                err_details = f" - Body: {e.response.text}"
            last_err = f"{e}{err_details}"
            wait = min(RETRY_BACKOFF_SEC * (attempt + 1), 60)
            log.warning("OmniVoice async request to %s failed (attempt %d/%d): %s - retrying in %ds",
                        endpoint, attempt + 1, MAX_RETRIES, last_err, wait)
            await asyncio.sleep(wait)
    raise RuntimeError(f"OmniVoice async request to {endpoint} failed after {MAX_RETRIES} attempts: {last_err}")


# ##################################################################
# tts design to file
# generate a voice from description and save the wav locally
def tts_design_to_file(description: str, text: str, output_path: Path,
                       language: str = "en", temperature: float = 0.9,
                       speed: float | None = None) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 100:
        return output_path

    data = {
        "text": text,
        "instruct": sanitize_instruct(description),
        # "language_id": language,
        "response_format": "wav",
        "speed": speed if speed is not None else get_speed(),
        "num_step": 32,
        "guidance_scale": 2.0,
    }

    content = _submit_request("audio/design", data=data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return output_path


# ##################################################################
# tts design to file async
# generate a voice from description and save the wav locally (async)
async def tts_design_to_file_async(description: str, text: str, output_path: Path,
                                 language: str = "en", temperature: float = 0.9,
                                 speed: float | None = None) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 100:
        return output_path

    data = {
        "text": text,
        "instruct": sanitize_instruct(description),
        # "language_id": language,
        "response_format": "wav",
        "speed": speed if speed is not None else get_speed(),
        "num_step": 32,
        "guidance_scale": 2.0,
    }

    content = await _submit_request_async("audio/design", data=data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return output_path


# ##################################################################
# tts clone to file
# clone a voice from a reference WAV and save the result locally
def tts_clone_to_file(ref_wav: Path, text: str, output_path: Path,
                      language: str = "en", temperature: float = 0.7,
                      speed: float | None = None, ref_text: str | None = None) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 100:
        return output_path

    data = {
        "text": text,
        # "language_id": language,
        "response_format": "wav",
        "speed": speed if speed is not None else get_speed(),
        "num_step": 32,
        "guidance_scale": 2.0,
    }
    if ref_text:
        data["ref_text"] = ref_text

    with open(ref_wav, "rb") as f:
        files = {"ref_audio": (ref_wav.name, f, "audio/wav")}
        content = _submit_request("audio/clone", data=data, files=files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return output_path


# ##################################################################
# tts clone to file async
# clone a voice from a reference WAV and save the result locally (async)
async def tts_clone_to_file_async(ref_wav: Path, text: str, output_path: Path,
                                language: str = "en", temperature: float = 0.7,
                                speed: float | None = None, ref_text: str | None = None) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 100:
        return output_path

    data = {
        "text": text,
        # "language_id": language,
        "response_format": "wav",
        "speed": speed if speed is not None else get_speed(),
        "num_step": 32,
        "guidance_scale": 2.0,
    }
    if ref_text:
        data["ref_text"] = ref_text

    # Reading file for async request
    content_bytes = ref_wav.read_bytes()
    files = {"ref_audio": (ref_wav.name, content_bytes, "audio/wav")}
    content = await _submit_request_async("audio/clone", data=data, files=files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content)
    return output_path


# ##################################################################
# tts clone many
# submit a batch of tts-clone jobs (one per line)
def tts_clone_many(jobs: list[dict], language: str = "en",
                   temperature: float = 0.7, speed: float | None = None) -> list[Path]:
    # We process sequentially to avoid overwhelming the local GPU model
    completed_paths = []
    from tqdm import tqdm
    for j in tqdm(jobs, desc="Cloning lines", leave=False):
        out_path: Path = j["output_path"]
        if out_path.exists() and out_path.stat().st_size >= 100:
            completed_paths.append(out_path)
            continue

        ref_wav: Path = j["ref_wav"]
        ref_text = j.get("ref_text")
        tts_clone_to_file(ref_wav, j["text"], out_path, language, temperature, speed, ref_text=ref_text)
        completed_paths.append(out_path)

    return completed_paths


__all__ = [
    "tts_design_to_file", "tts_clone_to_file", "tts_clone_many",
    "tts_design_to_file_async", "tts_clone_to_file_async",
]
