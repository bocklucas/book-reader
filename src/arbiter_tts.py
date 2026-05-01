import base64
import logging
import time
from pathlib import Path

from arbiter_client import ArbiterClient, ArbiterError

log = logging.getLogger(__name__)

MAX_RETRIES = 10
RETRY_BACKOFF_SEC = 5


# ##################################################################
# submit
# submit a job of given type and return its id, retrying on transient errors
def _submit(client: ArbiterClient, job_type: str, params: dict) -> str:
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.submit(job_type, **params)
        except (ArbiterError, ConnectionError, OSError) as e:
            last_err = e
            wait = min(RETRY_BACKOFF_SEC * (attempt + 1), 60)
            log.warning("submit %s attempt %d/%d failed: %s — retrying in %ds",
                        job_type, attempt + 1, MAX_RETRIES, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"submit {job_type} failed after {MAX_RETRIES} attempts: {last_err}")


# ##################################################################
# fetch
# poll an existing job and write result; on failure resubmit and retry
def _fetch(client: ArbiterClient, job_id: str, job_type: str, params: dict,
           output_path: Path) -> None:
    current_jid = job_id
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            client.poll(current_jid, interval=0.5, timeout=900)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(client.get_result_bytes(current_jid))
            if output_path.stat().st_size >= 100:
                return
            raise RuntimeError("empty output")
        except (ArbiterError, RuntimeError, ConnectionError, OSError) as e:
            last_err = e
            wait = min(RETRY_BACKOFF_SEC * (attempt + 1), 60)
            log.warning("fetch %s attempt %d/%d for %s failed: %s — resubmitting in %ds",
                        job_type, attempt + 1, MAX_RETRIES, output_path.name, e, wait)
            time.sleep(wait)
            try:
                current_jid = _submit(client, job_type, params)
            except Exception as e2:
                last_err = e2
    raise RuntimeError(f"fetch {job_type} failed after {MAX_RETRIES} attempts for {output_path}: {last_err}")


# ##################################################################
# tts design to file
# generate a voice from description and save the wav locally
def tts_design_to_file(description: str, text: str, output_path: Path,
                       language: str = "English", temperature: float = 0.9) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 100:
        return output_path
    client = ArbiterClient(timeout=60)
    params = {
        "text": text,
        "instruct": description,
        "language": language,
        "temperature": temperature,
        "force": True,
    }
    jid = _submit(client, "tts-design", params)
    _fetch(client, jid, "tts-design", params, output_path)
    return output_path


# ##################################################################
# tts clone to file
# clone a voice from a reference WAV and save the result locally
def tts_clone_to_file(ref_wav: Path, text: str, output_path: Path,
                      language: str = "English", temperature: float = 0.7) -> Path:
    if output_path.exists() and output_path.stat().st_size >= 100:
        return output_path
    client = ArbiterClient(timeout=120)
    ref_b64 = base64.b64encode(ref_wav.read_bytes()).decode()
    params = {
        "text": text,
        "ref_audio": ref_b64,
        "language": language,
        "temperature": temperature,
        "force": True,
    }
    jid = _submit(client, "tts-clone", params)
    _fetch(client, jid, "tts-clone", params, output_path)
    return output_path


# ##################################################################
# tts clone many
# submit a batch of tts-clone jobs (one per line) using per-line ref_wav
def tts_clone_many(jobs: list[dict], language: str = "English",
                   temperature: float = 0.7) -> list[Path]:
    client = ArbiterClient(timeout=120)
    ref_cache: dict[Path, str] = {}
    pending: list[dict] = []
    for j in jobs:
        out_path: Path = j["output_path"]
        if out_path.exists() and out_path.stat().st_size >= 100:
            continue
        ref_wav: Path = j["ref_wav"]
        if ref_wav not in ref_cache:
            ref_cache[ref_wav] = base64.b64encode(ref_wav.read_bytes()).decode()
        params = {
            "text": j["text"],
            "ref_audio": ref_cache[ref_wav],
            "language": language,
            "temperature": temperature,
            "force": True,
        }
        jid = _submit(client, "tts-clone", params)
        pending.append({
            "job_id": jid,
            "output_path": out_path,
            "params": params,
        })
    for p in pending:
        _fetch(client, p["job_id"], "tts-clone", p["params"], p["output_path"])
    return [j["output_path"] for j in jobs]


__all__ = [
    "tts_design_to_file", "tts_clone_to_file", "tts_clone_many",
]
