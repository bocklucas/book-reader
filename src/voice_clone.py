import asyncio
import json
from pathlib import Path

from src.omnivoice_tts import tts_design_to_file
from src.state import get_hash, load_hashes, save_hashes, check_hash
from src.tts_config import get_speed

REF_SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Sphinx of black quartz, judge my vow. "
    "She sells seashells by the seashore on a sunny afternoon. "
    "How vexingly quick daft zebras jump."
)


_voice_hash_lock = asyncio.Lock()


# ##################################################################
# clone voice
# generate a single reference WAV per character via tts-design (called once)
def clone_voice(name: str, description: str, output_dir: Path) -> Path:
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    wav_path = voices_dir / f"{name}.wav"
    
    hash_path = voices_dir / ".voice_hashes.json"
    hashes = load_hashes(hash_path)
    current_speed = get_speed()
    current_hash = get_hash({"description": description, "speed": current_speed})
    
    if wav_path.exists() and wav_path.stat().st_size >= 100 and check_hash(hashes, name, current_hash):
        return wav_path

    wav_path.unlink(missing_ok=True)
    tts_design_to_file(description, REF_SAMPLE_TEXT, wav_path, temperature=0.9, speed=current_speed)
    hashes[name] = current_hash
    save_hashes(hash_path, hashes)
    return wav_path


# ##################################################################
# clone voice async
# generate a single reference WAV per character via tts-design (async)
async def clone_voice_async(name: str, description: str, output_dir: Path) -> Path:
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    wav_path = voices_dir / f"{name}.wav"

    hash_path = voices_dir / ".voice_hashes.json"
    current_speed = get_speed()
    current_hash = get_hash({"description": description, "speed": current_speed})

    async with _voice_hash_lock:
        hashes = load_hashes(hash_path)
        if wav_path.exists() and wav_path.stat().st_size >= 100 and check_hash(hashes, name, current_hash):
            return wav_path

    wav_path.unlink(missing_ok=True)
    from src.omnivoice_tts import tts_design_to_file_async
    await tts_design_to_file_async(description, REF_SAMPLE_TEXT, wav_path, temperature=0.9, speed=current_speed)

    async with _voice_hash_lock:
        hashes = load_hashes(hash_path)
        hashes[name] = current_hash
        save_hashes(hash_path, hashes)
    return wav_path


# ##################################################################
# clone all voices
# generate reference WAVs for every character in voices.json (sequential)
def clone_all_voices(output_dir: Path) -> list[Path]:
    voices_json_path = output_dir / "voices.json"
    if not voices_json_path.exists():
        raise ValueError("voices.json not found")
    voices_dir = output_dir / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    
    hash_path = voices_dir / ".voice_hashes.json"
    hashes = load_hashes(hash_path)
    
    voices = json.loads(voices_json_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    from tqdm import tqdm
    hashes_changed = False
    
    for char_id, info in tqdm(voices.items(), desc="Designing voices"):
        wav_path = voices_dir / f"{char_id}.wav"
        paths.append(wav_path)
        
        # Handle both {"char_id": {"description": "..."}} and {"char_id": "..."} formats
        description = info["description"] if isinstance(info, dict) else info
        current_speed = get_speed()
        current_hash = get_hash({"description": description, "speed": current_speed})
        
        if wav_path.exists() and wav_path.stat().st_size >= 100 and check_hash(hashes, char_id, current_hash):
            continue

        wav_path.unlink(missing_ok=True)
        tts_design_to_file(description, REF_SAMPLE_TEXT, wav_path, temperature=0.9, speed=current_speed)
        hashes[char_id] = current_hash
        hashes_changed = True
        
    if hashes_changed:
        save_hashes(hash_path, hashes)
        
    return paths


# ##################################################################
# voice path
# return the local wav path for a character voice
def voice_path(voices_dir: Path, name: str) -> Path:
    p = voices_dir / f"{name}.wav"
    if not p.exists():
        raise ValueError(f"voice file not found for {name}: {p}")
    return p
