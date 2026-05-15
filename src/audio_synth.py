import json
import subprocess
import tempfile
from pathlib import Path

from src.omnivoice_tts import tts_clone_many
from src.voice_clone import voice_path
from src.state import get_hash, load_hashes, save_hashes, check_hash, get_file_hash
from src.tts_config import get_speed

SAMPLE_RATE = 24000


# ##################################################################
# concat wavs
# concatenate per-line wavs into a single chapter wav at SAMPLE_RATE mono
def concat_wavs(line_paths: list[Path], output_path: Path) -> None:
    if not line_paths:
        raise ValueError("No line files to concatenate")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in line_paths:
            f.write(f"file '{p}'\n")
        list_file = Path(f.name)
    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {result.stderr}")
    finally:
        list_file.unlink(missing_ok=True)


# ##################################################################
# synthesize chapter
# synthesize each script line via arbiter tts-clone with character ref WAVs
def synthesize_chapter(script_path: Path, audio_dir: Path,
                       voices_dir: Path, voices_config: dict,
                       voice_hashes: dict[str, str]) -> Path:
    from src.voice_clone import REF_SAMPLE_TEXT
    chapter_name = script_path.stem
    output_path = audio_dir / f"{chapter_name}.wav"

    work_dir = audio_dir / f".lines_{chapter_name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    jobs_by_voice = {}  # {voice_name: [jobs]}
    line_paths: list[Path] = []
    used_line_files = set()
    total_lines = 0
    cached_lines = 0

    with open(script_path, "r", encoding="utf-8") as f:
        for idx, raw in enumerate(f):
            raw = raw.strip()
            if not raw:
                continue
            entry = json.loads(raw)
            # Handle both formats: {"X": "text"} and {"speaker": "X", "text": "text"}
            speaker = entry.get("speaker") or entry.get("speaker_id")
            text = entry.get("text")

            if speaker is None or text is None:
                # Fallback to standard format: {"X": "text"}
                # Find the first key that isn't speaker/text
                for k, v in entry.items():
                    if k not in ("speaker", "speaker_id", "text"):
                        speaker = k
                        text = v
                        break
                # If still nothing, just take the first key
                if speaker is None and entry:
                    speaker = list(entry.keys())[0]
                    text = entry[speaker]

            if not text or not text.strip():
                continue

            # Use narrator if speaker not in config
            if speaker not in voices_config:
                if "narrator" not in voices_config:
                    raise ValueError(f"speaker {speaker!r} not in voices and no narrator fallback")
                speaker = "narrator"

            total_lines += 1

            # Hash consists of text, voice description, and the hash of the reference WAV
            voice_info = voices_config[speaker]
            voice_desc = voice_info["description"] if isinstance(voice_info, dict) else voice_info

            ref_wav = voice_path(voices_dir, speaker)
            ref_hash = voice_hashes.get(speaker, "")
            if not ref_hash:
                ref_hash = get_file_hash(ref_wav)
                voice_hashes[speaker] = ref_hash

            current_speed = get_speed()
            # We use a combined hash as the unique ID for this line's audio
            line_hash = get_hash({
                "text": text,
                "voice": voice_desc,
                "ref_hash": ref_hash,
                "speed": current_speed
            })

            # Use the hash as the filename to allow sharing and prevent issues with line shifts
            line_path = work_dir / f"{line_hash}.wav"
            line_paths.append(line_path)
            used_line_files.add(line_path.name)

            # Only generate if the file doesn't exist or is too small
            if not line_path.exists() or line_path.stat().st_size < 100:
                if speaker not in jobs_by_voice:
                    jobs_by_voice[speaker] = []
                jobs_by_voice[speaker].append({
                    "ref_wav": ref_wav,
                    "text": text,
                    "output_path": line_path,
                    "ref_text": REF_SAMPLE_TEXT
                })
            else:
                cached_lines += 1

    new_lines = total_lines - cached_lines
    if new_lines == 0:
        print(f"    {chapter_name}: all {total_lines} lines cached")
    else:
        print(f"    {chapter_name}: {new_lines} to generate, {cached_lines} cached (of {total_lines} total)")

    if jobs_by_voice:
        for speaker, jobs in jobs_by_voice.items():
            print(f"    {chapter_name}: generating {len(jobs)} lines for [{speaker}]")
            tts_clone_many(jobs)

        if output_path.exists():
            output_path.unlink()

    if not output_path.exists():
        concat_wavs(line_paths, output_path)

    # Cleanup orphaned line files in this chapter's directory
    orphaned = 0
    for p in work_dir.glob("*.wav"):
        if p.name not in used_line_files:
            p.unlink()
            orphaned += 1
    if orphaned:
        print(f"    {chapter_name}: cleaned up {orphaned} orphaned line files")

    return output_path


# ##################################################################
# synthesize all chapters
# convert all scripts to audio using clone-from-ref per line
def synthesize_all_chapters(output_dir: Path, max_chapters: int = 0) -> list[Path]:
    script_dir = output_dir / "scripts"
    audio_dir = output_dir / "audio"
    voices_dir = output_dir / "voices"
    if not script_dir.exists():
        raise ValueError("script directory not found")
    if not voices_dir.exists():
        raise ValueError("voices directory not found — run voices_clone step first")
    
    voices_json = output_dir / "voices.json"
    voices_config = json.loads(voices_json.read_text(encoding="utf-8"))
    
    audio_dir.mkdir(parents=True, exist_ok=True)
    script_files = sorted(script_dir.glob("*.jsonl"))
    if max_chapters > 0:
        script_files = script_files[:max_chapters]
    
    all_created = []
    voice_hashes: dict[str, str] = {}
    from tqdm import tqdm
    for script_path in tqdm(script_files, desc="Synthesizing chapters"):
        output_path = synthesize_chapter(script_path, audio_dir, voices_dir, voices_config, voice_hashes)
        all_created.append(output_path)
    return all_created
