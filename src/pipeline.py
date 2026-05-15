import asyncio
import json
import time
from pathlib import Path

try:
    from colorama import Fore, Style, init

    init(autoreset=True)
except ImportError:  # pragma: no cover — colorama is optional for tests
    class _Dummy:
        def __getattr__(self, name: str) -> str:
            return ""

    Fore = Style = _Dummy()  # type: ignore[assignment]

from src.audio_synth import synthesize_chapter
from src.character_analysis import analyze_characters
from src.embeddings_db import EmbeddingsDB
from src.state import get_hash, get_file_hash, load_hashes, save_hashes, check_hash
from src.epub_extract import extract_epub, get_output_dir
from src.m4b_assemble import assemble_m4b
from src.model_swap import ensure_model
from src.script_generate import generate_chapter_script
from src.voice_clone import clone_voice_async
from src.voice_description import generate_voices


_stage_start: dict[int, float] = {}


def print_step(step_num: int, name: str) -> None:
    _stage_start[step_num] = time.monotonic()
    print(f"\n{Fore.BLUE}[Stage {step_num}]{Style.RESET_ALL} {Fore.WHITE}{name}{Style.RESET_ALL}")


def _fmt_elapsed(stage: int | None = None, since: float | None = None) -> str:
    start = since if since is not None else _stage_start.get(stage or 0)
    if start is None:
        return ""
    secs = time.monotonic() - start
    if secs < 60:
        return f" ({secs:.1f}s)"
    mins, s = divmod(secs, 60)
    return f" ({int(mins)}m {s:.0f}s)"


def print_done(message: str, *, stage: int | None = None) -> None:
    elapsed = _fmt_elapsed(stage) if stage else ""
    print(f"  {Fore.GREEN}✓{Style.RESET_ALL} {message}{elapsed}")


def _log(stage: int, message: str) -> None:
    tag = f"{Fore.BLUE}[S{stage}]{Style.RESET_ALL}"
    print(f"  {tag} {message}")


# ##################################################################
# read script speakers
# parse a JSONL script file and return the set of speaker ids used
def read_script_speakers(script_path: Path) -> set[str]:
    speakers: set[str] = set()
    with open(script_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or not raw.startswith("{"):
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            speaker = entry.get("speaker") or entry.get("speaker_id")
            if speaker is None and len(entry) == 1:
                speaker = next(iter(entry.keys()))
            if speaker is not None:
                speakers.add(str(speaker))
    return speakers


# ##################################################################
# load chapter files
# return list of chapter txt files under output_dir/chapters
def load_chapter_files(output_dir: Path) -> list[Path]:
    chapters_dir = output_dir / "chapters"
    return sorted(chapters_dir.glob("*.txt"))


# ##################################################################
# run pipeline async
# orchestrate the full pipeline with streaming producer/consumer and parallel voice clones
async def run_pipeline_async(
    epub_path: Path,
    output_dir: Path | None = None,
    max_chapters: int | None = None,
    voice_overrides_path: Path | None = None,
) -> Path:
    if output_dir is None:
        output_dir = get_output_dir(epub_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    _pipeline_start = time.monotonic()
    print(f"{Fore.CYAN}Book Reader Pipeline{Style.RESET_ALL}")
    print(f"  Input: {epub_path}")
    print(f"  Output: {output_dir}")

    try:
        # Stage 1 — small model: extract chapters
        print_step(1, "Extract chapters from EPUB")
        await ensure_model("small")
        _chapters_dir = output_dir / "chapters"
        _epub_cached = _chapters_dir.exists() and any(_chapters_dir.glob("*.txt"))
        title, author, written = await extract_epub(epub_path, output_dir)
        if _epub_cached:
            print_done(f"{len(written)} chapter files (cached)", stage=1)
        else:
            print_done(f"Extracted {len(written)} chapter files", stage=1)

        # Stage 2 — small model: analyze characters (parallel via Semaphore(5) inside)
        print_step(2, "Analyze characters")
        await ensure_model("small")
        _characters_path = output_dir / "characters.json"
        _chars_cached = _characters_path.exists()
        if _chars_cached:
            print(f"  (cached) Skipping character analysis")
        characters_path = await analyze_characters(output_dir, title, author, voice_overrides_path)
        characters = json.loads(characters_path.read_text(encoding="utf-8"))
        if _chars_cached:
            print_done(f"{len(characters)} characters (cached)", stage=2)
        else:
            print_done(f"Identified {len(characters)} characters", stage=2)

        # Stage 3 — no LLM: build embeddings DB
        print_step(3, "Build embeddings DB")
        db = EmbeddingsDB(output_dir)
        chapter_files = load_chapter_files(output_dir)
        _real_chapters = [f for f in chapter_files if f.name != "00-intro.txt"]

        _emb_hash_path = output_dir / ".embedding_hashes.json"
        _emb_hashes = load_hashes(_emb_hash_path)
        _emb_input = get_hash({f.name: get_file_hash(f) for f in _real_chapters})
        _emb_cached = db.load() and check_hash(_emb_hashes, "input", _emb_input)

        if _emb_cached:
            print(f"  (cached) Skipping embedding generation")
        else:
            for idx, ch_path in enumerate(_real_chapters, 1):
                print(f"  Chunking {idx}/{len(_real_chapters)}: {ch_path.stem}", end="\r", flush=True)
                db.add_chapter(ch_path.read_text(encoding="utf-8"), ch_path.name)
            if _real_chapters:
                print()
            db.finalize_and_save()
            _emb_hashes["input"] = _emb_input
            save_hashes(_emb_hash_path, _emb_hashes)
        print_done(f"Embeddings DB ready{' (cached)' if _emb_cached else ''}", stage=3)

        # Stage 4 — small model: voice descriptions via RAG
        print_step(4, "Generate voice descriptions")
        await ensure_model("small")
        _voices_path = output_dir / "voices.json"
        _voices_cached = _voices_path.exists()
        if _voices_cached:
            print(f"  (cached) Skipping LLM voice generation")
        voices_path = await generate_voices(output_dir, db, voice_overrides_path)
        voices_config = json.loads(voices_path.read_text(encoding="utf-8"))
        if _voices_cached:
            print_done(f"{len(voices_config)} characters (cached)", stage=4)
        else:
            print_done(f"Generated voice descriptions for {len(voices_config)} characters", stage=4)

        # Stage 5 — clone reference voices (parallel, completes before stage 6)
        print_step(5, "Clone reference voices")
        cached_voices: list[str] = []
        cloned_voices: list[str] = []

        _voice_hashes_path = output_dir / "voices" / ".voice_hashes.json"
        _stage5_hashes = load_hashes(_voice_hashes_path)
        _voices_hash = get_file_hash(_voices_path)
        
        if _voices_path.exists() and check_hash(_stage5_hashes, "overall", _voices_hash):
            print_done(f"All {len(voices_config)} voices up to date (cached)", stage=5)
        else:
            async def clone_one(name: str) -> None:
                info = voices_config[name]
                description = info["description"] if isinstance(info, dict) else info
                wav_path = output_dir / "voices" / f"{name}.wav"
                mtime_before = wav_path.stat().st_mtime_ns if wav_path.exists() else None
                await clone_voice_async(name, description, output_dir)
                mtime_after = wav_path.stat().st_mtime_ns
                if mtime_before == mtime_after:
                    cached_voices.append(name)
                else:
                    cloned_voices.append(name)
                    _log(5, f"{Fore.GREEN}✓{Style.RESET_ALL} {name} cloned")

            await asyncio.gather(*(clone_one(n) for n in voices_config))
            
            # Update overall hash after successful cloning
            _stage5_hashes = load_hashes(_voice_hashes_path)
            _stage5_hashes["overall"] = _voices_hash
            save_hashes(_voice_hashes_path, _stage5_hashes)
            
            if cloned_voices:
                print_done(f"{len(cached_voices)} cached, {len(cloned_voices)} cloned", stage=5)
            else:
                print_done(f"All {len(cached_voices)} voices cached", stage=5)

        # Stage 6 — large model: streaming script producer
        print_step(6, "Generate scripts (producer) + synthesize audio (consumer)")
        await ensure_model("large")

        # Optionally limit number of chapters processed downstream
        script_chapter_files = [f for f in chapter_files]
        if max_chapters is not None and max_chapters > 0:
            # keep intro + first max_chapters real chapters
            intro = [f for f in script_chapter_files if f.name == "00-intro.txt"]
            rest = [f for f in script_chapter_files if f.name != "00-intro.txt"]
            script_chapter_files = intro + rest[:max_chapters]

        script_queue: asyncio.Queue = asyncio.Queue()
        _total_chapters = len(script_chapter_files)
        _script_counter = 0
        _synth_counter = 0

        async def script_producer() -> None:
            nonlocal _script_counter
            for ch_path in script_chapter_files:
                _script_counter += 1
                _script_dir = output_dir / "scripts"
                _script_cached = (_script_dir / f"{ch_path.stem}.jsonl").exists()
                label = "(cached)" if _script_cached else "generating..."
                _log(6, f"Script {_script_counter}/{_total_chapters}: {ch_path.stem} {label}")
                sp = await generate_chapter_script(ch_path, characters_path)
                if not _script_cached:
                    _log(6, f"{Fore.GREEN}✓{Style.RESET_ALL} Script {ch_path.stem}")
                await script_queue.put(sp)
            await script_queue.put(None)

        # Stage 7 — consumer: synthesize chapters under Semaphore(2)
        audio_sem = asyncio.Semaphore(2)
        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        voices_dir = output_dir / "voices"
        audio_files: list[Path] = []
        voice_hashes: dict[str, str] = {}

        def _prepopulate_voice_hashes(speakers: set[str]) -> None:
            for s in speakers:
                if s not in voice_hashes:
                    ref = voices_dir / f"{s}.wav"
                    if ref.exists():
                        voice_hashes[s] = get_file_hash(ref)

        async def synth_one(script_path: Path) -> None:
            nonlocal _synth_counter
            speakers_needed = read_script_speakers(script_path)
            speakers_needed.add("narrator")
            _prepopulate_voice_hashes(speakers_needed)
            _log(7, f"Audio {script_path.stem}: synthesizing...")
            async with audio_sem:
                wav = await asyncio.to_thread(
                    synthesize_chapter,
                    script_path,
                    audio_dir,
                    voices_dir,
                    voices_config,
                    voice_hashes,
                )
                _synth_counter += 1
                _log(7, f"{Fore.GREEN}✓{Style.RESET_ALL} Audio {_synth_counter}/{_total_chapters}: {script_path.stem}")
                audio_files.append(wav)

        async def script_consumer() -> None:
            synth_tasks: list[asyncio.Task] = []
            while True:
                sp = await script_queue.get()
                if sp is None:
                    break
                synth_tasks.append(asyncio.create_task(synth_one(sp)))
            if synth_tasks:
                await asyncio.gather(*synth_tasks)

        _parallel_start = time.monotonic()
        await asyncio.gather(script_producer(), script_consumer())
        print_done(f"Synthesized {len(audio_files)} chapter audio files", stage=6)

        # Stage 8 — assemble M4B
        print_step(8, "Assemble M4B audiobook")
        m4b_path = assemble_m4b(output_dir, title, author)
        print_done(f"Created {m4b_path.name}", stage=8)

        total = _fmt_elapsed(since=_pipeline_start)
        print(f"\n{Fore.GREEN}Complete!{Style.RESET_ALL} Audiobook: {m4b_path}{total}")
        return m4b_path

    except (KeyboardInterrupt, asyncio.CancelledError):
        print(f"\n{Fore.YELLOW}⚠ Interrupted — exiting gracefully.{Style.RESET_ALL}")
        return output_dir / "INTERRUPTED"


# ##################################################################
# run pipeline
# synchronous wrapper for run_pipeline_async
def run_pipeline(
    epub_path: Path,
    max_chapters: int | None = None,
    voice_overrides_path: Path | None = None,
) -> Path:
    try:
        return asyncio.run(run_pipeline_async(
            epub_path, max_chapters=max_chapters, voice_overrides_path=voice_overrides_path,
        ))
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}⚠ Interrupted — exiting gracefully.{Style.RESET_ALL}")
        return get_output_dir(epub_path) / "INTERRUPTED"
