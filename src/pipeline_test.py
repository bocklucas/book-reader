import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src import pipeline as pipeline_mod
from src.pipeline import read_script_speakers, run_pipeline_async


# ##################################################################
# test read script speakers
# verify speaker extraction from JSONL script files
def test_read_script_speakers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sp = Path(tmpdir) / "ch.jsonl"
        sp.write_text(
            '{"speaker": "narrator", "text": "hello"}\n'
            '{"speaker": "alice", "text": "hi"}\n'
            '{"speaker": "narrator", "text": "she said"}\n'
            '{"speaker_id": "bob", "text": "yo"}\n'
        )
        assert read_script_speakers(sp) == {"narrator", "alice", "bob"}


# ##################################################################
# test pipeline ordering and streaming
# integration-style test verifying:
#   - stage ordering (embeddings finalize before voice_description)
#   - ensure_model("large") called before any generate_chapter_script
#   - all voice clones complete before any script generation
def test_pipeline_orchestration() -> None:
    events_log: list[str] = []
    ensure_model_calls: list[str] = []

    async def fake_extract_epub(epub_path: Path, output_dir: Path):
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "00-intro.txt").write_text("Test Book by Test Author.")
        (chapters_dir / "01-ch1.txt").write_text("Alice spoke to Bob.\n\nHello world.")
        events_log.append("extract")
        return "Test Book", "Test Author", [
            chapters_dir / "00-intro.txt",
            chapters_dir / "01-ch1.txt",
        ]

    async def fake_ensure_model(tier: str) -> None:
        ensure_model_calls.append(tier)
        events_log.append(f"ensure_model:{tier}")

    async def fake_analyze_characters(output_dir: Path, title: str, author: str, voice_overrides_path=None) -> Path:
        events_log.append("analyze_characters")
        chars = {
            "narrator": {"name": "Narrator", "bio": "warm voice"},
            "alice": {"name": "Alice", "bio": "young woman"},
            "bob": {"name": "Bob", "bio": "older man"},
        }
        characters_path = output_dir / "characters.json"
        characters_path.write_text(json.dumps(chars))
        return characters_path

    fake_db_instance = MagicMock()
    fake_db_instance.add_chapter = MagicMock(
        side_effect=lambda *a, **kw: events_log.append("db.add_chapter")
    )
    fake_db_instance.finalize_and_save = MagicMock(
        side_effect=lambda: events_log.append("db.finalize_and_save")
    )

    def fake_db_ctor(output_dir: Path):
        events_log.append("db.__init__")
        return fake_db_instance

    async def fake_generate_voices(output_dir: Path, db, voice_overrides_path=None) -> Path:
        events_log.append("generate_voices")
        voices = {
            "narrator": {"description": "male, middle-aged, moderate pitch"},
            "alice": {"description": "female, young adult, high pitch"},
            "bob": {"description": "male, elderly, low pitch"},
        }
        voices_path = output_dir / "voices.json"
        voices_path.write_text(json.dumps(voices))
        return voices_path

    async def fake_clone_voice_async(name: str, description: str, output_dir: Path) -> Path:
        events_log.append(f"clone:{name}")
        voices_dir = output_dir / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        wav = voices_dir / f"{name}.wav"
        wav.write_bytes(b"x" * 200)
        return wav

    async def fake_generate_chapter_script(chapter_path: Path, characters_path: Path) -> Path:
        events_log.append(f"script:{chapter_path.name}")
        assert "large" in ensure_model_calls, (
            "generate_chapter_script invoked before ensure_model('large')"
        )
        output_dir = characters_path.parent
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        sp = scripts_dir / f"{chapter_path.stem}.jsonl"
        sp.write_text(
            '{"speaker": "narrator", "text": "intro line"}\n'
            '{"speaker": "alice", "text": "hello"}\n'
        )
        return sp

    def fake_synthesize_chapter(script_path, audio_dir, voices_dir, voices_config, voice_hashes):
        assert (voices_dir / "narrator.wav").exists(), (
            "narrator.wav missing when synthesize_chapter ran"
        )
        events_log.append(f"synth:{script_path.name}")
        wav = audio_dir / f"{script_path.stem}.wav"
        wav.write_bytes(b"y" * 200)
        return wav

    def fake_assemble_m4b(output_dir: Path, title: str, author: str, **kwargs) -> Path:
        events_log.append("assemble_m4b")
        m4b = output_dir / f"{output_dir.name}.m4b"
        m4b.write_bytes(b"m4b")
        return m4b

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        epub_path = tmpdir / "book.epub"
        epub_path.write_bytes(b"")
        output_dir = tmpdir / "out"

        with patch.object(pipeline_mod, "extract_epub", side_effect=fake_extract_epub), \
             patch.object(pipeline_mod, "ensure_model", new=AsyncMock(side_effect=fake_ensure_model)), \
             patch.object(pipeline_mod, "analyze_characters", new=AsyncMock(side_effect=fake_analyze_characters)), \
             patch.object(pipeline_mod, "EmbeddingsDB", side_effect=fake_db_ctor), \
             patch.object(pipeline_mod, "generate_voices", new=AsyncMock(side_effect=fake_generate_voices)), \
             patch.object(pipeline_mod, "clone_voice_async", new=AsyncMock(side_effect=fake_clone_voice_async)), \
             patch.object(pipeline_mod, "generate_chapter_script", new=AsyncMock(side_effect=fake_generate_chapter_script)), \
             patch.object(pipeline_mod, "synthesize_chapter", side_effect=fake_synthesize_chapter), \
             patch.object(pipeline_mod, "assemble_m4b", side_effect=fake_assemble_m4b):
            m4b = asyncio.run(run_pipeline_async(epub_path, output_dir=output_dir))

        assert m4b.exists()

        # Embeddings finalize before voice description
        assert "db.finalize_and_save" in events_log
        assert events_log.index("db.finalize_and_save") < events_log.index("generate_voices")

        # ensure_model("large") before any script generation
        large_event_idx = next(i for i, e in enumerate(events_log) if e == "ensure_model:large")
        first_script_idx = next(i for i, e in enumerate(events_log) if e.startswith("script:"))
        assert large_event_idx < first_script_idx

        assert ensure_model_calls.count("small") == 3
        assert ensure_model_calls.count("large") == 1

        # All clones complete before any script generation (stage 5 before stage 6)
        last_clone_idx = max(i for i, e in enumerate(events_log) if e.startswith("clone:"))
        assert last_clone_idx < first_script_idx, (
            "all voice clones must complete before script generation starts"
        )

        # Synth ran
        assert any(e.startswith("synth:") for e in events_log)


# ##################################################################
# test voice clone caching
# verify that pre-existing wav files skip clone_voice_async
def test_voice_clone_caching_skips_existing() -> None:
    events_log: list[str] = []

    async def fake_extract_epub(epub_path: Path, output_dir: Path):
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(parents=True, exist_ok=True)
        (chapters_dir / "00-intro.txt").write_text("Book by Author.")
        (chapters_dir / "01-ch1.txt").write_text("text")
        return "Book", "Author", [
            chapters_dir / "00-intro.txt",
            chapters_dir / "01-ch1.txt",
        ]

    async def fake_ensure_model(tier: str) -> None:
        pass

    async def fake_analyze_characters(output_dir: Path, title: str, author: str, voice_overrides_path=None) -> Path:
        chars = {
            "narrator": {"name": "Narrator", "bio": "x"},
            "alice": {"name": "Alice", "bio": "x"},
        }
        cp = output_dir / "characters.json"
        cp.write_text(json.dumps(chars))
        return cp

    fake_db = MagicMock()
    fake_db.add_chapter = MagicMock()
    fake_db.finalize_and_save = MagicMock()

    async def fake_generate_voices(output_dir: Path, db, voice_overrides_path=None) -> Path:
        voices = {
            "narrator": {"description": "n"},
            "alice": {"description": "a"},
        }
        vp = output_dir / "voices.json"
        vp.write_text(json.dumps(voices))
        return vp

    async def fake_clone_voice_async(name: str, description: str, output_dir: Path) -> Path:
        events_log.append(f"clone:{name}")
        voices_dir = output_dir / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        wav = voices_dir / f"{name}.wav"
        wav.write_bytes(b"x" * 200)
        return wav

    async def fake_generate_chapter_script(chapter_path: Path, characters_path: Path) -> Path:
        output_dir = characters_path.parent
        scripts_dir = output_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        sp = scripts_dir / f"{chapter_path.stem}.jsonl"
        sp.write_text('{"speaker": "narrator", "text": "intro"}\n')
        return sp

    def fake_synthesize_chapter(script_path, audio_dir, voices_dir, voices_config, voice_hashes):
        wav = audio_dir / f"{script_path.stem}.wav"
        wav.write_bytes(b"y")
        return wav

    def fake_assemble_m4b(output_dir: Path, title: str, author: str, **kwargs) -> Path:
        m4b = output_dir / "x.m4b"
        m4b.write_bytes(b"")
        return m4b

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        epub_path = tmpdir / "book.epub"
        epub_path.write_bytes(b"")
        output_dir = tmpdir / "out"
        # Pre-create narrator voice so it's cached
        voices_dir = output_dir / "voices"
        voices_dir.mkdir(parents=True, exist_ok=True)
        (voices_dir / "narrator.wav").write_bytes(b"x" * 200)

        with patch.object(pipeline_mod, "extract_epub", side_effect=fake_extract_epub), \
             patch.object(pipeline_mod, "ensure_model", new=AsyncMock(side_effect=fake_ensure_model)), \
             patch.object(pipeline_mod, "analyze_characters", new=AsyncMock(side_effect=fake_analyze_characters)), \
             patch.object(pipeline_mod, "EmbeddingsDB", return_value=fake_db), \
             patch.object(pipeline_mod, "generate_voices", new=AsyncMock(side_effect=fake_generate_voices)), \
             patch.object(pipeline_mod, "clone_voice_async", new=AsyncMock(side_effect=fake_clone_voice_async)), \
             patch.object(pipeline_mod, "generate_chapter_script", new=AsyncMock(side_effect=fake_generate_chapter_script)), \
             patch.object(pipeline_mod, "synthesize_chapter", side_effect=fake_synthesize_chapter), \
             patch.object(pipeline_mod, "assemble_m4b", side_effect=fake_assemble_m4b):
            asyncio.run(run_pipeline_async(epub_path, output_dir=output_dir))

    # Only alice should have been cloned; narrator was cached
    assert events_log == ["clone:alice"]
