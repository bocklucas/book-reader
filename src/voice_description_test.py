import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.voice_description import (
    generate_voice_descriptions_async,
    generate_voices,
    sanitize_description,
)


# ##################################################################
# helper: build a fake embeddings db
# returns a MagicMock with search() returning the provided chunks
def _fake_db(search_results: list) -> MagicMock:
    db = MagicMock()
    # search returns list[tuple[dict, float]] per the real api
    db.search.return_value = search_results
    return db


# ##################################################################
# sanitize_description: drops forbidden tags, keeps allowed ones
def test_sanitize_description_strips_forbidden_tags() -> None:
    result = sanitize_description("female, low pitch, gravelly, british accent")
    assert "female" in result
    assert "low pitch" in result
    assert "british accent" in result
    assert "gravelly" not in result


# ##################################################################
# sanitize_description: enforces one per category
def test_sanitize_description_one_per_category() -> None:
    result = sanitize_description("male, female, low pitch, high pitch")
    tags = [t.strip() for t in result.split(",")]
    # only one gender and one pitch should remain
    assert tags.count("male") + tags.count("female") == 1
    assert tags.count("low pitch") + tags.count("high pitch") == 1


# ##################################################################
# sanitize_description: empty / garbage input -> fallback
def test_sanitize_description_fallback_on_garbage() -> None:
    assert sanitize_description("totally not a tag, nonsense") == "male, moderate pitch"
    assert sanitize_description("") == "male, moderate pitch"


# ##################################################################
# generate_voice_descriptions_async: every character gets a sanitized description
def test_generate_voice_descriptions_all_keys_sanitized() -> None:
    characters = {
        "john": {"name": "John Smith", "bio": "tall man, forties, dark hair"},
        "mary": {"name": "Mary Jones", "bio": "young woman, witty"},
        "narrator": {"name": "Narrator", "bio": "audiobook narrator"},
    }
    # llm response contains a forbidden tag that must be stripped
    fake_resp = "female, low pitch, gravelly, british accent"
    fake_search = [
        ({"chapter": "ch1.txt", "chunk_index": 0, "text": "John walked in slowly."}, 0.9),
    ]
    db = _fake_db(fake_search)

    with patch(
        "src.voice_description.query_small",
        new=AsyncMock(return_value=fake_resp),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            voices = asyncio.run(
                generate_voice_descriptions_async(characters, db, Path(tmp))
            )

    assert set(voices.keys()) == {"john", "mary", "narrator"}
    # forbidden tag stripped from each non-narrator
    for cid in ("john", "mary"):
        desc = voices[cid]["description"]
        assert "gravelly" not in desc
        assert "female" in desc
        assert "british accent" in desc
    # narrator gets the fixed default tag set (no rag, no llm call needed)
    narrator_desc = voices["narrator"]["description"]
    assert "male" in narrator_desc
    assert "moderate pitch" in narrator_desc


# ##################################################################
# generate_voice_descriptions_async: empty rag -> falls back to bio context
def test_generate_voice_descriptions_empty_rag_falls_back_to_bio() -> None:
    characters = {
        "alex": {"name": "Alex Doe", "bio": "elderly gentleman with a british accent"},
    }
    db = _fake_db([])  # no passages

    captured_prompts: list[str] = []

    async def fake_query(prompt: str, **kwargs) -> str:
        captured_prompts.append(prompt)
        return "male, elderly, british accent"

    with patch("src.voice_description.query_small", new=AsyncMock(side_effect=fake_query)):
        with tempfile.TemporaryDirectory() as tmp:
            voices = asyncio.run(
                generate_voice_descriptions_async(characters, db, Path(tmp))
            )

    # the bio should appear in the prompt because rag returned nothing
    assert captured_prompts, "query_small should have been called"
    assert "elderly gentleman with a british accent" in captured_prompts[0]
    desc = voices["alex"]["description"]
    assert "elderly" in desc
    assert "british accent" in desc


# ##################################################################
# generate_voice_descriptions_async: search filter targets first name lowercased
def test_generate_voice_descriptions_filter_by_first_name() -> None:
    characters = {
        "jane": {"name": "Jane Doe", "bio": "a clever investigator"},
    }
    db = _fake_db([
        ({"chapter": "ch1.txt", "chunk_index": 0, "text": "Jane spoke softly."}, 0.8),
    ])

    with patch(
        "src.voice_description.query_small",
        new=AsyncMock(return_value="female, moderate pitch"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            asyncio.run(generate_voice_descriptions_async(characters, db, Path(tmp)))

    # verify search was called with a filter_func that accepts "jane" in text
    assert db.search.called
    _, kwargs = db.search.call_args
    filter_func = kwargs["filter_func"]
    assert filter_func({"text": "Jane laughed."}) is True
    assert filter_func({"text": "Someone else spoke."}) is False


# ##################################################################
# generate_voices: writes voices.json and is idempotent
def test_generate_voices_writes_and_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "out"
        output_dir.mkdir(parents=True)
        characters = {"john": {"name": "John", "bio": "a tall man"}}
        (output_dir / "characters.json").write_text(json.dumps(characters), encoding="utf-8")

        db = _fake_db([
            ({"chapter": "ch1.txt", "chunk_index": 0, "text": "John walked in."}, 0.9),
        ])

        with patch(
            "src.voice_description.query_small",
            new=AsyncMock(return_value="male, middle-aged, moderate pitch"),
        ):
            result_path = asyncio.run(generate_voices(output_dir, db))

        assert result_path.exists()
        voices = json.loads(result_path.read_text(encoding="utf-8"))
        assert "john" in voices
        assert voices["john"]["description"] == "male, middle-aged, moderate pitch"

        # second call should be a no-op (idempotent)
        original = result_path.read_text(encoding="utf-8")
        # overwrite with sentinel; if generate_voices re-runs it would clobber.
        sentinel = '{"preserved": true}'
        result_path.write_text(sentinel, encoding="utf-8")
        with patch(
            "src.voice_description.query_small",
            new=AsyncMock(return_value="female, high pitch"),
        ):
            asyncio.run(generate_voices(output_dir, db))
        assert result_path.read_text(encoding="utf-8") == sentinel
        # restore for tidiness
        result_path.write_text(original, encoding="utf-8")
