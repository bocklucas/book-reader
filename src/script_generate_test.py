import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.script_generate import (
    CHUNK_MAX_CHARS,
    CONTEXT_TRAILING_CHARS,
    JSONL_CONTINUATION_PROMPT,
    build_chunk_prompt,
    generate_all_scripts,
    generate_chapter_script,
    generate_scripts_sync,
    parse_jsonl_response,
    split_into_chunks,
)


# ##################################################################
# helper: async mock returning queued responses
class QueuedResponder:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    async def __call__(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("QueuedResponder ran out of responses")
        return self.responses.pop(0)


# ##################################################################
# test parse jsonl response plain
def test_parse_jsonl_response_plain() -> None:
    plain = '{"speaker": "narrator", "text": "Hello"}\n{"speaker": "john", "text": "Hi"}'
    result = parse_jsonl_response(plain)
    assert len(result) == 2
    assert result[0] == {"speaker": "narrator", "text": "Hello"}
    assert result[1] == {"speaker": "john", "text": "Hi"}


# ##################################################################
# test parse jsonl response markdown
def test_parse_jsonl_response_markdown() -> None:
    markdown = '```jsonl\n{"speaker": "narrator", "text": "Hello"}\n{"speaker": "john", "text": "Hi"}\n```'
    result = parse_jsonl_response(markdown)
    assert len(result) == 2


# ##################################################################
# test parse jsonl response trailing junk
# tolerates preamble, blank lines, and broken trailing lines
def test_parse_jsonl_response_trailing_junk() -> None:
    raw = (
        "Sure, here you go:\n"
        '{"speaker": "narrator", "text": "Opening line"}\n'
        "\n"
        '{"speaker": "mary", "text": "Hi there"}\n'
        '{"speaker": "broken'  # truncated/invalid trailing line
    )
    result = parse_jsonl_response(raw)
    assert len(result) == 2
    assert result[0]["speaker"] == "narrator"
    assert result[1]["speaker"] == "mary"


# ##################################################################
# test parse jsonl legacy single-key form
def test_parse_jsonl_response_legacy_form() -> None:
    raw = '{"narrator": "Hello"}\n{"john": "Hi"}'
    result = parse_jsonl_response(raw)
    # legacy single-key dicts are passed through; downstream normalizes them
    assert len(result) == 2
    assert "narrator" in result[0] or result[0].get("speaker") == "narrator"


# ##################################################################
# test split into chunks short
def test_split_into_chunks_short() -> None:
    text = "short bit of text"
    chunks = split_into_chunks(text, chunk_max=CHUNK_MAX_CHARS)
    assert chunks == [text]


# ##################################################################
# test split into chunks long
def test_split_into_chunks_long() -> None:
    paragraph = ("This is a sentence. " * 40).strip()  # ~800 chars
    text = "\n\n".join([paragraph] * 12)  # ~10k chars
    chunks = split_into_chunks(text, chunk_max=CHUNK_MAX_CHARS)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= CHUNK_MAX_CHARS


# ##################################################################
# test build chunk prompt first vs continuation
def test_build_chunk_prompt_first_includes_start_directive() -> None:
    prompt = build_chunk_prompt(
        "Hello world.", "Chapter One", ["narrator", "john"], is_first_chunk=True
    )
    assert "Start with" in prompt
    assert "previous lines" not in prompt.lower()


def test_build_chunk_prompt_continuation_includes_prior_context() -> None:
    prior = "earlier source text snippet"
    prompt = build_chunk_prompt(
        "next chunk text",
        "Chapter One",
        ["narrator", "john"],
        is_first_chunk=False,
        prior_context=prior,
    )
    assert "PREVIOUS LINES" in prompt
    assert prior in prompt
    assert "do not re-emit" in prompt


# ##################################################################
# fixture: tmp output dir with chapters and characters.json
def _make_output_dir(tmpdir: Path, chapter_text: str) -> Path:
    output_dir = tmpdir / "output"
    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True)
    chapter_path = chapters_dir / "01-chapter_one.txt"
    chapter_path.write_text(chapter_text, encoding="utf-8")
    characters = {
        "john": {"name": "John", "bio": "A man"},
        "mary": {"name": "Mary", "bio": "A woman"},
    }
    (output_dir / "characters.json").write_text(json.dumps(characters), encoding="utf-8")
    return output_dir


# ##################################################################
# test single-chunk chapter
@pytest.mark.asyncio
async def test_generate_chapter_script_single_chunk() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output_dir = _make_output_dir(tmp, "John walked in. \"Hi,\" said Mary.")
        chapter_path = output_dir / "chapters" / "01-chapter_one.txt"
        characters_path = output_dir / "characters.json"

        responder = QueuedResponder([
            '{"speaker": "narrator", "text": "Chapter One"}\n'
            '{"speaker": "narrator", "text": "John walked in."}\n'
            '{"speaker": "mary", "text": "Hi"}\n'
        ])
        with patch("src.script_generate.query_large", responder):
            script_path = await generate_chapter_script(chapter_path, characters_path)

        assert len(responder.calls) == 1
        # verify continuation params passed correctly
        kwargs = responder.calls[0]["kwargs"]
        assert kwargs["max_continuations"] == 5
        assert kwargs["continuation_prompt"] == JSONL_CONTINUATION_PROMPT
        # verify output file
        assert script_path.exists()
        assert script_path.parent.name == "scripts"
        lines = script_path.read_text().strip().split("\n")
        assert len(lines) == 3
        first = json.loads(lines[0])
        assert first == {"speaker": "narrator", "text": "Chapter One"}
        last = json.loads(lines[2])
        assert last["speaker"] == "mary"


# ##################################################################
# test multi-chunk chapter with trailing context
@pytest.mark.asyncio
async def test_generate_chapter_script_multi_chunk_with_context() -> None:
    # build a chapter that definitely exceeds CHUNK_MAX_CHARS
    para = ("This is a paragraph with several sentences. " * 20).strip()  # ~880 chars
    big_text = "\n\n".join([para] * 10)  # ~8.8k chars -> 3+ chunks
    assert len(big_text) > CHUNK_MAX_CHARS * 2

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output_dir = _make_output_dir(tmp, big_text)
        chapter_path = output_dir / "chapters" / "01-chapter_one.txt"
        characters_path = output_dir / "characters.json"

        # one response per expected chunk (compute via split)
        chunks = split_into_chunks(big_text, CHUNK_MAX_CHARS)
        n_chunks = len(chunks)
        assert n_chunks >= 2
        responses = [
            f'{{"speaker": "narrator", "text": "chunk {i} line"}}\n'
            for i in range(n_chunks)
        ]
        responder = QueuedResponder(responses)
        with patch("src.script_generate.query_large", responder):
            script_path = await generate_chapter_script(chapter_path, characters_path)

        assert len(responder.calls) == n_chunks
        # first call: no PREVIOUS LINES block
        assert "PREVIOUS LINES" not in responder.calls[0]["prompt"]
        # second call: must include trailing slice of chunk[0]
        second_prompt = responder.calls[1]["prompt"]
        assert "PREVIOUS LINES" in second_prompt
        expected_context = chunks[0][-CONTEXT_TRAILING_CHARS:]
        # take a stable substring near the tail to avoid whitespace differences
        tail_sample = expected_context[-200:]
        assert tail_sample in second_prompt
        # all continuation params set on every call
        for call in responder.calls:
            assert call["kwargs"]["max_continuations"] == 5
            assert call["kwargs"]["continuation_prompt"] == JSONL_CONTINUATION_PROMPT
        # output concatenates lines across chunks
        out_lines = script_path.read_text().strip().split("\n")
        assert len(out_lines) == n_chunks


# ##################################################################
# test no characters.json mutation (no cast-discovery side effects)
@pytest.mark.asyncio
async def test_generate_chapter_script_does_not_mutate_characters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output_dir = _make_output_dir(tmp, "John walked in.")
        chapter_path = output_dir / "chapters" / "01-chapter_one.txt"
        characters_path = output_dir / "characters.json"
        original = characters_path.read_text(encoding="utf-8")

        responder = QueuedResponder([
            '{"speaker": "narrator", "text": "Hi"}\n'
            '{"speaker": "stranger", "text": "boo"}\n'  # speaker not in cast
        ])
        with patch("src.script_generate.query_large", responder):
            await generate_chapter_script(chapter_path, characters_path)
        assert characters_path.read_text(encoding="utf-8") == original


# ##################################################################
# test intro chapter emits verbatim narrator line, no LLM call
@pytest.mark.asyncio
async def test_generate_chapter_script_intro_no_llm() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output_dir = tmp / "output"
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(parents=True)
        intro = chapters_dir / "00-intro.txt"
        intro.write_text("Test Book by Author.", encoding="utf-8")
        characters_path = output_dir / "characters.json"
        characters_path.write_text(json.dumps({"narrator": {"name": "Narrator", "bio": ""}}), encoding="utf-8")

        responder = QueuedResponder([])  # should not be called
        with patch("src.script_generate.query_large", responder):
            script_path = await generate_chapter_script(intro, characters_path)
        assert len(responder.calls) == 0
        line = json.loads(script_path.read_text().strip())
        assert line == {"speaker": "narrator", "text": "Test Book by Author."}


# ##################################################################
# test generate_all_scripts skips existing files
def test_generate_scripts_sync_idempotent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        output_dir = _make_output_dir(tmp, "John walked in.")
        # pre-create the expected output to be preserved
        script_dir = output_dir / "scripts"
        script_dir.mkdir(parents=True, exist_ok=True)
        existing = script_dir / "01-chapter_one.jsonl"
        existing.write_text("PRESERVED", encoding="utf-8")

        responder = QueuedResponder([])  # zero responses; must not be called
        with patch("src.script_generate.query_large", responder):
            scripts = generate_scripts_sync(output_dir)
        assert len(responder.calls) == 0
        assert existing.read_text(encoding="utf-8") == "PRESERVED"
        assert existing in scripts
