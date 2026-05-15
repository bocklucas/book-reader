![](banner.jpg)

```markdown
# Book Reader

Convert EPUB books into M4B audiobooks with distinct voices for each character using local AI — no cloud APIs required.

## Overview

Book Reader is a command-line pipeline that takes an EPUB file and produces a fully chaptered M4B audiobook. Each character in the book is assigned a unique synthesised voice via a local LLM (llama.cpp) and a local TTS engine (OmniVoice). The output includes chapter markers, a cover image, and chime announcements between chapters.

The pipeline is fully async with streaming producer/consumer stages: script generation and audio synthesis run concurrently so audio rendering begins as soon as the first chapter script is ready.

## Requirements

- Python 3.10+
- `ffmpeg` available on your PATH
- A running [llama.cpp](https://github.com/ggerganov/llama.cpp) server (OpenAI-compatible `/v1/chat/completions`)
- A running [OmniVoice-FastAPI](https://github.com/^\*) TTS server

### Tested LLM Configuration

```bash
llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS \
  -ngl 99 -c 65536 -np 1 -fa on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --host 0.0.0.0 --port 11435 \
  -t 12 --chat-template-kwargs '{"preserve_thinking": true}'
```

## Installation

### Docker (recommended)

```bash
git clone <repo-url>
cd book-reader
cp .env.example .env   # edit with your LLM/TTS server URLs
mkdir -p input output
```

Place your EPUB files in the `input/` directory, then run:

```bash
docker compose run --rm book-reader create input/MyBook.epub
```

Output will appear in `output/`. The container uses `network_mode: host` so `localhost` URLs in `.env` reach services running on the host machine.

Other commands work the same way:

```bash
docker compose run --rm book-reader create input/MyBook.epub --max-chapters 5
```

To rebuild the image after pulling updates:

```bash
docker compose build
```

### Local (without Docker)

Clone the repository and install dependencies into a local virtual environment:

```bash
git clone <repo-url>
cd book-reader
./run install
```

This creates a `.venv` directory and installs all Python dependencies automatically.

Copy `.env.example` to `.env` and adjust the URLs/models for your setup:

```bash
cp .env.example .env
```

## Pipeline Stages

The conversion process runs as a sequence of async stages:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Extract | Extracts chapter text from the EPUB file |
| 2 | Characters | Analyses each chapter to identify speaking characters (chunked, with deterministic deduplication) |
| 3 | Embeddings | Builds a semantic embeddings database for RAG-powered voice description |
| 4 | Voices | Generates voice descriptions for each character using RAG context |
| 5 | Clone | Generates reference voice WAVs via OmniVoice TTS-design |
| 6 | Scripts + Audio | Streaming producer/consumer — generates speaker-attributed scripts and synthesises audio concurrently |
| 7 | M4B | Assembles all audio into a final M4B file with chapter markers |

Every stage uses content-hash caching — re-running skips work whose inputs haven't changed.

## Usage

### Convert an entire book

```bash
./run create "My Book.epub"
```

### Convert only the first N chapters

```bash
./run create "My Book.epub" --max-chapters 5
```

### Override models or keep both loaded

```bash
./run create "My Book.epub" --small-model qwen3-4b --large-model qwen3-35b --keep-models-loaded
```

### Supply voice overrides

```bash
./run create "My Book.epub" --voice-overrides voice_overrides.json
```

### Run tests

```bash
./run test src/
./run test src/epub_extract_test.py::test_normalize_name
```

### Lint the source

```bash
./run lint
```

## Examples

**Full conversion of a single EPUB:**

```bash
./run create "John Doe - The Galactic Odyssey.epub"
```

Output is written to `output/John Doe - The Galactic Odyssey/`.

**Previewing the first five chapters before committing to a full run:**

```bash
./run create "John Doe - The Galactic Odyssey.epub" --max-chapters 5
```

**Resuming a partially completed pipeline:**

Every stage uses content-hash caching. Simply re-run — completed work is not repeated:

```bash
./run create "My Book.epub"
```

## Configuration

All configuration is via environment variables (or `.env` file). See `.env.example` for the full list:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMACPP_BASE_URL` | `http://localhost:8080/v1` | llama.cpp server URL |
| `LLAMACPP_SMALL_MODEL` | (from `LLAMACPP_MODEL`) | Model for fast tasks (extraction, character analysis) |
| `LLAMACPP_LARGE_MODEL` | (from `LLAMACPP_MODEL`) | Model for quality tasks (script generation) |
| `MODEL_SWAP_URL` | _(none)_ | Optional model-swap proxy (e.g. llama-swap) |
| `KEEP_MODELS_LOADED` | `false` | Keep both models resident in memory |
| `OMNIVOICE_BASE_URL` | `http://localhost:8880/v1` | OmniVoice TTS server URL |
| `OMNIVOICE_SPEED` | `1.0` | Speech speed multiplier |

## Output Structure

```
output/
└── <epub-stem>/
    ├── characters.json      # Character profiles with chapter associations
    ├── voices.json          # TTS voice descriptions
    ├── cover.jpeg           # Cover art extracted from EPUB
    ├── chunks.json          # Embeddings DB chunks
    ├── embeddings.npy       # Embeddings DB vectors
    ├── chapters/            # Extracted chapter text files
    ├── scripts/             # Speaker-attributed JSONL scripts
    ├── voices/              # Reference voice WAV files per character
    ├── audio/               # Synthesised WAV files per chapter
    └── <stem>.m4b           # Final audiobook
```

## Architecture

### LLM Backend

The project uses a local llama.cpp server via its OpenAI-compatible API. Two model tiers are supported (`small` for fast tasks like character extraction, `large` for script generation). An optional model-swap proxy can hot-swap models between stages on single-GPU setups.

The LLM client streams responses, supports automatic continuation on truncation, and strips `<think>` blocks from reasoning models.

### TTS Backend

Voice synthesis uses OmniVoice-FastAPI with two modes:
- **Design** — generate a reference voice from a text description (used once per character)
- **Clone** — synthesise speech using a reference voice WAV (used per script line)

### Script Generation

Script generation uses a deterministic 3-stage decomposed pipeline:
1. **Slice** — regex-based splitting of text into dialogue, narration, and attribution tags (verbatim-preserving)
2. **Annotate** — LLM assigns speaker IDs to dialogue fragments
3. **Compile** — deterministic assembly into final JSONL

This avoids the hallucination and text-mutation issues of single-pass LLM script generation.

### Character Matching

Character deduplication uses a 3-tier reconciliation strategy:
1. **Deterministic** — normalized string matching (accent stripping, prefix removal, substring containment)
2. **Word-overlap + LLM confirmation** — soft signal from significant word overlap, confirmed by a focused YES/NO LLM query
3. **General LLM reconciliation** — fallback for ambiguous cases

### Caching

Every stage uses content-addressed hashing. Inputs (file contents, descriptions, speed settings) are hashed and compared against stored hashes. Changed inputs trigger regeneration; unchanged inputs are skipped. Individual audio lines are hashed so re-runs only regenerate lines whose voice or text changed.

## Notes

- The M4B assembly step is skipped if the output `.m4b` already exists. Delete it manually to force a rebuild.
- All pipeline stages are idempotent — re-running is always safe.
- The `step` subcommand has been removed; use `./run create` with `--max-chapters` to limit scope.
```
