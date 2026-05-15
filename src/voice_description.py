import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.llm_client import query_small

if TYPE_CHECKING:
    from src.embeddings_db import EmbeddingsDB


# ##################################################################
# sanitize description
# ensure only one attribute per category is present to avoid api conflicts.
# falls back to "male, moderate pitch" when no allowed tag is found.
def sanitize_description(text: str) -> str:
    categories = {
        "gender": ["male", "female"],
        "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
        "pitch": ["very low pitch", "low pitch", "moderate pitch", "high pitch", "very high pitch"],
        "style": ["whisper"],
        "accent": [
            "american accent", "australian accent", "british accent", "canadian accent",
            "chinese accent", "indian accent", "japanese accent", "korean accent",
            "portuguese accent", "russian accent",
        ],
        "dialect": [],
    }

    # flatten all allowed keywords for quick lookup
    all_allowed = []
    keyword_to_category = {}
    for cat, keywords in categories.items():
        for k in keywords:
            all_allowed.append(k)
            keyword_to_category[k] = cat

    # parse tags from input string
    tags = [t.strip().lower() for t in text.split(",") if t.strip()]

    # keep only the first occurrence for each category
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
# build voice prompt
# constructs the per-character omnivoice tag extraction prompt
def _build_voice_prompt(name: str, bio: str, context: str) -> str:
    return f"""Create OmniVoice TTS voice tags for the following character, grounded in the passages from the book.

Character: {name}
Bio: {bio}

Passages from the book mentioning this character:
{context}

Produce a comma-separated list of 2-5 voice attributes from this EXACT list:
- Gender: male, female
- Age: child, teenager, young adult, middle-aged, elderly
- Pitch: very low pitch, low pitch, moderate pitch, high pitch, very high pitch
- Style: whisper
- Accent: american accent, australian accent, british accent, canadian accent, chinese accent, indian accent, japanese accent, korean accent, portuguese accent, russian accent

Return ONLY the comma-separated list. No prose."""


# ##################################################################
# rag passages for character
# fetches top-k passages mentioning the character's first name. returns "" if none.
def _rag_passages_for_character(
    embeddings_db: "EmbeddingsDB",
    character_display_name: str,
    top_k: int = 5,
) -> str:
    first_name = character_display_name.split()[0].lower() if character_display_name.strip() else ""
    query = (
        f"Physical description, voice, speech pattern, dialogue, age, and background "
        f"of {character_display_name}"
    )

    def _filter(chunk: dict) -> bool:
        if not first_name:
            return True
        return first_name in chunk.get("text", "").lower()

    results = embeddings_db.search(query, top_k=top_k, filter_func=_filter)

    # embeddings_db.search returns list[tuple[dict, float]]; pull chunks out.
    passages = []
    for item in results:
        if isinstance(item, tuple):
            chunk = item[0]
        else:
            chunk = item
        passages.append(chunk.get("text", ""))

    return "\n\n---\n\n".join(p for p in passages if p)


# ##################################################################
# generate one voice description
# rag + llm + sanitize for a single character
async def _generate_one_voice_description(
    char_id: str,
    char_data: dict,
    embeddings_db: "EmbeddingsDB",
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict]:
    async with semaphore:
        name = char_data.get("name", char_id)
        bio = char_data.get("bio", "")

        # narrator is special: no rag, fixed neutral tags.
        if char_id == "narrator":
            return char_id, {"description": "male, middle-aged, moderate pitch"}

        context = _rag_passages_for_character(embeddings_db, name)
        if not context:
            # fall back to bio-only context
            context = bio or "(no passages available)"

        prompt = _build_voice_prompt(name, bio, context)
        tags_raw = await query_small(prompt, enable_thinking=False)
        tags = sanitize_description(tags_raw)
        return char_id, {"description": tags}


# ##################################################################
# generate voice descriptions async
# per-character rag flow with bounded concurrency
async def generate_voice_descriptions_async(
    characters: dict,
    embeddings_db: "EmbeddingsDB",
    output_dir: Path,
) -> dict:
    """For each character in `characters`, retrieve top-5 passages via RAG,
    ask query_small to extract OmniVoice tags grounded in those passages,
    sanitize, and return the voices dict ready to write as voices.json."""
    semaphore = asyncio.Semaphore(5)
    tasks = [
        _generate_one_voice_description(char_id, char_data, embeddings_db, semaphore)
        for char_id, char_data in characters.items()
    ]
    results = await asyncio.gather(*tasks)
    voices: dict = {}
    for char_id, entry in results:
        voices[char_id] = entry
    return voices


# ##################################################################
# generate voices
# main entry point to create voices.json from characters.json using rag
def load_voice_overrides(overrides_path: Path | None) -> dict:
    if overrides_path is None or not overrides_path.exists():
        return {}
    return json.loads(overrides_path.read_text(encoding="utf-8"))


async def generate_voices(
    output_dir: Path,
    embeddings_db: "EmbeddingsDB",
    voice_overrides_path: Path | None = None,
) -> Path:
    characters_path = output_dir / "characters.json"
    voices_path = output_dir / "voices.json"
    overrides = load_voice_overrides(voice_overrides_path)
    if not characters_path.exists():
        raise ValueError("characters.json not found")
    characters = json.loads(characters_path.read_text(encoding="utf-8"))

    voices = {}
    if voices_path.exists():
        voices = json.loads(voices_path.read_text(encoding="utf-8"))
    
    # Identify missing characters
    missing_chars = {cid: cdata for cid, cdata in characters.items() if cid not in voices}
    
    if missing_chars:
        new_voices = await generate_voice_descriptions_async(missing_chars, embeddings_db, output_dir)
        voices.update(new_voices)
        voices_path.write_text(json.dumps(voices, indent=2), encoding="utf-8")

    # Always apply overrides
    if overrides:
        changed = False
        for char_id, override in overrides.items():
            if voices.get(char_id) != override:
                voices[char_id] = override
                changed = True
        if changed:
            voices_path.write_text(json.dumps(voices, indent=2), encoding="utf-8")

    return voices_path


# ##################################################################
# generate voices sync
# synchronous wrapper for generate_voices
def generate_voices_sync(output_dir: Path, embeddings_db: "EmbeddingsDB") -> Path:
    return asyncio.run(generate_voices(output_dir, embeddings_db))
