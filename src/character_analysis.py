import asyncio
import json
import unicodedata
from pathlib import Path

from src.llm_client import query_small
from src.script_pipeline_utils import get_semantic_chunks


# ##################################################################
# parse json response
# extract json from claude response handling markdown code blocks and preamble
def parse_json_response(text: str) -> dict:
    text = text.strip()
    if not text:
        return {"characters": {}}
    if "```" in text:
        start = text.find("```")
        end = text.rfind("```")
        if start != end:
            block = text[start:end + 3]
            lines = block.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
    if not text.startswith("{"):
        brace_pos = text.find("{")
        if brace_pos != -1:
            text = text[brace_pos:]
            end_brace = text.rfind("}")
            if end_brace != -1:
                text = text[:end_brace + 1]
    if not text or not text.startswith("{"):
        return {"characters": {}}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"characters": {}}


# ##################################################################
# normalize for comparison
# strip accents, articles, and common prefixes for matching
def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.lower().strip()
    for prefix in ["the_", "don_", "dona_", "doña_", "mr_", "mrs_", "ms_", "dr_"]:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    return normalized


# ##################################################################
# analyze chapter
# extract character information from a single chapter (blind — no master list).
# chunks within the chapter are reconciled accumulatively so that the same
# character described with different IDs across chunks is merged.
async def analyze_chapter(chapter_path: Path, chapter_num: int) -> dict:
    text = chapter_path.read_text(encoding="utf-8")
    chunks = get_semantic_chunks(text, 15000)

    # accumulate characters across chunks within this chapter
    chapter_characters: dict = {}

    for i, chunk in enumerate(chunks):
        prompt = f"""Analyze this text and identify characters who speak or have internal monologue.

For each speaking character, extract ONLY voice-relevant physical details. ALWAYS include nationality/region and a specific age if there are ANY contextual clues (setting, era, vocabulary, place names, period detail) — these are the most important signals for voice. Do not default to "young" or "American" without evidence.

- Gender (from pronouns or descriptions)
- Age — be specific where possible (e.g. "early 30s", "around 50"). Only use "young/middle-aged/elderly" if no clue. War setting alone does NOT mean young.
- Nationality / regional accent (REQUIRED if any contextual evidence exists — WWI British, South African, Egyptian, Australian, etc. Use period and setting cues, not just accent words.)
- Physical build (large, small, thin, heavy, etc.)
- Voice/speech patterns (gruff, soft, educated, crude, accent, lisping, etc.)
- Distinctive physical traits affecting voice (old, frail, booming, wheezing, etc.)

Return ONLY valid JSON:
{{
  "characters": {{
    "character_id": {{
      "name": "Display Name",
      "details": "Physical and voice description only"
    }}
  }}
}}

Rules:
- Include ONLY characters who actually speak (quoted dialogue) or have internal monologue
- Do NOT include characters who are merely mentioned
- Character IDs: lowercase with underscores (e.g., "john_doe")
- Details must focus on VOICE generation - what would help create their voice
- EXCLUDE: plot roles, story function, relationships to other characters, emotional descriptions
- INCLUDE: "elderly man with gravelly voice" / "young woman, speaks formally" / "large man, booming voice" / "British soldier, ~30s, weary, working-class accent" / "South African woman, late 20s, clear professional tone"
- NO cross-character references (don't mention other characters in the details)

Chapter {chapter_num} text excerpt:
{chunk}"""

        response = await query_small(prompt, enable_thinking=False)
        chunk_result = parse_json_response(response)
        chunk_chars = chunk_result.get("characters", {})

        if not chunk_chars:
            continue

        # reconcile this chunk's characters against the chapter's running list
        for char_id, info in chunk_chars.items():
            details = info.get("details", info.get("bio", ""))
            name = info.get("name", char_id)

            if not chapter_characters:
                # first chunk — everything is new
                chapter_characters[char_id] = {"name": name, "details": details}
                continue

            matched_id = await reconcile_single_character(
                char_id, info, chapter_characters
            )
            if matched_id is not None:
                if details and details not in chapter_characters[matched_id]["details"]:
                    chapter_characters[matched_id]["details"] += " " + details
                if len(name) > len(chapter_characters[matched_id].get("name", "")):
                    chapter_characters[matched_id]["name"] = name
            else:
                chapter_characters[char_id] = {"name": name, "details": details}

    return {"characters": chapter_characters}


# ##################################################################
# deterministic match
# attempt to match a new character against the master list using string
# comparison only. returns the canonical ID if matched, None otherwise.
def deterministic_match(
    new_id: str,
    new_info: dict,
    master_characters: dict,
) -> str | None:
    new_norm = normalize_for_comparison(new_id)
    new_name = new_info.get("name", "")
    new_name_norm = normalize_for_comparison(new_name.replace(" ", "_"))

    for master_id, master_info in master_characters.items():
        master_norm = normalize_for_comparison(master_id)
        master_name = master_info.get("name", "")
        master_name_norm = normalize_for_comparison(master_name.replace(" ", "_"))

        # exact ID match
        if new_norm == master_norm:
            return master_id

        # exact name match (normalized)
        if new_name_norm and master_name_norm and new_name_norm == master_name_norm:
            return master_id

        # ID-to-name cross match: new_id matches master's name or vice versa
        if new_norm and master_name_norm and new_norm == master_name_norm:
            return master_id
        if new_name_norm and master_norm and new_name_norm == master_norm:
            return master_id

        # substring containment on IDs (but not trivially short)
        if len(new_norm) >= 3 and len(master_norm) >= 3:
            if new_norm in master_norm or master_norm in new_norm:
                return master_id

    return None


# ##################################################################
# find word overlap candidate
# find a single best candidate from the existing list based on significant
# word overlap. returns (candidate_id, candidate_info) or (None, None).
# this is a "soft" signal — the result must be confirmed by the LLM.
def find_word_overlap_candidate(
    new_id: str,
    new_info: dict,
    master_characters: dict,
) -> tuple[str | None, dict | None]:
    new_norm = normalize_for_comparison(new_id)
    new_name = new_info.get("name", "")
    new_words = {w for w in new_norm.split("_") if len(w) > 2}
    new_name_words = {w.lower() for w in new_name.split() if len(w) > 2}
    all_new_words = new_words | new_name_words

    if len(all_new_words) < 2:
        return None, None

    best_id = None
    best_info = None
    best_ratio = 0.0

    for master_id, master_info in master_characters.items():
        master_norm = normalize_for_comparison(master_id)
        master_name = master_info.get("name", "")
        master_words = {w for w in master_norm.split("_") if len(w) > 2}
        master_name_words = {w.lower() for w in master_name.split() if len(w) > 2}
        all_master_words = master_words | master_name_words

        if len(all_master_words) < 2:
            continue

        shorter = min(all_new_words, all_master_words, key=len)
        overlap = all_new_words & all_master_words
        ratio = len(overlap) / len(shorter) if shorter else 0.0

        if ratio >= 0.67 and ratio > best_ratio:
            best_ratio = ratio
            best_id = master_id
            best_info = master_info

    return best_id, best_info


# ##################################################################
# confirm character match via LLM
# ask a focused YES/NO question about whether two characters are the same.
# much simpler question than the general reconciliation prompt.
async def confirm_character_match_llm(
    new_id: str,
    new_info: dict,
    candidate_id: str,
    candidate_info: dict,
) -> bool:
    new_name = new_info.get("name", new_id)
    new_details = new_info.get("details", new_info.get("bio", ""))
    cand_name = candidate_info.get("name", candidate_id)
    cand_bio = candidate_info.get("bio", candidate_info.get("details", ""))

    prompt = f"""Are these two characters the SAME PERSON?

Character A: {new_name} — {new_details}
Character B: {cand_name} — {cand_bio}

Answer YES if they are the same person (same character, possibly referred to differently).
Answer NO if they are different people.

Answer with ONLY YES or NO:"""

    response = await query_small(prompt, enable_thinking=False)
    answer = response.strip().lower().split()[0] if response.strip() else "no"
    answer = answer.strip(".,;:!?'\"")
    return answer == "yes"


# ##################################################################
# reconcile single character
# full reconciliation flow for one character against an existing list:
# 1. deterministic match (hard) → instant
# 2. word-overlap candidate → LLM confirmation (soft)
# 3. general LLM reconciliation (fallback)
# returns the matching existing ID, or None if genuinely new.
async def reconcile_single_character(
    new_id: str,
    new_info: dict,
    existing_characters: dict,
) -> str | None:
    # step 1: deterministic match
    match = deterministic_match(new_id, new_info, existing_characters)
    if match is not None:
        return match

    # step 2: word-overlap candidate → LLM confirmation
    overlap_id, overlap_info = find_word_overlap_candidate(
        new_id, new_info, existing_characters
    )
    if overlap_id is not None:
        confirmed = await confirm_character_match_llm(
            new_id, new_info, overlap_id, overlap_info
        )
        if confirmed:
            return overlap_id

    # step 3: general LLM reconciliation against plausible candidates
    candidates = get_plausible_candidates(new_id, new_info, existing_characters)
    if not candidates:
        return None

    return await reconcile_character_llm(new_id, new_info, candidates)


# ##################################################################
# reconcile character via LLM
# ask the LLM a focused question: is this new character the same as
# any existing character? returns the matching master ID, or None if new.
async def reconcile_character_llm(
    new_id: str,
    new_info: dict,
    candidates: dict,
) -> str | None:
    if not candidates:
        return None

    new_name = new_info.get("name", new_id)
    new_details = new_info.get("details", "")

    candidate_lines = []
    for cand_id, cand_info in candidates.items():
        cand_name = cand_info.get("name", cand_id)
        cand_bio = cand_info.get("bio", cand_info.get("details", ""))
        if len(cand_bio) > 120:
            cand_bio = cand_bio[:117] + "..."
        candidate_lines.append(f"  {cand_id}: {cand_name} — {cand_bio}")

    candidates_str = "\n".join(candidate_lines)

    prompt = f"""Is this new character the SAME PERSON as any existing character below?

NEW CHARACTER:
  ID: {new_id}
  Name: {new_name}
  Details: {new_details}

EXISTING CHARACTERS:
{candidates_str}

Rules:
- Answer with ONLY the matching existing character ID if they are definitely the same person
- Answer with ONLY the word NONE if this is a new, different character
- Characters are the same person ONLY if they share the same name or are clearly the same individual
- Different people with similar roles are NOT the same (e.g. two soldiers, two children)
- When in doubt, answer NONE — it is always safer to treat someone as new

Answer:"""

    response = await query_small(prompt, enable_thinking=False)
    answer = response.strip().lower().strip('"').strip("'").strip()

    # extract just the first word/ID from the response
    answer = answer.split("\n")[0].strip()
    answer = answer.split(" ")[0].strip()
    answer = answer.rstrip(".,;:")

    if answer == "none" or not answer:
        return None

    # verify the answer is actually a valid candidate ID
    if answer in candidates:
        return answer

    # try normalized match against candidate IDs
    answer_norm = normalize_for_comparison(answer)
    for cand_id in candidates:
        if normalize_for_comparison(cand_id) == answer_norm:
            return cand_id

    # LLM returned something we don't recognize — treat as new
    return None


# ##################################################################
# get plausible candidates
# filter the master list to characters that could plausibly match the
# new character, to keep the LLM prompt focused
def get_plausible_candidates(
    new_id: str,
    new_info: dict,
    master_characters: dict,
) -> dict:
    # for local LLMs with limited reasoning, we only send plausible
    # candidates. if the master list is small enough, send all of them.
    if len(master_characters) <= 10:
        return dict(master_characters)

    # otherwise, filter to characters with some textual overlap
    new_name = new_info.get("name", "").lower()
    new_details = new_info.get("details", "").lower()
    new_norm = normalize_for_comparison(new_id)
    new_words = set(new_norm.split("_")) | set(new_name.split())

    candidates = {}
    for master_id, master_info in master_characters.items():
        master_name = master_info.get("name", "").lower()
        master_norm = normalize_for_comparison(master_id)
        master_words = set(master_norm.split("_")) | set(master_name.split())

        # any shared meaningful word (length > 2) between IDs/names
        shared = new_words & master_words
        shared = {w for w in shared if len(w) > 2}
        if shared:
            candidates[master_id] = master_info
            continue

        # any name appears in the other's details
        if new_name and new_name in master_info.get("bio", master_info.get("details", "")).lower():
            candidates[master_id] = master_info
            continue
        if master_name and master_name in new_details:
            candidates[master_id] = master_info

    return candidates


# ##################################################################
# reconcile chapter characters
# given characters extracted from one chapter and the current master list,
# determine which are existing characters and which are genuinely new.
# returns a dict mapping canonical_id -> character info with "details" key.
async def reconcile_chapter_characters(
    chapter_chars: dict,
    master_characters: dict,
    chapter_stem: str,
) -> dict:
    result = {}

    for new_id, new_info in chapter_chars.items():
        matched_id = await reconcile_single_character(
            new_id, new_info, master_characters
        )
        if matched_id is not None:
            result[matched_id] = new_info
            print(f"    Matched '{new_id}' → existing '{matched_id}'")
        else:
            result[new_id] = new_info
            print(f"    New character: '{new_id}'")

    return result


# ##################################################################
# create narrator entry
# generate narrator character based on book metadata and tone
async def create_narrator_entry(title: str, author: str, sample_text: str) -> dict:
    prompt = f"""Based on this book's title, author, and sample text, describe the ideal narrator.

Book: "{title}" by {author}

Sample text:
{sample_text[:3000]}

The narrator should have:
- Clarity and authority as a foundation
- A tone that matches the book's mood and genre
- Subtle personality influenced by what we know about the author or story

Return ONLY valid JSON:
{{
  "name": "Narrator",
  "bio": "A detailed description of the narrator's voice, tone, and personality for this specific book"
}}"""

    response = await query_small(prompt)
    return parse_json_response(response)


# ##################################################################
# chapter cache helpers
def _load_chapter_cache(cache_dir: Path, stem: str) -> dict | None:
    cache_file = cache_dir / f"{stem}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_chapter_cache(cache_dir: Path, stem: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{stem}.json"
    cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ##################################################################
# analyze characters
# main entry point — sequential accumulative character discovery.
# processes chapters in order, building up a master character list.
async def analyze_characters(
    output_dir: Path,
    title: str,
    author: str,
    voice_overrides_path: Path | None = None,
) -> Path:
    chapters_dir = output_dir / "chapters"
    characters_path = output_dir / "characters.json"
    if characters_path.exists():
        return characters_path
    cache_dir = output_dir / "characters_cache"
    chapter_files = sorted(chapters_dir.glob("*.txt"))
    if not chapter_files:
        raise ValueError("No chapter files found")

    # filter out intro, collect work items
    sample_text = ""
    work: list[tuple[Path, int]] = []
    for i, chapter_path in enumerate(chapter_files):
        if chapter_path.name == "00-intro.txt":
            continue
        if not sample_text:
            sample_text = chapter_path.read_text(encoding="utf-8")[:3000]
        work.append((chapter_path, i))

    total = len(work)
    master_characters: dict = {}

    # process chapters sequentially — accumulative discovery
    for chapter_idx, (chapter_path, chapter_num) in enumerate(work):
        stem = chapter_path.stem

        # Step 1: blind extract (cached per-chapter)
        cached = _load_chapter_cache(cache_dir, stem)
        if cached is not None:
            chapter_chars = cached.get("characters", {})
            print(f"  [{chapter_idx + 1}/{total}] {stem}: extracted {len(chapter_chars)} characters (cached)")
        else:
            result = await analyze_chapter(chapter_path, chapter_num)
            _save_chapter_cache(cache_dir, stem, result)
            chapter_chars = result.get("characters", {})
            print(f"  [{chapter_idx + 1}/{total}] {stem}: extracted {len(chapter_chars)} characters")

        if not chapter_chars:
            continue

        # Step 2+3: reconcile against master list
        if not master_characters:
            # first chapter with characters — everything is new
            for char_id, info in chapter_chars.items():
                master_characters[char_id] = {
                    "name": info.get("name", char_id),
                    "bio": info.get("details", info.get("bio", "")),
                    "chapters": [stem],
                }
            print(f"    Initial registry: {list(master_characters.keys())}")
        else:
            reconciled = await reconcile_chapter_characters(
                chapter_chars, master_characters, stem
            )

            # update master list with reconciled results
            for char_id, info in reconciled.items():
                details = info.get("details", info.get("bio", ""))
                name = info.get("name", char_id)

                if char_id in master_characters:
                    # existing character — append details, add chapter
                    if details and details not in master_characters[char_id]["bio"]:
                        master_characters[char_id]["bio"] += " " + details
                    if stem not in master_characters[char_id]["chapters"]:
                        master_characters[char_id]["chapters"].append(stem)
                    # prefer longer display name
                    if len(name) > len(master_characters[char_id].get("name", "")):
                        master_characters[char_id]["name"] = name
                else:
                    # genuinely new character
                    master_characters[char_id] = {
                        "name": name,
                        "bio": details,
                        "chapters": [stem],
                    }

        print(f"    Registry now: {len(master_characters)} characters")

    # add narrator
    has_narrator_override = False
    if voice_overrides_path and voice_overrides_path.exists():
        overrides = json.loads(voice_overrides_path.read_text(encoding="utf-8"))
        has_narrator_override = "narrator" in overrides
    if has_narrator_override:
        master_characters["narrator"] = {"name": "Narrator", "bio": "Overridden by voice_overrides.json", "chapters": []}
    else:
        narrator_info = await create_narrator_entry(title, author, sample_text)
        narrator_info["chapters"] = []
        master_characters["narrator"] = narrator_info

    characters_path.write_text(json.dumps(master_characters, indent=2), encoding="utf-8")
    return characters_path


# ##################################################################
# analyze characters sync
# synchronous wrapper for analyze_characters
def analyze_characters_sync(
    output_dir: Path, title: str, author: str, voice_overrides_path: Path | None = None,
) -> Path:
    return asyncio.run(analyze_characters(output_dir, title, author, voice_overrides_path))
