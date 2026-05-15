import asyncio
import json
from pathlib import Path

from src.script_pipeline_utils import slice_text, annotate_speakers, compile_script, VerbatimError, get_semantic_chunks

# chunk size for splitting chapter text into LLM-sized pieces
CHUNK_MAX_CHARS = 1000
# trailing slice of prior chunk source carried as context for speaker continuity
CONTEXT_TRAILING_CHARS = 500

async def run_decomposed_pipeline(text: str, prior_context: str, speaker_info_str: str, prior_labels: dict = None) -> list[dict]:
    """
    Executes the 3-stage decomposed pipeline on a chunk of text.
    """
    # Stage 1: Slice
    print("  -> Slicing...", end=" ", flush=True)
    for attempt in range(3):
        try:
            fragments = await slice_text(text)
            break
        except VerbatimError as e:
            if attempt == 2:
                print("FAIL")
                raise e
            print("retry", end=" ", flush=True)
    print("OK")
    
    # Stage 2: Annotate
    print("  -> Annotating...", end=" ", flush=True)
    labels = await annotate_speakers(text, prior_context, fragments, speaker_info_str)
    print("OK")
    
    # Stage 3: Compile
    print("  -> Compiling...", end=" ", flush=True)
    result = compile_script(fragments, labels)
    print("OK")
    
    return result

# ##################################################################
# load speaker info
def load_speaker_info(characters_path: Path, chapter_stem: str = None) -> str:
    if not characters_path.exists():
        raise ValueError(f"characters.json not found at {characters_path}")
    chars = json.loads(characters_path.read_text(encoding="utf-8"))
    
    info_list = []
    for speaker_id, data in chars.items():
        if speaker_id != "narrator" and chapter_stem is not None:
            chapters = data.get("chapters", [])
            if chapters and chapter_stem not in chapters:
                continue

        name = data.get("name", speaker_id)
        bio = data.get("bio", "")
        # truncate bio to keep prompt reasonable
        if len(bio) > 150:
            bio = bio[:147] + "..."
        info_list.append(f"{speaker_id}: {name} - {bio}")
    
    if "narrator" not in chars:
        info_list.append("narrator: Narrator - The narrator of the story")
    
    return "\n".join(info_list)

# ##################################################################
# generate chapter script
async def generate_chapter_script(chapter_path: Path, characters_path: Path) -> Path:
    speaker_info_str = load_speaker_info(characters_path, chapter_path.stem)
    chapter_text = chapter_path.read_text(encoding="utf-8")

    output_dir = characters_path.parent
    script_dir = output_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{chapter_path.stem}.jsonl"

    # intro is emitted verbatim as narrator
    if chapter_path.name == "00-intro.txt":
        lines = [{"speaker": "narrator", "text": chapter_text.strip()}]
        with open(script_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return script_path

    # Semantic Chunking
    all_lines: list[dict] = []
    prior_labels = {}
    
    chunks = get_semantic_chunks(chapter_text, CHUNK_MAX_CHARS)
    
    for i, chunk in enumerate(chunks):
        print(f"Processing semantic chunk {i+1}/{len(chunks)} (len: {len(chunk)})...")
        
        # Add overlap for context if not the first chunk
        prior_context = ""
        if i > 0:
            # Use the last few sentences of the previous chunk as context
            prior_context = chunks[i-1][-CONTEXT_TRAILING_CHARS:]
        
        # Process chunk
        try:
            chunk_lines = await run_decomposed_pipeline(chunk, prior_context, speaker_info_str, prior_labels)
            
            all_lines.extend(chunk_lines)
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")

    with open(script_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return script_path

# ##################################################################
# generate all scripts
async def generate_all_scripts(output_dir: Path) -> list[Path]:
    chapters_dir = output_dir / "chapters"
    characters_path = output_dir / "characters.json"
    script_dir = output_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    chapter_files = sorted(chapters_dir.glob("*.txt"))
    created: list[Path] = []
    for chapter_path in chapter_files:
        script_path = script_dir / f"{chapter_path.stem}.jsonl"
        if script_path.exists():
            created.append(script_path)
            continue
        result = await generate_chapter_script(chapter_path, characters_path)
        created.append(result)
    return created

def generate_scripts_sync(output_dir: Path) -> list[Path]:
    return asyncio.run(generate_all_scripts(output_dir))

def generate_single_script_sync(output_dir: Path, chapter_num: int) -> Path:
    chapters_dir = output_dir / "chapters"
    characters_path = output_dir / "characters.json"
    chapter_files = sorted(chapters_dir.glob("*.txt"))
    if chapter_num < 0 or chapter_num >= len(chapter_files):
        raise ValueError(f"Chapter {chapter_num} not found")
    chapter_path = chapter_files[chapter_num]
    script_path = output_dir / "scripts" / f"{chapter_path.stem}.jsonl"
    if script_path.exists():
        script_path.unlink()
    return asyncio.run(generate_chapter_script(chapter_path, characters_path))
