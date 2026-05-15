import asyncio
import json
import argparse
from pathlib import Path

from src.script_pipeline_utils import slice_text, get_semantic_chunks
from src.script_generate import load_speaker_info
from src.llm_client import query_large

async def evaluate_chunk_attribution(chunk_text: str, prior_context: str, script_lines: list[tuple[int, dict]], speaker_info_str: str) -> list[str]:
    """
    Queries the LLM to verify if dialogue attributions are correct.
    script_lines is a list of tuples: (line_index_in_script, line_dict)
    """
    dialogue_lines = [
        f"Line {idx}: [Speaker: {line['speaker']}] {line['text']}"
        for idx, line in script_lines if line['speaker'] != 'narrator'
    ]
    
    if not dialogue_lines:
        return []
        
    dialogue_str = "\n".join(dialogue_lines)
    context_section = f"\nPRIOR CONTEXT:\n{prior_context}\n" if prior_context else ""
    
    prompt = f"""You are an audiobook script evaluator.
Your task is to verify that the dialogue lines are attributed to the correct speaker based on the text.

VALID SPEAKERS AND BIOS:
{speaker_info_str}
{context_section}
SOURCE TEXT:
{chunk_text}

SCRIPT DIALOGUE LINES TO EVALUATE:
{dialogue_str}

TASK:
For each Line in the SCRIPT DIALOGUE LINES, verify if the speaker is correct given the context in the SOURCE TEXT.
Only flag an ERROR if it is definitively wrong based on the provided text. If it is plausible, ambiguous, or correct, output OK.
Output your evaluation for each line exactly in this format:
Line X: OK
or
Line X: ERROR - Expected <correct_speaker>, but got <current_speaker>

Do not include any other text.
"""
    response = await query_large(prompt)
    
    import re
    line_map = {idx: line_dict for idx, line_dict in script_lines}
    
    errors = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if "ERROR" in line.upper():
            match = re.search(r'Line\s+(\d+)', line, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                if idx in line_map:
                    speaker = line_map[idx]['speaker']
                    text_snippet = line_map[idx]['text']
                    line += f"\n      -> Actual script output: [{speaker}] \"{text_snippet}\""
            errors.append(line)
            
    return errors

async def evaluate_script(chapter_path: Path, script_path: Path, characters_path: Path):
    print(f"\n==========================================")
    print(f"Evaluating {chapter_path.name}...")
    
    original_text = chapter_path.read_text(encoding="utf-8")
    
    if not script_path.exists():
        print(f"ERROR: Script file missing at {script_path}")
        return False
        
    with open(script_path, "r", encoding="utf-8") as f:
        script_lines = [json.loads(line) for line in f if line.strip()]
        
    try:
        chunks = get_semantic_chunks(original_text, 1000)
        fragments = []
        for chunk in chunks:
            chunk_fragments = await slice_text(chunk)
            fragments.extend(chunk_fragments)
    except Exception as e:
        print(f"FAILED: Could not slice original text verbatim. Error: {e}")
        return False

    compiled_fragments = []
    for label, text in fragments:
        clean_text = text.strip()
        if label == 'D':
            clean_text = clean_text.strip('"').strip('“').strip('”').strip("'").strip('‘').strip('’')
        clean_text = clean_text.replace('\n', ' ').strip()
        if clean_text:
            compiled_fragments.append((label, text, clean_text))
            
    if len(compiled_fragments) != len(script_lines):
        print(f"FAILED Verbatim Check: Mismatch in number of lines! Source had {len(compiled_fragments)}, script has {len(script_lines)}")
        return False
        
    verbatim_passed = True
    narration_passed = True
    for i, (expected_label, original_frag_text, expected_clean) in enumerate(compiled_fragments):
        actual_line = script_lines[i]
        actual_speaker = actual_line["speaker"]
        actual_text = actual_line["text"]
        
        if expected_clean != actual_text:
            print(f"FAILED Verbatim Check at line {i}:")
            print(f"  Expected: {expected_clean}")
            print(f"  Actual:   {actual_text}")
            verbatim_passed = False
            
        if expected_label in ('N', 'T') and actual_speaker != 'narrator':
            print(f"FAILED Narration Check at line {i}: Expected narrator, got {actual_speaker}")
            narration_passed = False
            
    if not verbatim_passed or not narration_passed:
        return False
        
    print("Passed Verbatim & Narration checks.")

    # Speaker attribution check
    speaker_info_str = load_speaker_info(characters_path, chapter_path.stem)
    
    # We will chunk by fragments to ensure we keep text and script lines aligned perfectly
    chunks = []
    current_text = ""
    current_lines = []
    
    script_idx = 0
    for label, text in fragments:
        current_text += text
        clean_text = text.strip()
        if label == 'D':
            clean_text = clean_text.strip('"').strip('“').strip('”').strip("'").strip('‘').strip('’')
        clean_text = clean_text.replace('\n', ' ').strip()
        
        if clean_text:
            current_lines.append((script_idx, script_lines[script_idx]))
            script_idx += 1
            
        if len(current_text) > 3000 and len(current_lines) > 0 and label in ('N', 'T') and ('.' in text or '\n' in text):
            chunks.append((current_text, current_lines))
            current_text = ""
            current_lines = []
            
    if current_text or current_lines:
        # If last chunk is too small, just append to previous
        if chunks and len(current_text) < 200:
            chunks[-1] = (chunks[-1][0] + current_text, chunks[-1][1] + current_lines)
        else:
            chunks.append((current_text, current_lines))
            
    all_errors = []
    prior_context = ""
    
    for i, (chunk_text, chunk_lines) in enumerate(chunks):
        print(f"  -> Evaluating chunk {i+1}/{len(chunks)} for attribution...")
        
        if chunk_lines:
            errors = await evaluate_chunk_attribution(chunk_text, prior_context, chunk_lines, speaker_info_str)
            all_errors.extend(errors)
            
        prior_context = chunk_text[-1000:]
        
    if all_errors:
        print("\nFAILED Character Attribution Check. Errors found:")
        for e in all_errors:
            print("  " + e)
        return False
        
    print("Passed Character Attribution check.")
    return True

async def main():
    parser = argparse.ArgumentParser(description="Evaluate a generated script against its chapter.")
    parser.add_argument("--dir", type=str, required=True, help="Path to output directory (e.g. local/output/mock_book)")
    parser.add_argument("--chapter", type=int, help="Chapter number to evaluate. If not provided, evaluates all.")
    args = parser.parse_args()
    
    output_dir = Path(args.dir)
    chapters_dir = output_dir / "chapters"
    scripts_dir = output_dir / "scripts"
    characters_path = output_dir / "characters.json"
    
    chapter_files = sorted(chapters_dir.glob("*.txt"))
    
    if args.chapter is not None:
        if args.chapter < 0 or args.chapter >= len(chapter_files):
            print(f"Invalid chapter number. Must be between 0 and {len(chapter_files)-1}")
            return
        chapter_files = [chapter_files[args.chapter]]
        
    all_success = True
    for chapter_path in chapter_files:
        if chapter_path.name == "00-intro.txt":
            continue
            
        script_path = scripts_dir / f"{chapter_path.stem}.jsonl"
        success = await evaluate_script(chapter_path, script_path, characters_path)
        if not success:
            all_success = False
            
    if all_success:
        print("\nALL EVALUATIONS PASSED!")
    else:
        print("\nSOME EVALUATIONS FAILED.")

if __name__ == "__main__":
    asyncio.run(main())
