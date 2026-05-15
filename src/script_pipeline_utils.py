import re
import nltk
from src.llm_client import query_large

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def get_semantic_chunks(text: str, max_chars: int) -> list[str]:
    """
    Splits text into chunks by paragraphs and then sentences, preserving all original whitespace.
    """
    if not text:
        return []
        
    parts = re.split(r'(\n+)', text)
    
    chunks = []
    current_chunk = ""
    
    for i in range(0, len(parts), 2):
        para = parts[i]
        delim = parts[i+1] if i+1 < len(parts) else ""
        
        if len(para) > max_chars:
            sentences = nltk.sent_tokenize(para)
            for j, sent in enumerate(sentences):
                sent_delim = delim if j == len(sentences) - 1 else " "
                segment = sent + sent_delim
                
                if len(current_chunk) + len(segment) > max_chars and current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = segment
                else:
                    current_chunk += segment
        else:
            segment = para + delim
            if len(current_chunk) + len(segment) > max_chars and current_chunk:
                chunks.append(current_chunk)
                current_chunk = segment
            else:
                current_chunk += segment
                
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks


class VerbatimError(Exception):
    """Raised when the sliced text does not exactly match the original input."""
    pass

def verify_verbatim(original_text, fragments):
    """
    Ensures that the joined text of fragments equals the original source text exactly.
    fragments: list of tuples (type, text)
    """
    reconstructed = "".join([f[1] for f in fragments])
    return reconstructed == original_text

async def slice_text(text: str) -> list[tuple[str, str]]:

    """
    Segments raw text into a list of atoms: (type, text).
    Type is 'N' (Narration), 'D' (Dialogue), or 'T' (Tag).
    Uses deterministic regex to avoid LLM hallucination and timeout errors.
    """
    pattern = r'(“.*?”|".*?"|‘.*?’)'
    parts = re.split(pattern, text, flags=re.DOTALL)
    
    fragments = []
    for part in parts:
        if not part:
            continue
        # If it starts with a quote, it's Dialogue
        if part.startswith('“') or part.startswith('"') or part.startswith('‘'):
            fragments.append(('D', part))
        else:
            # Check if it's a Tag or Narration.
            lower_part = part.lower()
            tag_words = ['said', 'replied', 'asked', 'cried', 'continued', 'answered', 'inquired', 'shouted']
            is_tag = any(w in lower_part for w in tag_words) and len(part.strip()) < 100
            if is_tag:
                fragments.append(('T', part))
            else:
                fragments.append(('N', part))
                
    # Verbatim Check
    reconstructed = "".join([f[1] for f in fragments])
    if reconstructed != text:
        raise VerbatimError(f"Verbatim check failed!\nOriginal: {repr(text)}\nResult: {repr(reconstructed)}")
    
    return fragments

async def annotate_speakers(text: str, prior_context: str, fragments: list[tuple[str, str]], speaker_info_str: str) -> dict[int, str]:
    """
    Assigns a speaker_id to every 'D' (Dialogue) atom.
    Returns a mapping: {fragment_index: speaker_id}
    """
    # Prepare the fragments list for the prompt
    fragment_list_str = "\n".join([f"{i} [{f[0]}]: {f[1]}" for i, f in enumerate(fragments)])
    
    context_section = f"\nPRIOR CONTEXT:\n{prior_context}\n" if prior_context else ""
    
    prompt = f"""You are a script annotator. Your task is to identify who is speaking in the dialogue segments of the provided text.

CRITICAL INSTRUCTION: Only select speakers who are actively present in the scene. Use the provided character biographies to match the speaker's description, location, and personality with the current scene's context. Do not select characters who are not physically present.

VALID SPEAKERS AND BIOS:
{speaker_info_str}
{context_section}
RAW TEXT:
{text}

SLICED FRAGMENTS:
{fragment_list_str}

TASK:
For every segment labeled [D], identify the correct speaker_id from the VALID SPEAKERS list based on their bio and the text.
Use the RAW TEXT to understand the context and dialogue tags.

OUTPUT FORMAT:
Only output the mapping as 'index: speaker_id'. One per line. No preamble.
Provide a mapping for EVERY segment labeled [D]. If a character speaks multiple segments, you may comma-separate the indices or list them on separate lines.

Example:
1: nova_pilot
3, 5: zane_mechanic
"""

    response = await query_large(prompt)
    
    # Parse the response: "1: nova_pilot"
    mapping = {}
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or ":" not in line:
            continue
        try:
            idx_part, speaker = line.split(":", 1)
            speaker = speaker.strip()
            
            # Handle potential commas, 'and', '&'
            idx_part = idx_part.replace("&", ",").replace("and", ",")
            
            for part in idx_part.split(","):
                part = part.strip()
                if not part:
                    continue
                # Handle possible ranges like "1-3"
                if "-" in part:
                    start_str, end_str = part.split("-", 1)
                    start = int(re.search(r'\d+', start_str).group())
                    end = int(re.search(r'\d+', end_str).group())
                    for idx in range(start, end + 1):
                        mapping[idx] = speaker
                else:
                    # Extract single numbers
                    nums = re.findall(r'\d+', part)
                    for n in nums:
                        mapping[int(n)] = speaker
        except (ValueError, AttributeError):
            continue
            
    return mapping

def compile_script(fragments: list[tuple[str, str]], mapping: dict[int, str]) -> list[dict]:
    """
    Assembles the final JSONL script from fragments and labels.
    """
    script = []
    last_speaker = "narrator"
    for i, (label, text) in enumerate(fragments):
        if label == 'D':
            # Robust Continuation Heuristic
            if i not in mapping:
                # If unmapped, check if it's a split continuation: [D] -> [T] -> [D]
                if i >= 2 and fragments[i-1][0] == 'T' and fragments[i-2][0] == 'D':
                    speaker = mapping.get(i-2, last_speaker)
                else:
                    # Unsafe to assume last_speaker if separated by Narration or if it's the start
                    speaker = "narrator"
            else:
                speaker = mapping[i]
                
            last_speaker = speaker
            # Strip quotes and newlines from dialogue
            clean_text = text.strip().strip('"').strip('“').strip('”').strip("'").strip('‘').strip('’')
            clean_text = clean_text.replace('\n', ' ').strip()
            if clean_text:
                script.append({"speaker": speaker, "text": clean_text})
        else:
            # Narration and Tags are always narrator
            clean_text = text.replace('\n', ' ').strip()
            if clean_text:
                script.append({"speaker": "narrator", "text": clean_text})
    return script
