# Book Reader

EPUB to M4B audiobook converter with multi-character voice synthesis.

## Pipeline Steps

1. `extract` - EPUB to chapter text files
2. `characters` - Claude Haiku per-chapter analysis → characters.json
3. `voices_desc` - Claude Sonnet single-call → voices.json
4. `voices_clone` - TTS voice cloning → voice zips
5. `scripts` - Claude Haiku → speaker-attributed JSONL
6. `audio` - TTS synthesis → WAV files
7. `m4b` - ffmpeg assembly → M4B with chapters

## Key Files

- `output/<epub-stem>/state.jsonl` - Append-only progress tracking
- `output/<epub-stem>/characters.json` - Character bios (physical/voice focus)
- `output/<epub-stem>/voices.json` - TTS voice descriptions
- `output/<epub-stem>/script/*.jsonl` - Speaker-attributed lines

## Gotchas

### Output folder naming
Uses exact EPUB stem with spaces: `Scott Lynch - Book Title/` not normalized.

### Character analysis prompt
Must request PHYSICAL descriptions for voice generation. "Crime boss" has no value. "Elderly man with gravelly voice" has value. Exclude plot roles and cross-character relationships.

### Script generation prompt
Must be extremely terse: "Output JSONL only. No explanations." Otherwise Haiku explains what it would do instead of outputting JSONL.

### Speaker ID matching
Scripts must use exact IDs from voices.json. Pass full speaker list to prompt.

### Sonnet for deduplication
Single call, not retries. Character dedup and voice generation both use Sonnet in one call each.

### Tests are real
No mocks. Tests call actual Claude APIs and TTS tools. Audio synthesis test may hang if TTS unavailable.

## Commands

```bash
./run create book.epub    # Full pipeline
./run step <step> book.epub  # Single step
./run test <target>       # Run tests
```
