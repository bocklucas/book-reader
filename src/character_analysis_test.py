import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.character_analysis import (
    _load_chapter_cache,
    _save_chapter_cache,
    analyze_characters_sync,
    deterministic_match,
    get_plausible_candidates,
    normalize_for_comparison,
    parse_json_response,
)


# ##################################################################
# test parse json response
# verify json extraction from plain and markdown responses
def test_parse_json_response() -> None:
    plain = '{"name": "test"}'
    assert parse_json_response(plain) == {"name": "test"}
    markdown = '```json\n{"name": "test"}\n```'
    assert parse_json_response(markdown) == {"name": "test"}


# ##################################################################
# test parse json response empty
# verify empty/invalid inputs return safe default
def test_parse_json_response_empty() -> None:
    assert parse_json_response("") == {"characters": {}}
    assert parse_json_response("not json at all") == {"characters": {}}


# ##################################################################
# test normalize for comparison
# verify normalization strips accents, articles, prefixes
def test_normalize_for_comparison() -> None:
    assert normalize_for_comparison("the_ranger") == "ranger"
    assert normalize_for_comparison("Ranger") == "ranger"
    assert normalize_for_comparison("doña_sofia") == "sofia"
    assert normalize_for_comparison("mr_smith") == "smith"
    assert normalize_for_comparison("Mrs_Jones") == "jones"


# ##################################################################
# test deterministic match exact id
# verify exact normalized ID match works
def test_deterministic_match_exact_id() -> None:
    master = {
        "ranger": {"name": "Ranger", "bio": "Tall and silent"},
        "elena": {"name": "Elena", "bio": "Young girl"},
    }
    result = deterministic_match(
        "ranger", {"name": "Ranger", "details": "Silent type"}, master
    )
    assert result == "ranger"


# ##################################################################
# test deterministic match normalized
# verify article stripping and case normalization
def test_deterministic_match_normalized() -> None:
    master = {
        "ranger": {"name": "Ranger", "bio": "Tall and silent"},
    }
    result = deterministic_match(
        "the_ranger", {"name": "The Ranger", "details": "Silent type"}, master
    )
    assert result == "ranger"


# ##################################################################
# test deterministic match name cross
# verify name-to-id cross matching
def test_deterministic_match_name_cross() -> None:
    master = {
        "iron_knight": {"name": "Iron Knight", "bio": "Clad in armor"},
    }
    # new ID is different but name matches
    result = deterministic_match(
        "iron_man", {"name": "Iron Knight", "details": "Armored figure"}, master
    )
    assert result == "iron_knight"


# ##################################################################
# test deterministic match substring
# verify substring containment matching on IDs
def test_deterministic_match_substring() -> None:
    master = {
        "elena": {"name": "Elena", "bio": "Young girl from the countryside"},
    }
    result = deterministic_match(
        "elena_ward", {"name": "Elena Ward", "details": "Young girl"}, master
    )
    assert result == "elena"


# ##################################################################
# test deterministic match no match
# verify no false positives on unrelated characters
def test_deterministic_match_no_match() -> None:
    master = {
        "ranger": {"name": "Ranger", "bio": "Tall and silent"},
        "elena": {"name": "Elena", "bio": "Young girl"},
    }
    result = deterministic_match(
        "iron_knight", {"name": "Iron Knight", "details": "Clad in armor"}, master
    )
    assert result is None

# ##################################################################
# test deterministic match does not catch word overlap
# word-overlap is now a soft signal, not a deterministic match
def test_deterministic_match_no_word_overlap() -> None:
    master = {
        "elder_village_healer": {"name": "Elder Village Healer", "bio": "Old healer"},
    }
    result = deterministic_match(
        "village_elder", {"name": "Village Elder", "details": "Old healer"}, master
    )
    # deterministic_match should NOT catch this — it needs LLM confirmation
    assert result is None


# ##################################################################
# test find word overlap candidate flags match
# verify word-overlap flags village_elder vs elder_village_healer
def test_find_word_overlap_candidate_flags_match() -> None:
    from src.character_analysis import find_word_overlap_candidate

    master = {
        "elder_village_healer": {"name": "Elder Village Healer", "bio": "Old healer"},
        "elena": {"name": "Elena", "bio": "Young girl"},
    }
    cand_id, cand_info = find_word_overlap_candidate(
        "village_elder", {"name": "Village Elder", "details": "Old healer"}, master
    )
    assert cand_id == "elder_village_healer"
    assert cand_info is not None


# ##################################################################
# test find word overlap candidate no false flag
# verify word-overlap does NOT flag queen_of_hearts vs queen_of_diamonds
def test_find_word_overlap_candidate_no_false_flag() -> None:
    from src.character_analysis import find_word_overlap_candidate

    master = {
        "queen_of_hearts": {"name": "Queen of Hearts", "bio": "Card queen"},
    }
    cand_id, cand_info = find_word_overlap_candidate(
        "queen_of_diamonds", {"name": "Queen of Diamonds", "details": "Another card queen"}, master
    )
    assert cand_id is None


# ##################################################################
# test confirm character match llm yes
# verify LLM confirmation parses YES correctly
def test_confirm_character_match_llm_yes() -> None:
    import asyncio
    from src.character_analysis import confirm_character_match_llm

    async def fake_query_small(prompt: str, **kwargs) -> str:
        return "YES"

    with patch(
        "src.character_analysis.query_small",
        new_callable=AsyncMock,
        side_effect=fake_query_small,
    ):
        result = asyncio.run(confirm_character_match_llm(
            "village_elder", {"name": "Village Elder", "details": "Old healer"},
            "elder_village_healer", {"name": "Elder Village Healer", "bio": "Old healer"},
        ))
    assert result is True


# ##################################################################
# test confirm character match llm no
# verify LLM confirmation parses NO correctly
def test_confirm_character_match_llm_no() -> None:
    import asyncio
    from src.character_analysis import confirm_character_match_llm

    async def fake_query_small(prompt: str, **kwargs) -> str:
        return "NO"

    with patch(
        "src.character_analysis.query_small",
        new_callable=AsyncMock,
        side_effect=fake_query_small,
    ):
        result = asyncio.run(confirm_character_match_llm(
            "queen_of_hearts", {"name": "Queen of Hearts", "details": "Card queen"},
            "queen_of_diamonds", {"name": "Queen of Diamonds", "bio": "Card queen"},
        ))
    assert result is False


# ##################################################################
# test deterministic match short ids no false match
# verify very short IDs don't trigger substring matches
def test_deterministic_match_short_ids_no_false_match() -> None:
    master = {
        "al": {"name": "Al", "bio": "Short man"},
    }
    # "al" is only 2 chars, substring matching requires >= 3
    result = deterministic_match(
        "alice", {"name": "Alice", "details": "Young woman"}, master
    )
    assert result is None


# ##################################################################
# test get plausible candidates small list
# verify all candidates returned when master list is small
def test_get_plausible_candidates_small_list() -> None:
    master = {
        "ranger": {"name": "Ranger", "bio": "Tall and silent"},
        "elena": {"name": "Elena", "bio": "Young girl"},
    }
    candidates = get_plausible_candidates(
        "iron_knight", {"name": "Iron Knight", "details": "Clad in armor"}, master
    )
    # small list (<= 10) returns everything
    assert candidates == master


# ##################################################################
# test get plausible candidates filters large list
# verify filtering works on larger master lists
def test_get_plausible_candidates_filters_large_list() -> None:
    # create a master list with 12 unrelated characters
    master = {}
    for i in range(12):
        master[f"char_{i}"] = {"name": f"Character {i}", "bio": f"Description {i}"}
    # add one that shares a word with our target
    master["wood_elf"] = {"name": "Wood Elf", "bio": "Forest creature"}

    candidates = get_plausible_candidates(
        "iron_knight", {"name": "Iron Knight", "details": "Clad in armor"}, master
    )
    # filtered list should be smaller than the full master
    # none of the char_0..char_11 should match
    assert len(candidates) < len(master)


# ##################################################################
# test chapter cache round trip
# verify save and load of per-chapter cache files
def test_chapter_cache_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "characters_cache"
        data = {"characters": {"alice": {"name": "Alice", "details": "Young woman."}}}
        _save_chapter_cache(cache_dir, "ch01", data)
        loaded = _load_chapter_cache(cache_dir, "ch01")
        assert loaded == data
        assert _load_chapter_cache(cache_dir, "ch02") is None


# ##################################################################
# test analyze characters full pipeline mocked
# patches query_small to simulate blind extraction + narrator creation.
# verifies the sequential accumulative pipeline assembles characters.json
def test_analyze_characters_mocked() -> None:
    chapter1_response = json.dumps({
        "characters": {
            "ranger": {
                "name": "Ranger",
                "details": "Tall and silent, earnest voice."
            },
            "elena": {
                "name": "Elena",
                "details": "Young girl, soft timid voice."
            }
        }
    })
    chapter2_response = json.dumps({
        "characters": {
            "the_ranger": {
                "name": "The Ranger",
                "details": "Silent figure, cheerful."
            },
            "iron_knight": {
                "name": "Iron Knight",
                "details": "Clad in armor, sad polite voice."
            }
        }
    })
    narrator_response = json.dumps({
        "name": "Narrator",
        "bio": "A warm, classic storytelling voice."
    })

    call_log: list[str] = []

    async def fake_query_small(prompt: str, **kwargs) -> str:
        if "ideal narrator" in prompt.lower() or "narrator" in prompt.lower()[:100]:
            call_log.append("narrator")
            return narrator_response
        # first call gets chapter 1 chars, second gets chapter 2 chars
        if len([c for c in call_log if c.startswith("chapter")]) == 0:
            call_log.append("chapter1")
            return chapter1_response
        else:
            call_log.append("chapter2")
            return chapter2_response

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(parents=True)
        intro = chapters_dir / "00-intro.txt"
        intro.write_text("The Lost Kingdom.")
        chapter1 = chapters_dir / "01-chapter_one.txt"
        chapter1.write_text("Elena met the Ranger on the road.")
        chapter2 = chapters_dir / "02-chapter_two.txt"
        chapter2.write_text("The Ranger and Elena found the Iron Knight.")

        with patch(
            "src.character_analysis.query_small",
            new_callable=AsyncMock,
            side_effect=fake_query_small,
        ):
            result_path = analyze_characters_sync(output_dir, "Test Book", "Test Author")

        assert result_path.exists()
        chars = json.loads(result_path.read_text())

        # narrator should always be present
        assert "narrator" in chars
        assert "name" in chars["narrator"]
        assert "bio" in chars["narrator"]

        # ranger should exist (the_ranger merged via deterministic match)
        assert "ranger" in chars
        # the_ranger should NOT be a separate entry
        assert "the_ranger" not in chars

        # elena and iron_knight should exist
        assert "elena" in chars
        assert "iron_knight" in chars

        # ranger should appear in both chapters
        assert "01-chapter_one" in chars["ranger"]["chapters"]
        assert "02-chapter_two" in chars["ranger"]["chapters"]

        # iron_knight should only appear in chapter 2
        assert "02-chapter_two" in chars["iron_knight"]["chapters"]


# ##################################################################
# test analyze characters resumes from cache
# simulates an interrupted run by pre-populating per-chapter cache,
# then verifies the LLM is NOT called for the cached chapter
def test_analyze_characters_resumes_from_cache() -> None:
    chapter_response = json.dumps({
        "characters": {
            "john": {"name": "John", "details": "Large man, mid-40s."}
        }
    })
    narrator_response = json.dumps({
        "name": "Narrator",
        "bio": "A warm narrator voice."
    })
    call_log: list[str] = []

    # reconcile response: mary is NOT the same as john
    reconcile_response = "NONE"

    async def fake_query_small(prompt: str, **kwargs) -> str:
        if "Based on this book" in prompt:
            call_log.append("narrator")
            return narrator_response
        if "Is this new character the SAME PERSON" in prompt:
            call_log.append("reconcile")
            return reconcile_response
        call_log.append("extract")
        return chapter_response

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        output_dir = tmpdir / "output"
        chapters_dir = output_dir / "chapters"
        chapters_dir.mkdir(parents=True)
        (chapters_dir / "00-intro.txt").write_text("Intro text.")
        (chapters_dir / "01-chapter_one.txt").write_text("John spoke loudly.")
        (chapters_dir / "02-chapter_two.txt").write_text("Mary whispered softly.")

        # pre-cache chapter 2 — should NOT trigger an extraction LLM call
        cached_data = {
            "characters": {
                "mary": {"name": "Mary", "details": "Soft-spoken woman."}
            }
        }
        _save_chapter_cache(output_dir / "characters_cache", "02-chapter_two", cached_data)

        with patch(
            "src.character_analysis.query_small",
            new_callable=AsyncMock,
            side_effect=fake_query_small,
        ):
            result_path = analyze_characters_sync(output_dir, "Test", "Author")

        chars = json.loads(result_path.read_text())
        assert "john" in chars
        assert "mary" in chars
        assert "narrator" in chars
        # 1 extract (ch1) + 1 reconcile (mary vs john) + 1 narrator = 3
        # ch2 extraction was cached, so no extract call for it
        assert call_log.count("extract") == 1
        assert call_log.count("narrator") == 1
