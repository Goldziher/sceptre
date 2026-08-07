"""Unit tests for `sceptre_rs_tools.text_metrics` (tokenizer, greedy matching, reading order)."""

from __future__ import annotations

import pytest
from sceptre_rs_tools.text_metrics import (
    MIN_READING_ORDER_ANCHORS,
    MatchedPair,
    f1_parts_from,
    greedy_match,
    reading_order_score,
    tokenize,
)

# -- tokenize -------------------------------------------------------------------------------


def test_tokenize_keeps_latin_words_whole() -> None:
    assert tokenize("Easy OCR rocks") == ["easy", "ocr", "rocks"]


def test_tokenize_expands_cjk_runs_into_overlapping_bigrams() -> None:
    assert tokenize("清潔できれいな") == ["清潔", "潔で", "でき", "きれ", "れい", "いな"]


def test_tokenize_keeps_a_lone_cjk_character_as_a_unigram() -> None:
    assert tokenize("清") == ["清"]


def test_tokenize_treats_pipe_as_whitespace() -> None:
    assert tokenize("a|b") == ["a", "b"]


def test_tokenize_strips_invisible_characters() -> None:
    assert tokenize("a\u200bb") == tokenize("ab")


def test_tokenize_applies_nfkc_normalization() -> None:
    # U+FF21 (fullwidth "A") NFKC-normalizes to ASCII "a" after lowercasing.
    assert tokenize("Ａ") == ["a"]


def test_tokenize_splits_mixed_cjk_and_latin_within_one_token() -> None:
    assert tokenize("画像1234test") == ["画像", "1234test"]


def test_tokenize_of_empty_string_is_empty() -> None:
    assert tokenize("") == []


# -- greedy_match / f1_parts_from ------------------------------------------------------------


def test_greedy_match_pairs_by_descending_similarity() -> None:
    similarity = {(0, 0): 0.9, (0, 1): 0.1, (1, 0): 0.2, (1, 1): 0.8}
    matches = greedy_match([0, 1], [0, 1], lambda p, r: similarity[(p, r)], threshold=0.0)
    assert sorted((m.pred_index, m.ref_index) for m in matches) == [(0, 0), (1, 1)]


def test_greedy_match_excludes_pairs_below_threshold() -> None:
    matches = greedy_match([0], [0], lambda p, r: 0.3, threshold=0.5)
    assert matches == []


def test_greedy_match_does_not_reuse_an_index() -> None:
    # Both predictions prefer reference 0; only one may take it.
    similarity = {(0, 0): 0.9, (1, 0): 0.8}
    matches = greedy_match([0, 1], [0], lambda p, r: similarity[(p, r)], threshold=0.0)
    assert len(matches) == 1
    assert matches[0] == MatchedPair(pred_index=0, ref_index=0, score=0.9)


def test_f1_parts_from_both_empty_is_perfect() -> None:
    assert f1_parts_from(0.0, 0, 0) == (1.0, 1.0, 1.0)


def test_f1_parts_from_one_empty_is_zero() -> None:
    assert f1_parts_from(0.0, 3, 0) == (0.0, 0.0, 0.0)
    assert f1_parts_from(0.0, 0, 3) == (0.0, 0.0, 0.0)


def test_f1_parts_from_computes_precision_recall_and_harmonic_mean() -> None:
    f1, precision, recall = f1_parts_from(2.0, 4, 2)
    assert precision == 0.5
    assert recall == 1.0
    assert f1 == pytest.approx(2 / 3)


# -- reading_order_score ----------------------------------------------------------------------


def test_reading_order_score_of_identical_order_is_one() -> None:
    text = " ".join(f"token{i}" for i in range(MIN_READING_ORDER_ANCHORS))
    assert reading_order_score(text, text) == 1.0


def test_reading_order_score_of_fully_reversed_order() -> None:
    tokens = [f"token{i}" for i in range(5)]
    hypothesis = " ".join(tokens)
    reference = " ".join(reversed(tokens))
    # LIS of a fully reversed sequence of length 5 is 1.
    assert reading_order_score(hypothesis, reference) == pytest.approx(1 / 5)


def test_reading_order_score_is_none_below_the_anchor_floor() -> None:
    tokens = [f"token{i}" for i in range(MIN_READING_ORDER_ANCHORS - 1)]
    text = " ".join(tokens)
    assert reading_order_score(text, text) is None


def test_reading_order_score_is_none_for_empty_text() -> None:
    assert reading_order_score("", "something") is None
    assert reading_order_score("something", "") is None


def test_reading_order_score_ignores_ambiguous_repeated_tokens() -> None:
    # "the" repeats on both sides, so it cannot anchor; the remaining distinct tokens do.
    hypothesis = "the cat the dog the bird"
    reference = "the dog the cat the bird"
    score = reading_order_score(hypothesis, reference)
    assert score is not None
    assert 0.0 <= score <= 1.0
