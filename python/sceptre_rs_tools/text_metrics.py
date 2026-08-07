"""Text tokenization and greedy matching primitives for the benchmark harness.

Ported from xberg's ``tools/benchmark-harness``:

* ``tokenize`` -- NFKC normalization, invisible-character stripping, and CJK bigram
  expansion (``src/quality.rs::tokenize`` / ``tokenize_cjk_bigrams``), minus the
  Markdown-link stripping and thousands-separator normalization that harness pulls in
  for extracted document text -- neither applies to a single OCR line.
* ``greedy_match`` / ``f1_parts_from`` -- the matching machinery from
  ``src/structural_sidecar.rs``, generalized here over any similarity function rather
  than SF1's Markdown-structure scoring. Applied to lines (box-IoU) instead of
  headings/tables/lists.
* ``reading_order_score`` -- anchor-LIS ordering score (``src/quality.rs``).

The Rust mirror lives in ``crates/sceptre/tests/helpers/mod.rs``; the two are kept in
lockstep by ``crates/sceptre/tests/data/metrics_vectors.json``, asserted at 1e-6 by both
``crates/sceptre/tests/helpers/mod.rs`` and ``python/tests/test_metrics_vectors.py``.
"""

from __future__ import annotations

import bisect
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

# Zero-width space, zero-width non-joiner/joiner, BOM, soft hyphen: characters a scanner or
# PDF extractor sometimes emits that carry no visible content and would otherwise silently
# break bigram boundaries. ~keep
_INVISIBLE_CHARACTERS = "\u200b\u200c\u200d\ufeff\xad"  # ZWSP, ZWNJ, ZWJ, BOM, soft hyphen

# A token occurring more than once on either side is ambiguous as a reading-order anchor
# (which occurrence maps to which?), so anchors require an exact one-to-one match. Below
# this many anchors, an ordering score is noise rather than a signal. xberg's own
# `MIN_READING_ORDER_ANCHORS` value was not available to this port at the time it was
# written; 3 is this port's own choice (not a verified copy of xberg's constant) and
# should be reconciled against xberg's source before being treated as load-bearing. ~keep
MIN_READING_ORDER_ANCHORS = 3


def _is_cjk_character(character: str) -> bool:
    """True for CJK ideographs, kana, and Hangul syllables."""
    code = ord(character)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # CJK Extension A
        or 0x3040 <= code <= 0x30FF  # Hiragana + Katakana
        or 0xAC00 <= code <= 0xD7A3  # Hangul syllables
        or 0xF900 <= code <= 0xFAFF  # CJK compatibility ideographs
    )


def _cjk_bigrams(run: str) -> list[str]:
    """Overlapping bigrams of a CJK run; a single character stays a unigram."""
    if len(run) <= 1:
        return [run] if run else []
    return [run[index : index + 2] for index in range(len(run) - 1)]


def _split_runs(token: str) -> list[tuple[str, bool]]:
    """Split a token into maximal runs of (is_cjk) characters, in order."""
    runs: list[tuple[str, bool]] = []
    buffer = ""
    buffer_is_cjk: bool | None = None
    for character in token:
        is_cjk = _is_cjk_character(character)
        if buffer_is_cjk is None or is_cjk == buffer_is_cjk:
            buffer += character
        else:
            runs.append((buffer, bool(buffer_is_cjk)))
            buffer = character
        buffer_is_cjk = is_cjk
    if buffer:
        runs.append((buffer, bool(buffer_is_cjk)))
    return runs


def _expand_token(token: str) -> list[str]:
    """Bigram-expand the CJK runs of one whitespace-delimited token; keep other runs whole."""
    expanded: list[str] = []
    for run, is_cjk in _split_runs(token):
        expanded.extend(_cjk_bigrams(run) if is_cjk else [run])
    return expanded


def tokenize(text: str) -> list[str]:
    """NFKC-normalize, strip invisible characters, lowercase, and CJK-bigram-expand.

    Whitespace-delimited scripts (English, Cyrillic, ...) tokenize as before -- one token
    per whitespace-separated run. A CJK run within a token expands into overlapping
    bigrams (a lone CJK character stays a unigram), so word-level F1 scores CJK text on
    adjacent-character pairs instead of treating an entire line as one degenerate token.
    """
    normalized = unicodedata.normalize("NFKC", text)
    for invisible in _INVISIBLE_CHARACTERS:
        normalized = normalized.replace(invisible, "")
    normalized = normalized.lower().replace("|", " ")
    tokens: list[str] = []
    for token in normalized.split():
        tokens.extend(_expand_token(token))
    return tokens


_Pred = TypeVar("_Pred")
_Ref = TypeVar("_Ref")


@dataclass(frozen=True)
class MatchedPair:
    """One greedy-matched (prediction, reference) index pair and its similarity."""

    pred_index: int
    ref_index: int
    score: float


def greedy_match(
    predicted: list[_Pred], reference: list[_Ref], similarity: Callable[[_Pred, _Ref], float], threshold: float
) -> list[MatchedPair]:
    """Greedily pair predictions with references by descending similarity.

    Every candidate pair scoring at least `threshold` is considered, highest similarity
    first; each prediction and each reference index is used in at most one pair. This is
    the matching machinery from xberg's `structural_sidecar.rs::greedy_match`, generalized
    over any similarity function (here: box-IoU between detected and reference lines,
    rather than SF1's Markdown-structure content similarity).
    """
    candidates = [
        (similarity(pred, ref_item), pred_index, ref_index)
        for pred_index, pred in enumerate(predicted)
        for ref_index, ref_item in enumerate(reference)
        if similarity(pred, ref_item) >= threshold
    ]
    candidates.sort(key=lambda candidate: candidate[0], reverse=True)

    used_preds: set[int] = set()
    used_refs: set[int] = set()
    matches: list[MatchedPair] = []
    for score, pred_index, ref_index in candidates:
        if pred_index in used_preds or ref_index in used_refs:
            continue
        used_preds.add(pred_index)
        used_refs.add(ref_index)
        matches.append(MatchedPair(pred_index, ref_index, score))
    return matches


def f1_parts_from(matched_credit: float, n_pred: int, n_ref: int) -> tuple[float, float, float]:
    """(f1, precision, recall) from a matched credit against prediction/reference counts.

    Both-empty scores (1.0, 1.0, 1.0); one-empty scores (0.0, 0.0, 0.0) -- ported from
    xberg's `structural_sidecar.rs::f1_parts_from`.
    """
    if n_pred == 0 and n_ref == 0:
        return (1.0, 1.0, 1.0)
    if n_pred == 0 or n_ref == 0:
        return (0.0, 0.0, 0.0)
    precision = matched_credit / n_pred
    recall = matched_credit / n_ref
    if precision + recall == 0.0:
        return (0.0, precision, recall)
    f1 = 2.0 * precision * recall / (precision + recall)
    return (f1, precision, recall)


def reading_order_score(hypothesis_text: str, reference_text: str) -> float | None:
    """Anchor-LIS reading-order agreement in `[0, 1]`, or `None` below the anchor floor.

    An "anchor" is a token that occurs exactly once in both `hypothesis_text` and
    `reference_text` -- unambiguous, so its position in one sequence can be compared to
    its position in the other. Anchors are ordered by their reference-text position, and
    the score is the length of the longest increasing subsequence of their
    hypothesis-text positions, divided by the anchor count: 1.0 means every anchor
    appears in the same relative order in both texts. Ported from xberg's
    `quality.rs::reading_order_score`; catches column-order and vertical-script ordering
    regressions that bag-of-words F1 and box-IoU are both blind to.
    """
    hypothesis_tokens = tokenize(hypothesis_text)
    reference_tokens = tokenize(reference_text)
    if not hypothesis_tokens or not reference_tokens:
        return None

    hypothesis_positions: dict[str, list[int]] = {}
    for index, token in enumerate(hypothesis_tokens):
        hypothesis_positions.setdefault(token, []).append(index)
    reference_positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference_tokens):
        reference_positions.setdefault(token, []).append(index)

    anchors = [
        (reference_positions[token][0], hypothesis_positions[token][0])
        for token in reference_positions
        if len(reference_positions[token]) == 1 and len(hypothesis_positions.get(token, [])) == 1
    ]
    if len(anchors) < MIN_READING_ORDER_ANCHORS:
        return None

    anchors.sort(key=lambda anchor: anchor[0])
    hypothesis_order = [hypothesis_index for _, hypothesis_index in anchors]
    return _longest_increasing_subsequence_len(hypothesis_order) / len(anchors)


def _longest_increasing_subsequence_len(values: list[int]) -> int:
    """Length of the longest strictly increasing subsequence, via patience sorting (O(n log n))."""
    tails: list[int] = []
    for value in values:
        position = bisect.bisect_left(tails, value)
        if position == len(tails):
            tails.append(value)
        else:
            tails[position] = value
    return len(tails)
