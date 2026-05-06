from __future__ import annotations

_REPAIR_HINTS = (
    "\ufffd",
    "Ã",
    "Â",
    "â€",
    "â€“",
    "â€”",
    "â€œ",
    "â€\x9d",
    "â€™",
    "ï¿½",
    "\u00e2\u0080",
)
_SUSPECT_MARKERS = _REPAIR_HINTS
_ASCII_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def _mojibake_score(text: str) -> int:
    marker_score = sum(text.count(marker) for marker in _SUSPECT_MARKERS)
    control_score = sum(1 for character in text if 0x80 <= ord(character) <= 0x9F)
    return marker_score + control_score


def _try_utf8_roundtrip(text: str, source_encoding: str) -> str:
    try:
        return text.encode(source_encoding).decode("utf-8")
    except UnicodeError:
        return text


def repair_common_mojibake(text: str) -> str:
    value = str(text or "")
    if not value or not any(marker in value for marker in _REPAIR_HINTS):
        return value

    best = value
    best_score = _mojibake_score(best)
    for _ in range(2):
        improved = False
        for source_encoding in ("latin-1", "cp1252"):
            candidate = _try_utf8_roundtrip(best, source_encoding)
            candidate_score = _mojibake_score(candidate)
            if candidate_score < best_score:
                best = candidate
                best_score = candidate_score
                improved = True
        if not improved:
            break
    return best


def normalize_typographic_punctuation(text: str) -> str:
    normalized = repair_common_mojibake(text)
    normalized = normalized.translate(_ASCII_PUNCTUATION)
    return normalized.replace("\u200b", "").replace("\ufeff", "")
