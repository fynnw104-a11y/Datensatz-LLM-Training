from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MULTIMODAL_DIR = ROOT / "data" / "processed" / "multimodal"
DEFAULT_SOURCE_GLOB = "data/processed/multimodal/pairs/*.json"
DEFAULT_OUTPUT_DIR = MULTIMODAL_DIR / "training_pairs"
DEFAULT_INDEX_PATH = MULTIMODAL_DIR / "training_pairs.jsonl"
DEFAULT_MANIFEST_PATH = MULTIMODAL_DIR / "training_pairs_manifest.json"
TRAINING_PROMPT_FILENAME = "_instruction.txt"
TRAINING_PROMPT = (
    "Analyze the attached trading-related image and return a concise JSON object using only directly visible "
    "evidence from the image crop. Do not use page context or hidden assumptions. If a field is not clearly "
    "visible, omit it instead of inferring it."
)
QUALITY_RANKS = {"low": 0, "medium": 1, "high": 2}
KNOWN_VENUES = {"BINANCE", "BITGET", "BYBIT", "COINBASE", "FXCM", "KRAKEN", "OKX"}
TIMEFRAME_TOKEN_MAP = {
    "1": "M1",
    "1m": "M1",
    "m1": "M1",
    "3": "M3",
    "3m": "M3",
    "m3": "M3",
    "5": "M5",
    "5m": "M5",
    "m5": "M5",
    "15": "M15",
    "15m": "M15",
    "m15": "M15",
    "30": "M30",
    "30m": "M30",
    "m30": "M30",
    "45": "M45",
    "45m": "M45",
    "m45": "M45",
    "1h": "H1",
    "h1": "H1",
    "2h": "H2",
    "h2": "H2",
    "4h": "H4",
    "h4": "H4",
    "6h": "H6",
    "h6": "H6",
    "8h": "H8",
    "h8": "H8",
    "12h": "H12",
    "h12": "H12",
    "1d": "D1",
    "d1": "D1",
    "1w": "W1",
    "w1": "W1",
    "1mo": "MN1",
    "mn1": "MN1",
}
SYMBOL_DISPLAY_NAMES = {
    "BTCUSDT": "Bitcoin / TetherUS",
    "BTCUSD": "Bitcoin / US Dollar",
    "ETHUSDT": "Ethereum / TetherUS",
    "ETHUSD": "Ethereum / US Dollar",
    "EURUSD": "Euro / US-Dollar",
    "GBPUSD": "British Pound / US-Dollar",
    "USDJPY": "US-Dollar / Japanese Yen",
    "AUDUSD": "Australian Dollar / US-Dollar",
    "NZDUSD": "New Zealand Dollar / US-Dollar",
    "USDCAD": "US-Dollar / Canadian Dollar",
    "USDCHF": "US-Dollar / Swiss Franc",
    "XAUUSD": "Gold / US Dollar",
    "XAGUSD": "Silver / US Dollar",
}
INSTRUMENT_COMPONENT_ALIASES = {
    "australian dollar": "AUD",
    "aud": "AUD",
    "bitcoin": "BTC",
    "btc": "BTC",
    "british pound": "GBP",
    "cad": "CAD",
    "canadian dollar": "CAD",
    "chf": "CHF",
    "ethereum": "ETH",
    "eth": "ETH",
    "eur": "EUR",
    "euro": "EUR",
    "gbp": "GBP",
    "gold": "XAU",
    "japanese yen": "JPY",
    "jpy": "JPY",
    "new zealand dollar": "NZD",
    "nzd": "NZD",
    "silver": "XAG",
    "swiss franc": "CHF",
    "tether": "USDT",
    "tether us": "USDT",
    "tether usd": "USDT",
    "tetherus": "USDT",
    "usd": "USD",
    "us dollar": "USD",
    "us-dollar": "USD",
    "usdt": "USDT",
    "yen": "JPY",
}
OBJECTIVE_LIMITATIONS = (
    ("some text is blurry or partially unreadable", ("blurry", "blurred", "unreadable", "obscured", "fragmented", "corrupted")),
    ("exact price values are difficult to read precisely", ("price scale", "price values", "price axis", "difficult to read precisely")),
    ("cropped edges limit full chart context", ("cropped", "crop", "edges limit", "broader context", "full context", "full session visibility")),
    ("some header or indicator text is partially obscured", ("header text", "indicator", "timeframe information", "header and indicator")),
)
VISIBLE_LABEL_SEPARATOR_PATTERN = re.compile(r"\s*[·•∙,|]\s*|\s+[+\-]\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact image+JSON training pairs from multimodal asset annotations."
    )
    parser.add_argument("--glob", dest="glob_pattern", default=DEFAULT_SOURCE_GLOB, help="Asset annotation glob pattern.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for compact training pairs.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="JSONL index for exported training pairs.")
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Manifest JSON summarizing the export run.",
    )
    parser.add_argument(
        "--min-quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimum annotation quality required for export.",
    )
    parser.add_argument(
        "--require-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only export assets that already contain ChatGPT enrichment.",
    )
    parser.add_argument(
        "--allow-review-required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also export assets that are still marked for manual review.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def root_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ascii_fold(value: str) -> str:
    return (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(text: str) -> str:
    folded = ascii_fold(text).lower()
    folded = re.sub(r"[^a-z0-9]+", " ", folded)
    return normalize_whitespace(folded)


def first_non_empty(*values: object) -> str:
    for value in values:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
    return ""


def normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = normalize_whitespace(item)
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def get_nested(mapping: object, *keys: str) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def annotation_quality(annotation: dict[str, Any]) -> str:
    quality = str(
        get_nested(annotation, "target_json", "provenance", "quality", "annotation_quality") or ""
    ).strip().lower()
    if quality in QUALITY_RANKS:
        return quality
    return "low" if annotation.get("review_required") else "medium"


def llm_confidence(annotation: dict[str, Any]) -> str:
    confidence = str(get_nested(annotation, "llm_enrichment", "structured_response", "confidence") or "").strip().lower()
    if confidence in QUALITY_RANKS:
        return confidence
    return ""


def source_uses_ocr(source_name: object) -> bool:
    return isinstance(source_name, str) and "ocr" in source_name


def visible_clean_text(annotation: dict[str, Any]) -> str:
    value = get_nested(annotation, "target_json", "observed", "visible_in_crop", "clean_text")
    return value.strip() if isinstance(value, str) else ""


def visible_ocr_text(annotation: dict[str, Any]) -> str:
    value = get_nested(annotation, "target_json", "observed", "visible_in_crop", "ocr_text")
    if isinstance(value, str) and value.strip():
        return value.strip()
    fallback = annotation.get("ocr_text")
    return fallback.strip() if isinstance(fallback, str) else ""


def visible_visual_tags(annotation: dict[str, Any]) -> list[str]:
    return normalize_string_list(get_nested(annotation, "target_json", "observed", "visible_in_crop", "visual_elements"))


def normalized_visible_fields(annotation: dict[str, Any]) -> dict[str, Any]:
    value = get_nested(annotation, "target_json", "observed", "visible_in_crop", "normalized_fields")
    return value if isinstance(value, dict) else {}


def field_sources(annotation: dict[str, Any]) -> dict[str, str]:
    value = get_nested(annotation, "target_json", "provenance", "field_sources")
    return value if isinstance(value, dict) else {}


def extract_visible_labels_line(clean_text: object) -> str:
    if not isinstance(clean_text, str):
        return ""
    for raw_line in clean_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("visible labels:"):
            return line.split(":", 1)[1].strip()
    return ""


def extract_prefixed_line(clean_text: object, prefix: str) -> str:
    if not isinstance(clean_text, str):
        return ""
    prefix_folded = prefix.lower()
    for raw_line in clean_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith(prefix_folded):
            return line.split(":", 1)[1].strip()
    return ""


def detect_venue(text: str) -> str:
    compact = ascii_fold(text).upper()
    for venue in sorted(KNOWN_VENUES):
        if venue in compact:
            return venue
    return ""


def normalize_visible_label_text(text: str) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""
    cleaned = cleaned.replace("•", "·").replace("∙", "·")
    cleaned = cleaned.replace("–", "-").replace("—", "-").replace("−", "-")
    cleaned = re.sub(r"\s*\+\s*", " + ", cleaned)
    cleaned = re.sub(r"\s*·\s*", " · ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s*\|\s*", " | ", cleaned)
    cleaned = re.sub(r"\s+-\s+", " - ", cleaned)
    return normalize_whitespace(cleaned)


def canonicalize_timeframe_token(token: str) -> str:
    normalized = ascii_fold(token).lower()
    normalized = normalized.replace("minutes", "m").replace("minute", "m")
    normalized = normalized.replace("hours", "h").replace("hour", "h")
    normalized = normalized.replace(" ", "")
    if normalized in TIMEFRAME_TOKEN_MAP:
        return TIMEFRAME_TOKEN_MAP[normalized]
    minute_match = re.fullmatch(r"(\d+)m", normalized)
    if minute_match:
        return f"M{minute_match.group(1)}"
    minute_prefix_match = re.fullmatch(r"m(\d+)", normalized)
    if minute_prefix_match:
        return f"M{minute_prefix_match.group(1)}"
    hour_match = re.fullmatch(r"(\d+)h", normalized)
    if hour_match:
        return f"H{hour_match.group(1)}"
    hour_prefix_match = re.fullmatch(r"h(\d+)", normalized)
    if hour_prefix_match:
        return f"H{hour_prefix_match.group(1)}"
    day_match = re.fullmatch(r"(\d+)d", normalized)
    if day_match:
        return f"D{day_match.group(1)}"
    week_match = re.fullmatch(r"(\d+)w", normalized)
    if week_match:
        return f"W{week_match.group(1)}"
    return ""


def split_visible_label_parts(label_line: str) -> list[str]:
    normalized = normalize_visible_label_text(label_line)
    parts: list[str] = []
    for raw_part in VISIBLE_LABEL_SEPARATOR_PATTERN.split(normalized):
        cleaned = normalize_whitespace(str(raw_part).strip("+-·|, "))
        if cleaned:
            parts.append(cleaned)
    return parts


def extract_timeframes_from_text(text: str) -> list[str]:
    normalized = ascii_fold(text).lower()
    candidates: list[str] = []
    for token in re.findall(r"\b\d+\s*(?:m|h|d|w|mo)?\b|\b(?:m|h|d|w|mn)\s*\d+\b", normalized):
        timeframe = canonicalize_timeframe_token(token)
        if timeframe:
            candidates.append(timeframe)
    return unique_preserving_order(candidates)


def timeframe_display_token(timeframe: str) -> str:
    minute_match = re.fullmatch(r"M(\d+)", timeframe)
    if minute_match:
        return minute_match.group(1)
    hour_match = re.fullmatch(r"H(\d+)", timeframe)
    if hour_match:
        return f"{hour_match.group(1)}h"
    day_match = re.fullmatch(r"D(\d+)", timeframe)
    if day_match:
        return f"{day_match.group(1)}d"
    week_match = re.fullmatch(r"W(\d+)", timeframe)
    if week_match:
        return f"{week_match.group(1)}w"
    if timeframe == "MN1":
        return "1mo"
    return timeframe


def canonical_visible_text(raw_visible_text: str, instrument_name: str, timeframes: list[str], venue: str, primary_symbol: str) -> str:
    label = instrument_name or primary_symbol
    parts = [label]
    if timeframes:
        parts.append("/".join(timeframe_display_token(value) for value in timeframes[:2]))
    if venue:
        parts.append(venue)
    normalized_parts = [normalize_whitespace(part) for part in parts if normalize_whitespace(part)]
    if len(normalized_parts) >= 2:
        return " - ".join(normalized_parts)
    return normalize_visible_label_text(raw_visible_text)


def lookup_symbol_from_display_name(instrument_name: str) -> str:
    normalized_name = normalize_label(instrument_name)
    for symbol, display_name in SYMBOL_DISPLAY_NAMES.items():
        if normalized_name == normalize_label(display_name):
            return symbol
    return ""


def resolve_instrument_component(text: str) -> str:
    normalized = normalize_label(text)
    if not normalized:
        return ""
    for alias in sorted(INSTRUMENT_COMPONENT_ALIASES, key=len, reverse=True):
        if alias in normalized:
            return INSTRUMENT_COMPONENT_ALIASES[alias]
    compact = re.sub(r"\s+", "", normalized).upper()
    if compact in {"AUD", "BTC", "CAD", "CHF", "ETH", "EUR", "GBP", "JPY", "NZD", "USD", "USDT", "XAG", "XAU"}:
        return compact
    return ""


def infer_symbol_from_instrument_name(instrument_name: str) -> str:
    direct_symbol = lookup_symbol_from_display_name(instrument_name)
    if direct_symbol:
        return direct_symbol

    compact = re.sub(r"\s+", "", ascii_fold(instrument_name)).upper()
    direct_match = re.search(r"\b([A-Z]{6}|[A-Z]{2,10}(?:USD|USDT))\b", compact)
    if direct_match:
        return direct_match.group(1)

    parts = [part.strip() for part in re.split(r"\s*/\s*", instrument_name, maxsplit=1) if part.strip()]
    if len(parts) != 2:
        return ""
    base_symbol = resolve_instrument_component(parts[0])
    quote_symbol = resolve_instrument_component(parts[1])
    if base_symbol and quote_symbol:
        return base_symbol + quote_symbol
    return ""


def parse_visible_label_fields(label_line: str) -> dict[str, Any]:
    if not label_line:
        return {}

    candidates = [normalize_visible_label_text(candidate) for candidate in label_line.split(";") if normalize_visible_label_text(candidate)]
    if not candidates:
        return {}

    def candidate_score(candidate: str) -> tuple[int, int]:
        score = 0
        if "/" in candidate:
            score += 2
        if detect_venue(candidate):
            score += 2
        if any(canonicalize_timeframe_token(part) for part in split_visible_label_parts(candidate)):
            score += 1
        return score, len(candidate)

    selected = max(candidates, key=candidate_score)
    parts = split_visible_label_parts(selected)
    if not parts:
        return {}

    instrument_name = parts[0]
    venue = ""
    timeframes: list[str] = []

    for part in parts[1:]:
        timeframe = canonicalize_timeframe_token(part)
        if timeframe:
            timeframes.append(timeframe)
            continue
        detected_venue = detect_venue(part)
        if detected_venue and not venue:
            venue = detected_venue

    if not venue:
        venue = detect_venue(selected)
    if not timeframes:
        timeframes = extract_timeframes_from_text(selected)

    return {
        "instrument_name": instrument_name,
        "primary_symbol": infer_symbol_from_instrument_name(instrument_name),
        "venue": venue,
        "timeframes": unique_preserving_order(timeframes),
    }


def symbol_is_visible(symbol: str, visible_label_text: str, ocr_text: str) -> bool:
    if not symbol:
        return False
    haystacks = [visible_label_text, ocr_text, SYMBOL_DISPLAY_NAMES.get(symbol, "")]
    compact_symbol = re.sub(r"\s+", "", ascii_fold(symbol)).upper()
    for haystack in haystacks:
        if not haystack:
            continue
        compact_haystack = re.sub(r"\s+", "", ascii_fold(haystack)).upper()
        if compact_symbol and compact_symbol in compact_haystack:
            return True
    display_name = SYMBOL_DISPLAY_NAMES.get(symbol, "")
    if display_name:
        return normalize_label(display_name) in normalize_label(visible_label_text) or normalize_label(display_name) in normalize_label(ocr_text)
    return False


def timeframe_is_visible(timeframe: str, visible_label_text: str, ocr_text: str) -> bool:
    if not timeframe:
        return False

    visible_text = ascii_fold("\n".join(part for part in [visible_label_text, ocr_text] if part)).lower()
    aliases = {timeframe.lower()}
    for raw_timeframe, canonical in TIMEFRAME_TOKEN_MAP.items():
        if canonical == timeframe:
            aliases.add(raw_timeframe)

    for alias in aliases:
        escaped = re.escape(alias)
        if re.search(rf"\b{escaped}\b", visible_text):
            return True

    minute_match = re.fullmatch(r"M(\d+)", timeframe)
    if minute_match:
        minute_value = minute_match.group(1)
        if re.search(rf"\b{re.escape(minute_value)}\b(?=\s*[,/\-|]?\s*(?:{'|'.join(sorted(venue.lower() for venue in KNOWN_VENUES))})\b)", visible_text):
            return True
    return False


def trust_ocr_backed_scalar(candidate: str, source_name: str, visible_label_text: str, ocr_text: str) -> str:
    if not candidate or not source_uses_ocr(source_name):
        return ""
    candidate_norm = normalize_label(candidate)
    if candidate_norm and (candidate_norm in normalize_label(visible_label_text) or candidate_norm in normalize_label(ocr_text)):
        return candidate
    return ""


def trust_ocr_backed_timeframes(candidates: list[str], source_name: str, visible_label_text: str, ocr_text: str) -> list[str]:
    if not candidates or not source_uses_ocr(source_name):
        return []
    return [value for value in candidates if timeframe_is_visible(value, visible_label_text, ocr_text)]


def derive_grounded_fields(annotation: dict[str, Any]) -> dict[str, Any]:
    raw_visible_label_text = first_non_empty(
        extract_visible_labels_line(visible_clean_text(annotation)),
        get_nested(annotation, "llm_enrichment", "structured_response", "visible_text"),
    )
    parsed_labels = parse_visible_label_fields(raw_visible_label_text)
    sources = field_sources(annotation)
    visible_fields = normalized_visible_fields(annotation)
    ocr_text = visible_ocr_text(annotation)

    normalized_primary_symbol = first_non_empty(
        visible_fields.get("primary_symbol"),
        annotation.get("primary_symbol"),
    )
    normalized_instrument_name = first_non_empty(
        visible_fields.get("instrument_name"),
        annotation.get("instrument_name"),
    )
    normalized_venue = first_non_empty(
        visible_fields.get("venue"),
        annotation.get("venue"),
    )
    normalized_timeframes = normalize_string_list(visible_fields.get("timeframes")) or normalize_string_list(annotation.get("timeframes"))

    instrument_name = first_non_empty(
        parsed_labels.get("instrument_name"),
        trust_ocr_backed_scalar(normalized_instrument_name, sources.get("instrument_name", ""), raw_visible_label_text, ocr_text),
    )

    venue = first_non_empty(
        parsed_labels.get("venue"),
        trust_ocr_backed_scalar(normalized_venue, sources.get("venue", ""), raw_visible_label_text, ocr_text),
    )

    timeframes = unique_preserving_order(
        normalize_string_list(parsed_labels.get("timeframes"))
        or trust_ocr_backed_timeframes(normalized_timeframes, sources.get("timeframes", ""), raw_visible_label_text, ocr_text)
    )

    primary_symbol = first_non_empty(
        parsed_labels.get("primary_symbol"),
        infer_symbol_from_instrument_name(instrument_name),
    )
    if not primary_symbol and normalized_primary_symbol and symbol_is_visible(normalized_primary_symbol, raw_visible_label_text, ocr_text):
        primary_symbol = normalized_primary_symbol

    if not instrument_name and primary_symbol:
        instrument_name = SYMBOL_DISPLAY_NAMES.get(primary_symbol, "")

    visible_text = canonical_visible_text(
        raw_visible_label_text,
        instrument_name=instrument_name,
        timeframes=timeframes,
        venue=venue,
        primary_symbol=primary_symbol,
    )

    return {
        "visible_text": visible_text,
        "primary_symbol": primary_symbol,
        "instrument_name": instrument_name,
        "venue": venue,
        "timeframes": timeframes,
    }


def classify_visual_element(raw_value: str) -> str:
    value = normalize_whitespace(raw_value)
    lowered = ascii_fold(value).lower().replace("_", " ")
    if not lowered:
        return ""
    if "candlestick" in lowered or "candle" in lowered:
        return "candlestick chart"
    if "session" in lowered:
        return "session overlay"
    if "circle" in lowered or "marker" in lowered or "highlight" in lowered:
        return "highlighted swing markers"
    if "arrow" in lowered:
        return "directional arrows"
    if "rectangle" in lowered or "shaded" in lowered or "overlay" in lowered or "zone" in lowered:
        return "shaded rectangular overlays"
    if "price scale" in lowered or "price axis" in lowered or "right-side price scale" in lowered:
        return "price scale"
    if "time axis" in lowered or "time labels" in lowered or "timeline" in lowered or "intraday time" in lowered:
        return "time axis"
    if "watermark" in lowered or "header" in lowered or "tradingview" in lowered:
        return "TradingView header or watermark"
    if "ocr text overlay" in lowered or "visible text overlay" in lowered:
        return "visible text overlay"
    if "symbol label" in lowered or "instrument label" in lowered:
        return "instrument label"
    if "timeframe label" in lowered:
        return "timeframe label"
    if "venue label" in lowered:
        return "venue label"
    if "price_axis_or_scale" in lowered:
        return "price scale"
    if "chart_panel" in lowered:
        return "chart panel"
    if "black and white candlesticks" in lowered:
        return "black and white candlesticks"
    return ""


def compact_visual_elements(annotation: dict[str, Any]) -> list[str]:
    raw_elements = [
        *normalize_string_list(get_nested(annotation, "llm_enrichment", "structured_response", "key_visual_elements")),
        *normalize_string_list(get_nested(annotation, "target_json", "description", "key_visual_elements")),
        *visible_visual_tags(annotation),
    ]
    normalized = [classify_visual_element(value) for value in raw_elements]
    return unique_preserving_order([value for value in normalized if value])


def compact_limitations(annotation: dict[str, Any]) -> list[str]:
    raw_values = [
        *normalize_string_list(get_nested(annotation, "llm_enrichment", "structured_response", "limitations")),
        *normalize_string_list(get_nested(annotation, "target_json", "description", "limitations")),
    ]
    normalized: list[str] = []
    for raw_value in raw_values:
        lowered = ascii_fold(raw_value).lower()
        if any(marker in lowered for marker in ("inferred", "implied", "likely", "potential", "may indicate", "context")):
            continue
        for canonical, triggers in OBJECTIVE_LIMITATIONS:
            if any(trigger in lowered for trigger in triggers):
                normalized.append(canonical)
                break
    return unique_preserving_order(normalized)


def asset_type_has_chart_evidence(annotation: dict[str, Any], key_visual_elements: list[str], grounded_fields: dict[str, Any]) -> bool:
    raw_text = " ".join(
        value
        for value in (
            first_non_empty(annotation.get("asset_type")),
            first_non_empty(annotation.get("page_type")),
            first_non_empty(annotation.get("caption")),
            first_non_empty(annotation.get("summary")),
            first_non_empty(get_nested(annotation, "llm_enrichment", "structured_response", "short_caption")),
            first_non_empty(get_nested(annotation, "llm_enrichment", "structured_response", "visual_summary")),
            " ".join(key_visual_elements),
            grounded_fields.get("visible_text", ""),
        )
        if value
    )
    lowered = ascii_fold(raw_text).lower()
    labels = annotation.get("labels") if isinstance(annotation.get("labels"), dict) else {}
    return bool(
        annotation.get("page_type") == "chart"
        or labels.get("likely_chart")
        or any(tag in visible_visual_tags(annotation) for tag in ("chart_panel", "price_axis_or_scale", "timeframe_label"))
        or any(marker in lowered for marker in ("chart", "candlestick", "tradingview", "price scale", "time axis"))
    )


def repaired_asset_type(annotation: dict[str, Any], grounded_fields: dict[str, Any], key_visual_elements: list[str]) -> str:
    raw_asset_type = first_non_empty(annotation.get("asset_type")).lower()
    if asset_type_has_chart_evidence(annotation, key_visual_elements, grounded_fields):
        return "chart"
    if raw_asset_type and raw_asset_type != "unknown":
        return raw_asset_type
    page_type = first_non_empty(annotation.get("page_type")).lower()
    if page_type and page_type != "unknown":
        return page_type
    return ""


def render_list(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def subject_label(primary_symbol: str, instrument_name: str) -> str:
    return primary_symbol or instrument_name or "trading-related"


def timeframe_label(timeframes: list[str]) -> str:
    if not timeframes:
        return ""
    if len(timeframes) == 1:
        return timeframes[0]
    return "/".join(timeframes[:2])


def caption_modifiers(key_visual_elements: list[str]) -> list[str]:
    preferred = []
    for value in key_visual_elements:
        if value in {"session overlay", "highlighted swing markers", "directional arrows", "shaded rectangular overlays"}:
            preferred.append(value)
    return unique_preserving_order(preferred)[:2]


def build_short_caption(asset_type: str, grounded_fields: dict[str, Any], key_visual_elements: list[str]) -> str:
    subject = subject_label(grounded_fields["primary_symbol"], grounded_fields["instrument_name"])
    timeframe = timeframe_label(grounded_fields["timeframes"])
    noun = asset_type.replace("_", " ") if asset_type else "visual"
    if subject == "trading-related":
        base = f"Trading-related {noun}"
        if timeframe:
            base += f" with visible timeframe {timeframe}"
    else:
        base = " ".join(part for part in (subject, timeframe, noun) if part)
    modifiers = caption_modifiers(key_visual_elements)
    if modifiers:
        prefix = "Annotated " if any(value in modifiers for value in ("highlighted swing markers", "directional arrows", "shaded rectangular overlays")) else ""
        return prefix + base + " with " + render_list(modifiers)
    return base


def build_visual_summary(asset_type: str, grounded_fields: dict[str, Any], key_visual_elements: list[str], annotation: dict[str, Any]) -> str:
    subject = subject_label(grounded_fields["primary_symbol"], grounded_fields["instrument_name"])
    timeframe = timeframe_label(grounded_fields["timeframes"])
    platform_visible = "tradingview" in ascii_fold(visible_ocr_text(annotation)).lower() or "TradingView header or watermark" in key_visual_elements

    if asset_type == "chart":
        lead = "TradingView chart" if platform_visible else "Chart"
    elif asset_type:
        lead = asset_type.replace("_", " ").capitalize()
    else:
        lead = "Trading-related visual"

    if subject != "trading-related" and timeframe:
        opening = f"{lead} for {subject} on {timeframe}."
    elif subject != "trading-related":
        opening = f"{lead} for {subject}."
    elif timeframe:
        opening = f"{lead} with visible timeframe {timeframe}."
    else:
        opening = f"{lead}."

    sentences = [opening]
    summary_elements = [
        value
        for value in key_visual_elements
        if value not in {"TradingView header or watermark", "instrument label", "timeframe label", "venue label", "visible text overlay"}
    ]
    if summary_elements:
        sentences.append("Visible elements include " + render_list(summary_elements[:4]) + ".")
    if grounded_fields["visible_text"]:
        sentences.append(f'Visible label text includes "{grounded_fields["visible_text"]}".')
    return " ".join(sentences[:3])


def build_training_response(annotation: dict[str, Any]) -> dict[str, Any]:
    grounded_fields = derive_grounded_fields(annotation)
    key_visual_elements = compact_visual_elements(annotation)
    asset_type = repaired_asset_type(annotation, grounded_fields, key_visual_elements)
    limitations = compact_limitations(annotation)

    response: dict[str, Any] = {}
    if asset_type:
        response["asset_type"] = asset_type
    if grounded_fields["primary_symbol"]:
        response["primary_symbol"] = grounded_fields["primary_symbol"]
    if grounded_fields["instrument_name"]:
        response["instrument_name"] = grounded_fields["instrument_name"]
    if grounded_fields["venue"]:
        response["venue"] = grounded_fields["venue"]
    if grounded_fields["timeframes"]:
        response["timeframes"] = grounded_fields["timeframes"]

    short_caption = build_short_caption(asset_type or "visual", grounded_fields, key_visual_elements)
    if short_caption:
        response["short_caption"] = short_caption

    visual_summary = build_visual_summary(asset_type or "visual", grounded_fields, key_visual_elements, annotation)
    if visual_summary:
        response["visual_summary"] = visual_summary

    if grounded_fields["visible_text"]:
        response["visible_text"] = grounded_fields["visible_text"]
    if key_visual_elements:
        response["key_visual_elements"] = key_visual_elements
    if limitations:
        response["limitations"] = limitations

    return response


def build_training_pair_payload(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "task_type": "strict_image_to_json",
        "response": build_training_response(annotation),
    }


def should_export_annotation(
    annotation: dict[str, Any],
    min_quality: str,
    require_llm: bool,
    allow_review_required: bool,
) -> tuple[bool, str]:
    if annotation.get("pair_type") != "visual_asset":
        return False, "not_visual_asset"
    if require_llm and not isinstance(annotation.get("llm_enrichment"), dict):
        return False, "missing_llm_enrichment"
    if not allow_review_required and bool(annotation.get("review_required")):
        return False, "review_required"
    if QUALITY_RANKS[annotation_quality(annotation)] < QUALITY_RANKS[min_quality]:
        return False, f"quality_below_{min_quality}"
    if require_llm:
        confidence = llm_confidence(annotation)
        if confidence and QUALITY_RANKS[confidence] < QUALITY_RANKS["medium"]:
            return False, "llm_confidence_low"

    response = build_training_response(annotation)
    if not response.get("short_caption"):
        return False, "missing_short_caption"
    if not response.get("visual_summary"):
        return False, "missing_visual_summary"
    if not any(response.get(field_name) for field_name in ("visible_text", "primary_symbol", "instrument_name", "venue", "timeframes")):
        return False, "missing_visible_signal"
    return True, "ok"


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy_file(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def export_training_pairs(
    glob_pattern: str,
    output_dir: Path,
    index_path: Path,
    manifest_path: Path,
    min_quality: str,
    require_llm: bool,
    allow_review_required: bool,
) -> dict[str, Any]:
    reset_output_dir(output_dir)
    (output_dir / TRAINING_PROMPT_FILENAME).write_text(TRAINING_PROMPT + "\n", encoding="utf-8")

    index_rows: list[dict[str, Any]] = []
    skipped_counts: dict[str, int] = {}
    exported_count = 0
    link_modes: dict[str, int] = {}

    for annotation_path in sorted(ROOT.glob(glob_pattern)):
        annotation = load_json(annotation_path)
        should_export, reason = should_export_annotation(
            annotation=annotation,
            min_quality=min_quality,
            require_llm=require_llm,
            allow_review_required=allow_review_required,
        )
        if not should_export:
            skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
            continue

        image_path = ROOT / str(annotation.get("image_path", ""))
        if not image_path.is_file():
            skipped_counts["missing_image"] = skipped_counts.get("missing_image", 0) + 1
            continue

        pair_basename = image_path.stem
        output_image_path = output_dir / image_path.name
        output_json_path = output_dir / f"{pair_basename}.json"
        link_mode = link_or_copy_file(image_path, output_image_path)
        link_modes[link_mode] = link_modes.get(link_mode, 0) + 1

        payload = build_training_pair_payload(annotation)
        response = payload["response"]
        write_json(output_json_path, payload)

        index_rows.append(
            {
                "id": annotation.get("id"),
                "image_path": root_relative(output_image_path),
                "json_path": root_relative(output_json_path),
                "source_image_path": annotation.get("image_path"),
                "source_annotation_path": root_relative(annotation_path),
                "source_pdf": annotation.get("source_pdf"),
                "page_number": annotation.get("page_number"),
                "asset_index": annotation.get("asset_index"),
                "asset_type": annotation.get("asset_type"),
                "exported_asset_type": response.get("asset_type"),
                "exported_primary_symbol": response.get("primary_symbol"),
                "exported_timeframes": response.get("timeframes", []),
                "annotation_quality": annotation_quality(annotation),
                "has_llm_enrichment": isinstance(annotation.get("llm_enrichment"), dict),
                "review_required": bool(annotation.get("review_required")),
                "link_mode": link_mode,
            }
        )
        exported_count += 1

    index_count = write_jsonl(index_path, index_rows)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_glob": glob_pattern,
        "output_dir": root_relative(output_dir),
        "index_path": root_relative(index_path),
        "instruction_path": root_relative(output_dir / TRAINING_PROMPT_FILENAME),
        "grounding_profile": "strict_visible_grounding",
        "min_quality": min_quality,
        "require_llm": require_llm,
        "allow_review_required": allow_review_required,
        "exported_pairs": exported_count,
        "index_rows": index_count,
        "skipped": skipped_counts,
        "image_link_modes": link_modes,
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = export_training_pairs(
        glob_pattern=args.glob_pattern,
        output_dir=Path(args.output_dir).resolve(),
        index_path=Path(args.index_path).resolve(),
        manifest_path=Path(args.manifest_path).resolve(),
        min_quality=args.min_quality,
        require_llm=args.require_llm,
        allow_review_required=args.allow_review_required,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
