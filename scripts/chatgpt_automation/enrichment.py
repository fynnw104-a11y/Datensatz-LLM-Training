from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from text_normalization import normalize_typographic_punctuation, repair_common_mojibake

PROMPT_SCHEMA_VERSION = "1.0"


def _normalize_prompt_value(value: Any) -> Any:
    if isinstance(value, str):
        return repair_common_mojibake(value)
    if isinstance(value, list):
        return [_normalize_prompt_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_prompt_value(inner) for key, inner in value.items()}
    return value


def build_multimodal_description_prompt(annotation: dict[str, Any], language: str = "en") -> str:
    language = language.strip().lower() or "en"
    output_language = "German" if language == "de" else "English"
    context_payload = {
        "annotation_id": annotation.get("id"),
        "asset_type": annotation.get("asset_type"),
        "page_type": annotation.get("page_type"),
        "summary": annotation.get("summary"),
        "description": annotation.get("description"),
        "context_heading": annotation.get("context_heading"),
        "context_text": annotation.get("context_text"),
        "ocr_text": annotation.get("ocr_text"),
        "combined_text": annotation.get("combined_text"),
        "primary_symbol": annotation.get("primary_symbol"),
        "instrument_name": annotation.get("instrument_name"),
        "venue": annotation.get("venue"),
        "timeframes": annotation.get("timeframes"),
        "bias": annotation.get("bias"),
        "direction": annotation.get("direction"),
        "setup_status": annotation.get("setup_status"),
        "trade_levels": annotation.get("trade_levels"),
        "trading_concepts": annotation.get("trading_concepts"),
        "labels": annotation.get("labels"),
    }
    context_payload = _normalize_prompt_value(context_payload)
    context_json = json.dumps(context_payload, ensure_ascii=False, indent=2)
    return (
        "You are creating training-grade image descriptions for a LoRA or vision-language fine-tuning dataset.\n"
        "The attached image is the primary source of truth.\n"
        "Use the supplied OCR/context only when it clearly matches what is plausible in the image.\n"
        "Do not hallucinate unreadable numbers, indicators, labels, or trading claims.\n"
        f"Write all natural-language fields in {output_language}.\n"
        "Return JSON only. No markdown. No code fences.\n\n"
        "{\n"
        '  "short_caption": "8-20 words",\n'
        '  "visual_summary": "1-3 sentences, image-grounded only",\n'
        '  "context_augmented_summary": "1-4 sentences, may use matching OCR/context",\n'
        '  "key_visual_elements": ["element_1", "element_2"],\n'
        '  "limitations": ["unclear_text", "cropped_axis"],\n'
        '  "visible_text": "short literal transcription of the most legible visible text",\n'
        '  "training_tags": ["chart", "candlesticks", "price_scale"],\n'
        '  "confidence": "high|medium|low"\n'
        "}\n\n"
        "Focus on what makes the image useful for training:\n"
        "- visible chart structure, panels, axes, legend areas, candles, lines, annotations, arrows, highlighted zones\n"
        "- visible symbol, timeframe, venue, labels, indicators, performance metrics, notes\n"
        "- whether the crop looks like a trading chart, a performance panel, or another market-related figure\n"
        "- for visible_text, copy one short contiguous snippet exactly as seen; do not join separate fragments with ellipses\n"
        "- mention uncertainty explicitly in limitations when text or details are unreadable\n\n"
        "Additional extracted context:\n"
        f"{context_json}"
    )


def normalize_llm_description(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = payload or {}

    def _text(key: str) -> str:
        value = payload.get(key)
        return normalize_typographic_punctuation(value).strip() if isinstance(value, str) else ""

    def _string_list(key: str) -> list[str]:
        value = payload.get(key)
        if not isinstance(value, list):
            return []
        items: list[str] = []
        for item in value:
            text = normalize_typographic_punctuation(str(item)).strip()
            if text:
                items.append(text)
        return items

    confidence = _text("confidence").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    return {
        "short_caption": _text("short_caption"),
        "visual_summary": _text("visual_summary"),
        "context_augmented_summary": _text("context_augmented_summary"),
        "key_visual_elements": _string_list("key_visual_elements"),
        "limitations": _string_list("limitations"),
        "visible_text": _text("visible_text"),
        "training_tags": _string_list("training_tags"),
        "confidence": confidence,
    }


def _first_non_empty(*values: object) -> str:
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return ""


def _fallback_string_list(preferred: list[str], existing_value: object) -> list[str]:
    if preferred:
        return preferred
    if not isinstance(existing_value, list):
        return []

    preserved: list[str] = []
    for item in existing_value:
        text = str(item).strip()
        if text:
            preserved.append(text)
    return preserved


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", repair_common_mojibake(value))
    return normalized.encode("ascii", "ignore").decode("ascii")


def _has_ocr_encoding_artifacts(text: object) -> bool:
    value = str(text or "")
    if any(marker in value for marker in ("\ufffd", "Ã", "Â", "â€", "Ãƒ", "Ã‚", "Ã¢â‚¬", "Ã¯Â¿Â½")):
        return True
    return any(0x80 <= ord(character) <= 0x9F for character in value)


def _clean_ocr_lines(text: object) -> list[str]:
    if not isinstance(text, str):
        return []
    lines: list[str] = []
    for raw_line in repair_common_mojibake(text).splitlines():
        cleaned = normalize_typographic_punctuation(raw_line).strip()
        cleaned = " ".join(cleaned.split())
        if cleaned:
            lines.append(cleaned)
    return lines


def _visible_text_line_score(line: str, annotation: dict[str, Any]) -> int:
    score = 0
    upper_line = line.upper()
    lower_line = line.lower()

    primary_symbol = str(annotation.get("primary_symbol") or "").upper()
    instrument_name = normalize_typographic_punctuation(str(annotation.get("instrument_name") or "")).lower()
    venue = str(annotation.get("venue") or "").upper()
    timeframes = [str(item).upper() for item in annotation.get("timeframes", []) if isinstance(item, str)]

    if primary_symbol and primary_symbol in upper_line:
        score += 4
    if instrument_name and any(token in lower_line for token in instrument_name.split() if len(token) >= 4):
        score += 3
    if venue and venue in upper_line:
        score += 2
    if "/" in line:
        score += 2
    if any(timeframe in upper_line for timeframe in timeframes):
        score += 2
    if any(
        timeframe.startswith("H") and timeframe[1:].isdigit() and re.search(rf"\b{re.escape(timeframe[1:])}h\b", lower_line)
        for timeframe in timeframes
    ):
        score += 2
    if any(
        timeframe.startswith("M") and timeframe[1:].isdigit() and re.search(rf"\b{re.escape(timeframe[1:])}\b", lower_line)
        for timeframe in timeframes
    ):
        score += 1
    if "bos" in lower_line:
        score += 1
    if "freigegeben" in lower_line:
        score -= 2
    if "tradingview.com" in lower_line:
        score -= 1
    return score


def _fallback_visible_text(annotation: dict[str, Any]) -> str:
    lines = _clean_ocr_lines(annotation.get("ocr_text"))
    if not lines:
        return ""

    ranked_lines = sorted(
        enumerate(lines),
        key=lambda item: (-_visible_text_line_score(item[1], annotation), item[0], len(item[1])),
    )
    for _index, line in ranked_lines:
        if len(line) >= 6:
            return line
    return lines[0]


def _coerce_visible_text(candidate: str, annotation: dict[str, Any]) -> str:
    normalized_candidate = normalize_typographic_punctuation(repair_common_mojibake(candidate)).strip()
    if normalized_candidate and "..." not in normalized_candidate and "…" not in normalized_candidate and "\n" not in normalized_candidate:
        return normalized_candidate
    fallback = _fallback_visible_text(annotation)
    return fallback or normalized_candidate


def _build_crop_clean_text_base(annotation: dict[str, Any]) -> list[str]:
    lines: list[str] = [f"Asset type: {annotation.get('asset_type', 'unknown')}"]
    primary_symbol = annotation.get("primary_symbol")
    instrument_name = annotation.get("instrument_name")
    venue = annotation.get("venue")
    timeframes = annotation.get("timeframes")

    if primary_symbol:
        lines.append(f"Instrument: {primary_symbol}")
    if instrument_name and instrument_name != primary_symbol:
        lines.append(f"Instrument label: {instrument_name}")
    if venue:
        lines.append(f"Venue: {venue}")
    if isinstance(timeframes, list) and timeframes:
        lines.append("Timeframes: " + ", ".join(str(item) for item in timeframes if str(item).strip()))
    return lines


def _refresh_visible_in_crop_clean_text(existing_clean_text: object, annotation: dict[str, Any], visible_text: str) -> str:
    if isinstance(existing_clean_text, str) and existing_clean_text.strip():
        lines = [line.strip() for line in existing_clean_text.splitlines() if line.strip()]
    else:
        lines = _build_crop_clean_text_base(annotation)

    retained_lines = [line for line in lines if not line.lower().startswith("visible labels:")]
    normalized_visible_text = visible_text.strip()
    if normalized_visible_text:
        retained_lines.append(f"Visible labels: {normalized_visible_text}")
    return "\n".join(retained_lines).strip()


def _llm_reports_uncertainty(limitations: list[str]) -> bool:
    uncertainty_markers = (
        "unclear",
        "unreadable",
        "illegible",
        "cropped",
        "difficult to verify",
        "inaccurate",
        "approximate",
        "partially",
    )
    return any(any(marker in item.lower() for marker in uncertainty_markers) for item in limitations)


def _build_enrichment_review_reasons(annotation: dict[str, Any], description: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    labels = annotation.get("labels", {})
    if isinstance(labels, dict) and labels.get("text_density") == "low":
        reasons.append("low_text_density")

    page_type_confidence = annotation.get("page_type_confidence")
    if isinstance(page_type_confidence, (int, float)) and float(page_type_confidence) < 0.6:
        reasons.append("low_page_type_confidence")

    if _has_ocr_encoding_artifacts(annotation.get("ocr_text")):
        reasons.append("ocr_encoding_artifacts")
    if not annotation.get("primary_symbol"):
        reasons.append("missing_primary_symbol")
    if not annotation.get("timeframes"):
        reasons.append("missing_timeframe")

    confidence = str(description.get("confidence") or "").strip().lower()
    if confidence in {"medium", "low"}:
        reasons.append(f"llm_confidence_{confidence}")
    if _llm_reports_uncertainty(description.get("limitations", [])):
        reasons.append("llm_reported_uncertainty")

    if annotation.get("review_required"):
        reasons.insert(0, "auto_review_flag")
    return _unique_preserving_order(reasons)


def _infer_enrichment_annotation_quality(annotation: dict[str, Any], description: dict[str, Any]) -> str:
    score = 0
    if annotation.get("primary_symbol"):
        score += 1
    if annotation.get("timeframes"):
        score += 1
    if annotation.get("venue"):
        score += 1
    if description.get("visible_text"):
        score += 1

    confidence = str(description.get("confidence") or "").strip().lower()
    if confidence == "high":
        score += 1
    elif confidence == "low":
        score -= 1

    page_type_confidence = annotation.get("page_type_confidence")
    if isinstance(page_type_confidence, (int, float)) and float(page_type_confidence) < 0.6:
        score -= 1
    if _has_ocr_encoding_artifacts(annotation.get("ocr_text")):
        score -= 1
    if _llm_reports_uncertainty(description.get("limitations", [])):
        score -= 1

    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def build_enriched_clean_text(description: dict[str, Any], annotation: dict[str, Any]) -> str:
    lines: list[str] = []
    short_caption = description.get("short_caption")
    visual_summary = description.get("visual_summary")
    context_summary = description.get("context_augmented_summary")
    visible_text = description.get("visible_text")
    tags = description.get("training_tags", [])
    limitations = description.get("limitations", [])
    confidence = description.get("confidence")

    if short_caption:
        lines.append(f"Short caption: {short_caption}")
    if visual_summary:
        lines.append(f"Visual summary: {visual_summary}")
    if context_summary:
        lines.append(f"Context-aware summary: {context_summary}")
    if visible_text:
        lines.append(f"Visible text: {visible_text}")
    if tags:
        lines.append("Training tags: " + ", ".join(tags))
    if confidence:
        lines.append(f"Confidence: {confidence}")
    if limitations:
        lines.append("Limitations: " + "; ".join(limitations))

    if annotation.get("primary_symbol"):
        lines.append(f"Primary symbol: {annotation['primary_symbol']}")
    if annotation.get("timeframes"):
        lines.append("Timeframes: " + ", ".join(annotation["timeframes"]))
    if annotation.get("venue"):
        lines.append(f"Venue: {annotation['venue']}")

    return "\n".join(lines).strip()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


def _merge_extraction_methods(methods: list[str], extra_method: str) -> list[str]:
    merged = [str(item) for item in methods if isinstance(item, str) and item.strip()]
    if extra_method not in merged:
        merged.append(extra_method)
    return merged


def _coerce_target_description_block(target_json: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    description_block = deepcopy(target_json.get("description", {}))
    if isinstance(description_block, dict):
        return description_block

    legacy_context_summary = _first_non_empty(
        description_block,
        annotation.get("description"),
        annotation.get("summary"),
        annotation.get("caption"),
    )
    legacy_short_caption = _first_non_empty(annotation.get("caption"), annotation.get("summary"))
    return {
        "short_caption": legacy_short_caption,
        "visual_summary": "",
        "context_augmented_summary": legacy_context_summary,
        "key_visual_elements": [],
        "limitations": [],
    }


def apply_asset_llm_enrichment(
    annotation: dict[str, Any],
    response_payload: dict[str, Any],
    raw_response_text: str,
    prompt: str,
    language: str,
    model_slug: str | None,
    conversation_url: str | None,
) -> dict[str, Any]:
    description = normalize_llm_description(response_payload)
    updated = deepcopy(annotation)
    raw_target_json = updated.get("target_json", {})
    target_json = deepcopy(raw_target_json) if isinstance(raw_target_json, dict) else {}
    description_block = _coerce_target_description_block(target_json, updated)

    short_caption = _first_non_empty(description["short_caption"], updated.get("caption"), updated.get("summary"))
    visual_summary = _first_non_empty(description["visual_summary"], description_block.get("visual_summary"))
    context_summary = _first_non_empty(
        description["context_augmented_summary"],
        description_block.get("context_augmented_summary"),
        updated.get("description"),
        visual_summary,
    )
    merged_description = {
        **description,
        "short_caption": short_caption,
        "visual_summary": visual_summary,
        "context_augmented_summary": context_summary,
        "visible_text": _coerce_visible_text(description["visible_text"], updated),
        "key_visual_elements": _fallback_string_list(
            description["key_visual_elements"],
            description_block.get("key_visual_elements"),
        ),
        "limitations": _fallback_string_list(
            description["limitations"],
            description_block.get("limitations"),
        ),
    }
    clean_text = build_enriched_clean_text(merged_description, updated)
    top_level_summary = _first_non_empty(visual_summary, short_caption)
    top_level_description = _first_non_empty(context_summary, visual_summary, short_caption)

    updated["caption"] = short_caption
    updated["summary"] = top_level_summary
    updated["description"] = top_level_description
    if clean_text:
        updated["clean_text"] = clean_text

    extraction_methods = _merge_extraction_methods(list(updated.get("extraction_methods", [])), "chatgpt_browser_llm")
    updated["extraction_methods"] = extraction_methods

    description_block["short_caption"] = short_caption
    description_block["visual_summary"] = visual_summary
    description_block["context_augmented_summary"] = context_summary
    description_block["key_visual_elements"] = merged_description["key_visual_elements"]
    description_block["limitations"] = merged_description["limitations"]
    target_json["description"] = description_block

    observed = deepcopy(target_json.get("observed", {}))
    if isinstance(observed, dict):
        visible_in_crop = deepcopy(observed.get("visible_in_crop", {}))
        if isinstance(visible_in_crop, dict):
            refreshed_crop_clean_text = _refresh_visible_in_crop_clean_text(
                visible_in_crop.get("clean_text"),
                updated,
                merged_description["visible_text"],
            )
            if refreshed_crop_clean_text:
                visible_in_crop["clean_text"] = refreshed_crop_clean_text
            observed["visible_in_crop"] = visible_in_crop
            target_json["observed"] = observed

    provenance = deepcopy(target_json.get("provenance", {}))
    provenance["extraction_methods"] = _merge_extraction_methods(
        list(provenance.get("extraction_methods", [])),
        "chatgpt_browser_llm",
    )
    review_reasons = _build_enrichment_review_reasons(updated, merged_description)
    review_required = bool(review_reasons)
    updated["review_required"] = review_required

    quality = deepcopy(provenance.get("quality", {}))
    if not isinstance(quality, dict):
        quality = {}
    quality["annotation_quality"] = _infer_enrichment_annotation_quality(updated, merged_description)
    page_type_confidence = updated.get("page_type_confidence")
    if isinstance(page_type_confidence, (int, float)):
        quality["page_type_confidence"] = float(page_type_confidence)
    provenance["quality"] = quality

    review = deepcopy(provenance.get("review", {}))
    if not isinstance(review, dict):
        review = {}
    review["required"] = review_required
    review["reasons"] = review_reasons
    provenance["review"] = review
    target_json["provenance"] = provenance
    updated["target_json"] = target_json

    updated["llm_enrichment"] = {
        "source": "chatgpt_browser_automation",
        "provider": "chatgpt_web",
        "schema_version": PROMPT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "model_slug": model_slug,
        "conversation_url": conversation_url,
        "prompt_hash": _prompt_hash(prompt),
        "raw_response_text": repair_common_mojibake(raw_response_text),
        "structured_response": merged_description,
    }
    return updated
