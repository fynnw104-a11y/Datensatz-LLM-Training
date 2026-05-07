from __future__ import annotations

import argparse
import json
import os
import shutil
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
    "Analyze the attached trading-related image and return a concise JSON object grounded only in visible "
    "content. Use only details that are actually visible in the image."
)
QUALITY_RANKS = {"low": 0, "medium": 1, "high": 2}


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
        cleaned = item.strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


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


def visible_text(annotation: dict[str, Any]) -> str:
    return first_non_empty(
        get_nested(annotation, "llm_enrichment", "structured_response", "visible_text"),
        extract_visible_labels_line(get_nested(annotation, "target_json", "observed", "visible_in_crop", "clean_text")),
        annotation.get("ocr_text"),
    )


def compact_visual_elements(annotation: dict[str, Any]) -> list[str]:
    return normalize_string_list(
        get_nested(annotation, "llm_enrichment", "structured_response", "key_visual_elements")
    ) or normalize_string_list(
        get_nested(annotation, "target_json", "description", "key_visual_elements")
    )


def compact_limitations(annotation: dict[str, Any]) -> list[str]:
    return normalize_string_list(
        get_nested(annotation, "llm_enrichment", "structured_response", "limitations")
    ) or normalize_string_list(
        get_nested(annotation, "target_json", "description", "limitations")
    )


def build_training_response(annotation: dict[str, Any]) -> dict[str, Any]:
    normalized_fields = get_nested(annotation, "target_json", "observed", "visible_in_crop", "normalized_fields")
    if not isinstance(normalized_fields, dict):
        normalized_fields = {}

    response: dict[str, Any] = {}
    asset_type = first_non_empty(annotation.get("asset_type"))
    if asset_type and asset_type != "unknown":
        response["asset_type"] = asset_type

    primary_symbol = first_non_empty(normalized_fields.get("primary_symbol"))
    if primary_symbol:
        response["primary_symbol"] = primary_symbol

    instrument_name = first_non_empty(normalized_fields.get("instrument_name"))
    if instrument_name:
        response["instrument_name"] = instrument_name

    venue = first_non_empty(normalized_fields.get("venue"))
    if venue:
        response["venue"] = venue

    timeframes = normalize_string_list(normalized_fields.get("timeframes"))
    if timeframes:
        response["timeframes"] = timeframes

    short_caption = first_non_empty(
        get_nested(annotation, "llm_enrichment", "structured_response", "short_caption"),
        get_nested(annotation, "target_json", "description", "short_caption"),
        annotation.get("caption"),
        annotation.get("summary"),
    )
    if short_caption:
        response["short_caption"] = short_caption

    visual_summary = first_non_empty(
        get_nested(annotation, "llm_enrichment", "structured_response", "visual_summary"),
        get_nested(annotation, "target_json", "description", "visual_summary"),
        annotation.get("summary"),
        annotation.get("caption"),
    )
    if visual_summary:
        response["visual_summary"] = visual_summary

    extracted_visible_text = visible_text(annotation)
    if extracted_visible_text:
        response["visible_text"] = extracted_visible_text

    key_visual_elements = compact_visual_elements(annotation)
    if key_visual_elements:
        response["key_visual_elements"] = key_visual_elements

    limitations = compact_limitations(annotation)
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
