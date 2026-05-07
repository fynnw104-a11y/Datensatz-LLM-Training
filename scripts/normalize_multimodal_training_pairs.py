from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from export_multimodal_training_pairs import (
    DEFAULT_INDEX_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_DIR,
    ROOT,
    TRAINING_PROMPT,
    TRAINING_PROMPT_FILENAME,
    build_training_pair_payload,
    load_json,
    root_relative,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize existing multimodal training pairs in place without re-exporting images."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory containing training pair JSON files.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="JSONL index describing existing training pairs.")
    parser.add_argument(
        "--manifest-path",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Manifest JSON summarizing the existing training pair set.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def resolve_training_json_path(output_dir: Path, row: dict[str, Any]) -> Path:
    relative_path = str(row.get("json_path") or "").strip()
    if relative_path:
        candidate = ROOT / relative_path
        if candidate.is_file():
            return candidate
    fallback = output_dir / f"{str(row.get('id') or '').strip()}.json"
    if fallback.is_file():
        return fallback
    raise RuntimeError(f"Could not resolve training pair JSON for row: {row}")


def resolve_source_annotation_path(training_json_path: Path, row: dict[str, Any]) -> Path:
    relative_path = str(row.get("source_annotation_path") or "").strip()
    candidates: list[Path] = []
    if relative_path:
        candidates.append(ROOT / relative_path)
    candidates.append(ROOT / "data" / "processed" / "multimodal" / "pairs" / training_json_path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"Could not resolve source annotation for {training_json_path.name}")


def normalize_training_pairs(output_dir: Path, index_path: Path, manifest_path: Path) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise RuntimeError(f"Training pair directory does not exist: {output_dir}")

    rows = load_jsonl(index_path)
    if not rows:
        raise RuntimeError(f"Training pair index is missing or empty: {index_path}")

    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        training_json_path = resolve_training_json_path(output_dir, row)
        source_annotation_path = resolve_source_annotation_path(training_json_path, row)
        annotation = load_json(source_annotation_path)
        payload = build_training_pair_payload(annotation)
        write_json(training_json_path, payload)

        response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
        updated_row = dict(row)
        updated_row["json_path"] = root_relative(training_json_path)
        updated_row["source_annotation_path"] = root_relative(source_annotation_path)
        updated_row["exported_asset_type"] = response.get("asset_type")
        updated_row["exported_primary_symbol"] = response.get("primary_symbol")
        updated_row["exported_timeframes"] = response.get("timeframes", [])
        updated_rows.append(updated_row)

    write_jsonl(index_path, updated_rows)
    (output_dir / TRAINING_PROMPT_FILENAME).write_text(TRAINING_PROMPT + "\n", encoding="utf-8")

    manifest: dict[str, Any] = load_json(manifest_path) if manifest_path.is_file() else {}
    manifest["output_dir"] = root_relative(output_dir)
    manifest["index_path"] = root_relative(index_path)
    manifest["instruction_path"] = root_relative(output_dir / TRAINING_PROMPT_FILENAME)
    manifest["index_rows"] = len(updated_rows)
    manifest["exported_pairs"] = len(updated_rows)
    manifest["normalized_pairs"] = len(updated_rows)
    manifest["normalized_at"] = datetime.now(timezone.utc).isoformat()
    manifest["normalizer"] = "strict_visible_grounding_consistency_v1"
    write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = normalize_training_pairs(
        output_dir=Path(args.output_dir).resolve(),
        index_path=Path(args.index_path).resolve(),
        manifest_path=Path(args.manifest_path).resolve(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
