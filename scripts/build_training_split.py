from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CURATED_FILE = ROOT / "data" / "curated" / "training_examples.jsonl"
PROCESSED_DIR = ROOT / "data" / "processed"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            rows.append(row)
    return rows


def validate_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    for item in messages:
        if not isinstance(item, dict):
            return False
        if item.get("role") not in {"system", "user", "assistant"}:
            return False
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            return False
    return True


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(row.get("id"), str) or not row["id"].strip():
        errors.append("missing id")
    if not validate_messages(row.get("messages")):
        errors.append("invalid messages")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("missing metadata")
    else:
        if not isinstance(metadata.get("task_type"), str) or not metadata["task_type"].strip():
            errors.append("missing metadata.task_type")
        if not isinstance(metadata.get("source_path"), str) or not metadata["source_path"].strip():
            errors.append("missing metadata.source_path")
    return errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rows = load_jsonl(CURATED_FILE)
    if not rows:
        print(
            json.dumps(
                {
                    "message": "No curated training examples found.",
                    "expected_file": str(CURATED_FILE),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    invalid_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    for row in rows:
        errors = validate_row(row)
        if errors:
            invalid_rows.append({"id": row.get("id"), "errors": errors})
        else:
            valid_rows.append(row)

    if invalid_rows:
        (PROCESSED_DIR / "invalid_training_examples.json").write_text(
            json.dumps(invalid_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    random.Random(42).shuffle(valid_rows)
    split_index = max(1, int(len(valid_rows) * 0.9)) if len(valid_rows) > 1 else len(valid_rows)
    train_rows = valid_rows[:split_index]
    eval_rows = valid_rows[split_index:]

    write_jsonl(PROCESSED_DIR / "train.jsonl", train_rows)
    write_jsonl(PROCESSED_DIR / "eval.jsonl", eval_rows)

    print(
        json.dumps(
            {
                "valid_examples": len(valid_rows),
                "invalid_examples": len(invalid_rows),
                "train_examples": len(train_rows),
                "eval_examples": len(eval_rows),
                "invalid_report": str(PROCESSED_DIR / "invalid_training_examples.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

