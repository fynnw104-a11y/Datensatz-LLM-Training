from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import ChatGPTClient


@dataclass(frozen=True)
class BatchJob:
    job_id: str
    prompt: str
    attachments: tuple[Path, ...]
    metadata: dict[str, Any]
    new_chat: bool = False


def load_jobs(path: Path) -> list[BatchJob]:
    jobs: list[BatchJob] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid batch job on line {line_number}: expected a JSON object.")
            job_id = str(payload.get("id", "")).strip()
            prompt = str(payload.get("prompt", "")).strip()
            if not job_id or not prompt:
                raise ValueError(f"Invalid batch job on line {line_number}: missing id or prompt.")
            attachments = tuple(Path(item) for item in payload.get("attachments", []))
            metadata = payload.get("metadata", {})
            jobs.append(
                BatchJob(
                    job_id=job_id,
                    prompt=prompt,
                    attachments=attachments,
                    metadata=metadata if isinstance(metadata, dict) else {"value": metadata},
                    new_chat=bool(payload.get("new_chat", False)),
                )
            )
    return jobs


def extract_json_fragment(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        parts = stripped.split("```")
        for part in parts:
            candidate = part.strip()
            if not candidate:
                continue
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                return candidate

    for opening, closing in (("{", "}"), ("[", "]")):
        start = stripped.find(opening)
        if start < 0:
            continue
        depth = 0
        for index in range(start, len(stripped)):
            char = stripped[index]
            if char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return stripped[start : index + 1]
    return None


def parse_json_response(text: str) -> dict[str, Any] | list[Any] | None:
    candidate = extract_json_fragment(text)
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    return None


def write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_batch_jobs(
    client: ChatGPTClient,
    jobs: list[BatchJob],
    allow_manual_login: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_jobs = len(jobs)
    for index, job in enumerate(jobs, start=1):
        print(f"[{index}/{total_jobs}] Processing ChatGPT job {job.job_id}...", flush=True)
        started_at = time.time()
        status = "ok"
        assistant_text = ""
        assistant_json: dict[str, Any] | list[Any] | None = None
        conversation_url = None
        model_slug = None
        error = None
        try:
            response = client.run_prompt(
                prompt=job.prompt,
                attachments=list(job.attachments),
                new_chat=job.new_chat,
                allow_manual_login=allow_manual_login,
            )
            assistant_text = response.text
            assistant_json = parse_json_response(response.text)
            conversation_url = response.url
            model_slug = response.model_slug
        except Exception as exc:
            status = "error"
            error = str(exc)

        if status == "ok":
            print(f"[{index}/{total_jobs}] ChatGPT job {job.job_id} completed.", flush=True)
        else:
            print(f"[{index}/{total_jobs}] ChatGPT job {job.job_id} failed: {error}", flush=True)

        rows.append(
            {
                "id": job.job_id,
                "status": status,
                "prompt": job.prompt,
                "attachments": [str(path) for path in job.attachments],
                "assistant_text": assistant_text,
                "assistant_json": assistant_json,
                "conversation_url": conversation_url,
                "model_slug": model_slug,
                "duration_seconds": round(time.time() - started_at, 2),
                "error": error,
                "metadata": job.metadata,
            }
        )
    return rows
