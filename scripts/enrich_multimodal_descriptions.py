from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatgpt_automation.batch import BatchJob, run_batch_jobs, write_results
from chatgpt_automation.client import ChatGPTClient
from chatgpt_automation.config import ROOT, load_config
from chatgpt_automation.enrichment import apply_asset_llm_enrichment, build_multimodal_description_prompt
from chatgpt_automation.selectors import SelectorCatalog

DEFAULT_ASSET_GLOB = "data/processed/multimodal/annotations/*/assets/*.json"
DEFAULT_RUNS_DIR = ROOT / "data" / "processed" / "chatgpt_runs"
DEFAULT_MAX_ASSETS_PER_CHAT = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Improve multimodal asset descriptions by sending image+context to ChatGPT via Selenium Stealth."
    )
    parser.add_argument("--config", default=None, help="Optional path to ChatGPT/config.json.")
    parser.add_argument("--glob", dest="glob_pattern", default=DEFAULT_ASSET_GLOB, help="Asset annotation glob pattern.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of annotations to process.")
    parser.add_argument("--language", default="en", choices=["en", "de"], help="Language for generated descriptions.")
    parser.add_argument("--output-dir", default=str(DEFAULT_RUNS_DIR), help="Directory for raw batch result logs.")
    parser.add_argument(
        "--skip-existing-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip annotations that already contain an llm_enrichment block.",
    )
    parser.add_argument(
        "--manual-login",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow manual login fallback if cookies/profile are not enough.",
    )
    parser.add_argument(
        "--keep-browser-open",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Leave the automation browser open after the run for manual inspection.",
    )
    parser.add_argument(
        "--max-assets-per-chat",
        type=int,
        default=DEFAULT_MAX_ASSETS_PER_CHAT,
        help="Reuse each ChatGPT conversation for up to N assets before starting a fresh one. Use 0 to keep one chat for the whole run.",
    )
    parser.add_argument(
        "--new-chat-per-asset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force a fresh ChatGPT conversation for every asset instead of reusing up to --max-assets-per-chat assets per chat.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run ChatGPT and write result logs, but do not edit JSONs.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def should_start_new_chat(job_index: int, max_assets_per_chat: int | None) -> bool:
    if job_index == 0:
        return True
    if max_assets_per_chat is None or max_assets_per_chat <= 0:
        return False
    return job_index % max_assets_per_chat == 0


def collect_annotation_jobs(
    glob_pattern: str,
    language: str,
    skip_existing_llm: bool,
    limit: int | None,
    max_assets_per_chat: int | None,
) -> tuple[list[BatchJob], dict[str, dict[str, Any]]]:
    jobs: list[BatchJob] = []
    context_by_id: dict[str, dict[str, Any]] = {}
    for annotation_path in sorted(ROOT.glob(glob_pattern)):
        annotation = load_json(annotation_path)
        if annotation.get("pair_type") != "visual_asset":
            continue
        if skip_existing_llm and isinstance(annotation.get("llm_enrichment"), dict):
            continue

        image_path = ROOT / str(annotation.get("image_path", ""))
        if not image_path.exists():
            continue

        prompt = build_multimodal_description_prompt(annotation, language=language)
        annotation_id = str(annotation.get("id", annotation_path.stem)).strip()
        jobs.append(
            BatchJob(
                job_id=annotation_id,
                prompt=prompt,
                attachments=(image_path,),
                metadata={
                    "annotation_path": str(annotation_path.resolve()),
                    "image_path": str(image_path.resolve()),
                },
                new_chat=should_start_new_chat(len(jobs), max_assets_per_chat),
            )
        )
        context_by_id[annotation_id] = {
            "annotation_path": annotation_path.resolve(),
            "annotation": annotation,
            "prompt": prompt,
        }
        if limit is not None and len(jobs) >= limit:
            break

    return jobs, context_by_id


def main() -> None:
    args = parse_args()
    max_assets_per_chat = 1 if args.new_chat_per_asset else args.max_assets_per_chat
    jobs, context_by_id = collect_annotation_jobs(
        glob_pattern=args.glob_pattern,
        language=args.language,
        skip_existing_llm=args.skip_existing_llm,
        limit=args.limit,
        max_assets_per_chat=max_assets_per_chat,
    )
    if not jobs:
        print("No matching asset annotations found for enrichment.")
        return

    base_config = load_config(args.config)
    keep_browser_open = base_config.keep_browser_open if args.keep_browser_open is None else args.keep_browser_open
    config = base_config.with_overrides(keep_browser_open=keep_browser_open)
    selector_catalog = SelectorCatalog.from_files(config.selector_files)
    client = ChatGPTClient(config=config, selector_catalog=selector_catalog)

    try:
        results = run_batch_jobs(client=client, jobs=jobs, allow_manual_login=args.manual_login)
    finally:
        if not config.keep_browser_open:
            client.close()
        else:
            print("Automation browser left open for manual inspection.", flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() / f"multimodal_enrichment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    write_results(results_path, results)

    updated_count = 0
    error_count = 0
    for row in results:
        if row.get("status") != "ok" or not isinstance(row.get("assistant_json"), dict):
            error_count += 1
            continue
        context = context_by_id.get(str(row["id"]))
        if context is None:
            error_count += 1
            continue

        updated_annotation = apply_asset_llm_enrichment(
            annotation=context["annotation"],
            response_payload=row["assistant_json"],
            raw_response_text=str(row.get("assistant_text", "")),
            prompt=context["prompt"],
            language=args.language,
            model_slug=row.get("model_slug"),
            conversation_url=row.get("conversation_url"),
        )

        if not args.dry_run:
            write_json(Path(context["annotation_path"]), updated_annotation)
        updated_count += 1

    print(f"Jobs: {len(results)}")
    print(f"Updated annotations: {updated_count}")
    print(f"Errors: {error_count}")
    print(f"Raw results: {results_path}")


if __name__ == "__main__":
    main()
