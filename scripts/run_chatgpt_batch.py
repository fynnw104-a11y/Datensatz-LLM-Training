from __future__ import annotations

import argparse
from pathlib import Path

from chatgpt_automation.batch import load_jobs, run_batch_jobs, write_results
from chatgpt_automation.client import ChatGPTClient
from chatgpt_automation.config import load_config
from chatgpt_automation.selectors import SelectorCatalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run generic ChatGPT browser batch jobs from JSONL.")
    parser.add_argument("--input", required=True, help="Path to a JSONL file with batch jobs.")
    parser.add_argument("--output", required=True, help="Path to the JSONL result file.")
    parser.add_argument("--config", default=None, help="Optional path to ChatGPT/config.json.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    keep_browser_open = base_config.keep_browser_open if args.keep_browser_open is None else args.keep_browser_open
    config = base_config.with_overrides(keep_browser_open=keep_browser_open)
    selector_catalog = SelectorCatalog.from_files(config.selector_files)
    jobs = load_jobs(Path(args.input).resolve())

    client = ChatGPTClient(config=config, selector_catalog=selector_catalog)
    try:
        rows = run_batch_jobs(client=client, jobs=jobs, allow_manual_login=args.manual_login)
    finally:
        if not config.keep_browser_open:
            client.close()
        else:
            print("Automation browser left open for manual inspection.", flush=True)

    write_results(Path(args.output).resolve(), rows)
    print(f"Wrote {len(rows)} batch results to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
