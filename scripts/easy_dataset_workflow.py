from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chatgpt_automation.browser import (
    discover_browser_executable,
    launch_browser_for_manual_login,
    normalize_cookie_payload,
    require_selenium,
    set_cookie_payload,
)
from chatgpt_automation.config import DEFAULT_CONFIG_PATH, load_config
from export_multimodal_training_pairs import should_export_annotation
from prepare_dataset import RAW_PDFS_DIR, detect_ocr_runtime, import_pymupdf

CHATGPT_DIR = ROOT / "ChatGPT"
CHATGPT_CONFIG_PATH = DEFAULT_CONFIG_PATH
CHATGPT_CONFIG_EXAMPLE_PATH = CHATGPT_DIR / "config.example.json"
CHATGPT_RUNTIME_DIR = ROOT / ".runtime" / "chatgpt"
CURATED_FILE = ROOT / "data" / "curated" / "training_examples.jsonl"
MULTIMODAL_ASSET_GLOB = "data/processed/multimodal/pairs/*.json"
MULTIMODAL_TRAINING_PAIR_GLOB = "data/processed/multimodal/training_pairs/*.json"
MULTIMODAL_TRAINING_PAIR_MANIFEST_PATH = ROOT / "data" / "processed" / "multimodal" / "training_pairs_manifest.json"
PROCESSED_CHATGPT_RUNS_DIR = ROOT / "data" / "processed" / "chatgpt_runs"
DEFAULT_MAX_ASSETS_PER_CHAT = 20
SAFE_CONFIG_OVERRIDES = {
    "user_data_dir": "../.runtime/chatgpt/browser_profile",
    "cookies_file": "../.runtime/chatgpt/cookies/ChatGPT.json",
}
CHATGPT_AUTH_COOKIE_MARKERS = (
    "session-token",
    "session_token",
    "access-token",
    "access_token",
    "auth-token",
    "auth_token",
)


@dataclass(frozen=True)
class CheckResult:
    label: str
    status: str
    detail: str


def count_real_files(root: Path, glob_pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.glob(glob_pattern) if path.is_file() and path.name != ".gitkeep")


def copy_file_if_missing(source: Path, target: Path, force: bool = False) -> tuple[bool, Path | None]:
    if not source.exists():
        return False, None
    if target.exists() and not force:
        return False, target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and force:
        backup_name = target.with_name(f"{target.stem}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}{target.suffix}")
        shutil.copy2(target, backup_name)
    shutil.copy2(source, target)
    return True, target


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def launcher_name() -> str:
    return "START_HERE.bat" if platform.system().lower() == "windows" else "./START_HERE.sh"


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cookie_source(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return normalize_cookie_payload(payload)


def write_chatgpt_cookies(cookies: list[dict[str, object]], replace: bool = False, config_path: str | Path | None = None) -> int:
    config = load_config(config_path)
    if config.cookies_file is None:
        raise RuntimeError("In der ChatGPT-Konfig ist keine cookies_file gesetzt.")
    return set_cookie_payload(config.cookies_file, cookies, replace=replace)


def build_cookie_from_args(args: argparse.Namespace) -> dict[str, object]:
    cookie: dict[str, object] = {
        "name": args.name,
        "value": args.value,
        "path": args.path,
    }
    if args.domain:
        cookie["domain"] = args.domain
    if args.expiry is not None:
        cookie["expiry"] = int(args.expiry)
    if args.same_site:
        cookie["sameSite"] = args.same_site
    if args.secure is not None:
        cookie["secure"] = bool(args.secure)
    if args.http_only is not None:
        cookie["httpOnly"] = bool(args.http_only)
    return cookie


def chatgpt_cookie_db_path(config_path: str | Path | None = None) -> Path:
    config = load_config(config_path)
    if config.user_data_dir is None:
        raise RuntimeError("In der ChatGPT-Konfig ist kein user_data_dir gesetzt.")
    return config.user_data_dir / "Default" / "Cookies"


def list_project_profile_chatgpt_cookie_names(config_path: str | Path | None = None) -> list[str]:
    cookie_db = chatgpt_cookie_db_path(config_path)
    if not cookie_db.exists():
        return []

    names: list[str] = []
    try:
        connection = sqlite3.connect(f"file:{cookie_db}?mode=ro", uri=True)
    except sqlite3.Error:
        return []

    try:
        rows = connection.execute(
            """
            select host_key, name
            from cookies
            where host_key like '%chatgpt%' or host_key like '%openai%'
            order by host_key, name
            """
        )
        names = [f"{host}:{name}" for host, name in rows]
    except sqlite3.Error:
        return []
    finally:
        connection.close()

    return names


def has_chatgpt_auth_cookie(cookie_names: Iterable[str]) -> bool:
    lowered = [name.lower() for name in cookie_names]
    return any(marker in name for name in lowered for marker in CHATGPT_AUTH_COOKIE_MARKERS)


def bootstrap_chatgpt_login(config_path: str | Path | None = None) -> None:
    setup_chatgpt_config(force=False)
    config = load_config(config_path)
    if config.user_data_dir is None:
        raise RuntimeError("In der ChatGPT-Konfig ist kein user_data_dir gesetzt.")

    print_header("ChatGPT-Login vorbereiten")
    print(f"Profilordner: {config.user_data_dir}")
    print("Es wird ein normales Browserfenster geoeffnet. Logge dich dort komplett ein.")
    print("Wichtig: Erst schliessen, wenn du das normale ChatGPT-Eingabefeld siehst.")

    process = launch_browser_for_manual_login(config, url=config.login_url)
    try:
        process.wait(timeout=config.manual_login_timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Login-Browser wurde nach {int(config.manual_login_timeout_seconds)} Sekunden nicht geschlossen. "
            "Bitte Fenster nach dem Login schliessen und den Befehl erneut starten."
        ) from exc

    cookie_names = list_project_profile_chatgpt_cookie_names(config_path)
    if not cookie_names:
        raise RuntimeError(
            "Im Projektprofil wurden keine ChatGPT/OpenAI-Cookies gefunden. "
            "Der Login wurde wahrscheinlich nicht in diesem Profil gespeichert."
        )

    print("\nGefundene ChatGPT/OpenAI-Cookies im Projektprofil:")
    for name in cookie_names:
        print(f"- {name}")

    if not has_chatgpt_auth_cookie(cookie_names):
        raise RuntimeError(
            "Es wurden zwar ChatGPT-Cookies gefunden, aber kein erkennbares Session-Cookie. "
            "Bleib beim Login bis zur normalen ChatGPT-Startseite mit Eingabefeld, "
            "oder importiere einen Cookie-Export mit einem Session-Cookie ueber `import-cookies`."
        )

    print("\nSession-Cookie erkannt. Das Enrichment kann jetzt mit diesem Profil laufen.")


def collect_enrichment_target_paths(skip_existing_llm: bool, limit: int | None) -> list[Path]:
    targets: list[Path] = []
    for annotation_path in sorted(ROOT.glob(MULTIMODAL_ASSET_GLOB)):
        annotation = load_json(annotation_path)
        if annotation.get("pair_type") != "visual_asset":
            continue
        if skip_existing_llm and isinstance(annotation.get("llm_enrichment"), dict):
            continue
        image_path = ROOT / str(annotation.get("image_path", ""))
        if not image_path.is_file():
            continue
        targets.append(annotation_path)
        if limit is not None and len(targets) >= limit:
            break
    return targets


def count_missing_llm_enrichment(annotation_paths: Iterable[Path]) -> int:
    missing = 0
    for annotation_path in annotation_paths:
        annotation = load_json(annotation_path)
        if not isinstance(annotation.get("llm_enrichment"), dict):
            missing += 1
    return missing


def count_exportable_llm_assets(
    min_quality: str = "medium",
    allow_review_required: bool = True,
) -> int:
    exportable = 0
    for annotation_path in sorted(ROOT.glob(MULTIMODAL_ASSET_GLOB)):
        annotation = load_json(annotation_path)
        should_export, _reason = should_export_annotation(
            annotation=annotation,
            min_quality=min_quality,
            require_llm=True,
            allow_review_required=allow_review_required,
        )
        if not should_export:
            continue
        image_path = ROOT / str(annotation.get("image_path", ""))
        if image_path.is_file():
            exportable += 1
    return exportable


def print_header(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def print_check(result: CheckResult) -> None:
    prefixes = {
        "ok": "[OK]",
        "warn": "[WARN]",
        "error": "[ERR]",
        "info": "[INFO]",
    }
    prefix = prefixes.get(result.status, "[INFO]")
    print(f"{prefix} {result.label}: {result.detail}")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    raw = input(question + suffix).strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "j", "ja"}


def prompt_optional_limit() -> int | None:
    raw = input("Wie viele Assets sollen angereichert werden? Enter = alle: ").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print("Ungueltige Zahl. Es werden alle passenden Assets genommen.")
        return None
    return value if value > 0 else None


def ensure_runtime_dirs(config_path: str | Path | None = None) -> list[Path]:
    config = load_config(config_path)
    created: list[Path] = []
    for path in (
        config.user_data_dir,
        config.cookies_file.parent if config.cookies_file else None,
        PROCESSED_CHATGPT_RUNS_DIR,
    ):
        if path is None:
            continue
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def _config_value_points_into_chatgpt_dir(config_dir: Path, raw_value: object) -> bool:
    if raw_value in (None, ""):
        return False
    candidate = Path(os.path.expandvars(str(raw_value)))
    resolved = candidate.resolve() if candidate.is_absolute() else (config_dir / candidate).resolve()
    try:
        resolved.relative_to(CHATGPT_DIR.resolve())
        return True
    except ValueError:
        return False


def sanitize_chatgpt_config_paths(config_path: str | Path | None = None) -> tuple[bool, Path]:
    resolved_config_path = Path(config_path).resolve() if config_path else CHATGPT_CONFIG_PATH.resolve()
    if not resolved_config_path.exists():
        return False, resolved_config_path

    payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unerwartetes Format in {repo_relative(resolved_config_path)}")

    config_dir = resolved_config_path.parent
    changed = False
    for field_name, safe_value in SAFE_CONFIG_OVERRIDES.items():
        if _config_value_points_into_chatgpt_dir(config_dir, payload.get(field_name)):
            payload[field_name] = safe_value
            changed = True

    if changed:
        resolved_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed, resolved_config_path


def relocate_repo_session_artifacts(config_path: str | Path | None = None) -> tuple[list[tuple[Path, Path]], list[tuple[Path, str]]]:
    config = load_config(config_path)
    archive_root = CHATGPT_RUNTIME_DIR / "repo_session_artifacts" / datetime.now().strftime("%Y%m%d_%H%M%S")
    destinations = (
        (CHATGPT_DIR / "browser_profile", config.user_data_dir),
        (CHATGPT_DIR / "cookies" / "ChatGPT.json", config.cookies_file),
    )

    moved: list[tuple[Path, Path]] = []
    failures: list[tuple[Path, str]] = []
    for source, active_target in destinations:
        if source is None or not source.exists():
            continue

        destination = active_target if active_target and not active_target.exists() else archive_root / source.relative_to(CHATGPT_DIR)
        if destination is None:
            continue

        destination = Path(destination)
        if source.resolve() == destination.resolve():
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
        except Exception as exc:
            failures.append((source, str(exc)))

    for parent in (CHATGPT_DIR / "cookies",):
        if parent.exists() and parent.is_dir():
            try:
                next(parent.iterdir())
            except StopIteration:
                parent.rmdir()
            except OSError:
                pass

    return moved, failures


def build_doctor_results(config_path: str | Path | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    python_status = "ok" if sys.version_info >= (3, 10) else "warn"
    results.append(CheckResult("Python", python_status, python_version))

    pdf_count = count_real_files(RAW_PDFS_DIR, "*.pdf")
    results.append(
        CheckResult(
            "PDF input",
            "ok" if pdf_count else "error",
            f"{pdf_count} PDF-Datei(en) unter {repo_relative(RAW_PDFS_DIR)}",
        )
    )

    pymupdf_available = import_pymupdf() is not None
    results.append(
        CheckResult(
            "PyMuPDF",
            "ok" if pymupdf_available else "error",
            "installiert" if pymupdf_available else "fehlt. `pip install -r requirements.txt` ausfuehren.",
        )
    )

    ocr_runtime = detect_ocr_runtime()
    ocr_status = "ok" if ocr_runtime.get("available") else "warn"
    results.append(CheckResult("OCR / Tesseract", ocr_status, json.dumps(ocr_runtime, ensure_ascii=False)))

    asset_count = count_real_files(ROOT, MULTIMODAL_ASSET_GLOB)
    results.append(
        CheckResult(
            "Asset-Annotationen",
            "ok" if asset_count else "warn",
            f"{asset_count} Asset-Paare im zentralen pairs-Ordner gefunden",
        )
    )

    training_pair_count = count_real_files(ROOT, MULTIMODAL_TRAINING_PAIR_GLOB)
    results.append(
        CheckResult(
            "Training-Paare",
            "ok" if training_pair_count else "info",
            (
                f"{training_pair_count} kompakte Training-Paare gefunden"
                if training_pair_count
                else "Noch kein kompakter Trainingsexport vorhanden."
            ),
        )
    )

    curated_status = "ok" if CURATED_FILE.exists() else "warn"
    curated_detail = repo_relative(CURATED_FILE) if CURATED_FILE.exists() else "keine kuratierte Trainingsdatei vorhanden"
    results.append(CheckResult("Train/Eval-Split Quelle", curated_status, curated_detail))

    config_exists = CHATGPT_CONFIG_PATH.exists()
    results.append(
        CheckResult(
            "ChatGPT-Konfig",
            "ok" if config_exists else "warn",
            repo_relative(CHATGPT_CONFIG_PATH) if config_exists else "fehlt. Wird bei Bedarf automatisch aus der Vorlage erzeugt.",
        )
    )

    selector_count = count_real_files(CHATGPT_DIR, "**/selectors.json")
    results.append(
        CheckResult(
            "ChatGPT-Selektoren",
            "ok" if selector_count else "error",
            f"{selector_count} Selektor-Datei(en) gefunden",
        )
    )

    try:
        config = load_config(config_path)
    except Exception as exc:
        results.append(CheckResult("ChatGPT-Config lesen", "error", str(exc)))
        return results

    if config.user_data_dir:
        results.append(CheckResult("ChatGPT Profilordner", "ok", repo_relative(config.user_data_dir)))
    if config.cookies_file:
        cookie_detail = repo_relative(config.cookies_file)
        cookie_status = "ok" if config.cookies_file.exists() else "warn"
        if not config.cookies_file.exists():
            cookie_detail += " (noch nicht vorhanden, manueller Login ist normal)"
        results.append(CheckResult("ChatGPT Cookie-Datei", cookie_status, cookie_detail))

    repo_sensitive_paths: list[str] = []
    for sensitive_path in (
        CHATGPT_DIR / "browser_profile",
        CHATGPT_DIR / "cookies" / "ChatGPT.json",
    ):
        if sensitive_path.exists():
            repo_sensitive_paths.append(repo_relative(sensitive_path))
    if repo_sensitive_paths:
        results.append(
            CheckResult(
                "Repo-interne Session-Artefakte",
                "warn",
                ", ".join(repo_sensitive_paths) + " existieren noch. `setup-chatgpt` verschiebt sie sicher nach `.runtime`.",
            )
        )

    try:
        require_selenium()
        selenium_status = "ok"
        selenium_detail = "Selenium + selenium-stealth installiert"
    except RuntimeError as exc:
        selenium_status = "warn"
        selenium_detail = str(exc)
    results.append(CheckResult("Selenium", selenium_status, selenium_detail))

    browser_path = discover_browser_executable(config.browser)
    browser_status = "ok" if browser_path else "warn"
    browser_detail = str(browser_path) if browser_path else f"Kein {config.browser}-Browser automatisch gefunden."
    results.append(CheckResult("Browser", browser_status, browser_detail))

    return results


def print_next_steps(results: Iterable[CheckResult]) -> None:
    errors = [result for result in results if result.status == "error"]
    warnings = [result for result in results if result.status == "warn"]
    if not errors and not warnings:
        print("\nAlles Wesentliche ist bereit.")
        return

    print("\nNaechste sinnvolle Schritte:")
    if any(result.label == "PDF input" and result.status == "error" for result in errors):
        print(f"- Lege mindestens eine PDF unter {repo_relative(RAW_PDFS_DIR)} ab.")
    if any(result.label == "PyMuPDF" and result.status == "error" for result in errors):
        print("- Installiere die Python-Abhaengigkeiten mit `pip install -r requirements.txt`.")
    if any(result.label == "ChatGPT-Konfig" and result.status == "warn" for result in warnings):
        print(f"- Starte `{launcher_name()}` und waehle `ChatGPT vorbereiten`.")
    if any(result.label == "Selenium" and result.status == "warn" for result in warnings):
        print("- Fuer ChatGPT-Enrichment werden Selenium, selenium-stealth und ein lokaler Browser benoetigt.")


def run_python_script(script_name: str, args: list[str] | None = None) -> None:
    command = [sys.executable, str(SCRIPTS_DIR / script_name)]
    if args:
        command.extend(args)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"{script_name} ist mit Exit-Code {completed.returncode} fehlgeschlagen.")


def run_prepare_dataset() -> None:
    pdf_count = count_real_files(RAW_PDFS_DIR, "*.pdf")
    if not pdf_count:
        raise RuntimeError(f"Keine PDFs unter {repo_relative(RAW_PDFS_DIR)} gefunden.")
    print_header("Dataset wird gebaut")
    run_python_script("prepare_dataset.py")


def run_training_split() -> None:
    if not CURATED_FILE.exists():
        raise RuntimeError(f"Keine kuratierte Datei gefunden: {repo_relative(CURATED_FILE)}")
    print_header("Train/Eval-Split wird gebaut")
    run_python_script("build_training_split.py")


def run_export_training_pairs(
    min_quality: str = "medium",
    require_llm: bool = True,
    allow_review_required: bool = False,
) -> dict[str, object]:
    if not count_real_files(ROOT, MULTIMODAL_ASSET_GLOB):
        raise RuntimeError("Es gibt noch keine Asset-Annotationen. Bitte zuerst das Dataset bauen.")
    print_header("Kompakte Training-Paare werden exportiert")
    args = ["--min-quality", min_quality]
    if not require_llm:
        args.append("--no-require-llm")
    if allow_review_required:
        args.append("--allow-review-required")
    run_python_script("export_multimodal_training_pairs.py", args)
    if not MULTIMODAL_TRAINING_PAIR_MANIFEST_PATH.exists():
        raise RuntimeError(
            "Der Export ist durchgelaufen, aber das Manifest fehlt: "
            f"{repo_relative(MULTIMODAL_TRAINING_PAIR_MANIFEST_PATH)}"
        )
    manifest = load_json(MULTIMODAL_TRAINING_PAIR_MANIFEST_PATH)
    exported_pairs = int(manifest.get("exported_pairs", 0) or 0)
    print(f"- Exportierte Paare: {exported_pairs}")
    skipped = manifest.get("skipped")
    if isinstance(skipped, dict) and skipped:
        skip_summary = ", ".join(f"{key}={value}" for key, value in sorted(skipped.items()))
        print(f"- Uebersprungen: {skip_summary}")
    return manifest


def run_auto_export_training_pairs(with_chatgpt: bool) -> dict[str, object]:
    manifest = run_export_training_pairs(
        min_quality="medium",
        require_llm=with_chatgpt,
        allow_review_required=True,
    )
    if int(manifest.get("exported_pairs", 0) or 0) > 0:
        return manifest

    skipped = manifest.get("skipped")
    review_required_skips = 0
    missing_llm_skips = 0
    if isinstance(skipped, dict):
        review_required_skips = int(skipped.get("review_required", 0) or 0)
        missing_llm_skips = int(skipped.get("missing_llm_enrichment", 0) or 0)

    if missing_llm_skips > 0:
        print(
            "- Der strikte Export war unvollstaendig, weil noch Assets ohne LLM-Enrichment vorhanden sind. "
            "Auto-Modus startet deshalb keinen Review-Fallback."
        )
        return manifest

    if review_required_skips <= 0:
        print(
            "- Der strikte Export hat keine Trainingspaare erzeugt. "
            f"Details stehen in {repo_relative(MULTIMODAL_TRAINING_PAIR_MANIFEST_PATH)}."
        )
        return manifest

    print(
        "- Der strikte Export war leer, weil nur review-pflichtige Assets uebrig waren. "
        "Auto-Modus versucht jetzt einen zweiten Lauf mit `allow_review_required`."
    )
    return run_export_training_pairs(
        min_quality="medium",
        require_llm=with_chatgpt,
        allow_review_required=True,
    )


def run_enrichment(
    limit: int | None,
    language: str,
    max_assets_per_chat: int,
    dry_run: bool,
    manual_login: bool,
    config_path: str | Path | None,
    reprocess_existing: bool,
    keep_browser_open: bool | None,
) -> None:
    asset_count = count_real_files(ROOT, MULTIMODAL_ASSET_GLOB)
    if not asset_count:
        raise RuntimeError("Es gibt noch keine Asset-Annotationen. Bitte zuerst das Dataset bauen.")

    ensure_runtime_dirs(config_path)
    config_args: list[str] = []
    if config_path is not None:
        config_args.extend(["--config", str(Path(config_path).resolve())])

    args = [*config_args, "--language", language, "--max-assets-per-chat", str(max_assets_per_chat)]
    if limit is not None:
        args.extend(["--limit", str(limit)])
    if dry_run:
        args.append("--dry-run")
    if not manual_login:
        args.append("--no-manual-login")
    if reprocess_existing:
        args.append("--no-skip-existing-llm")
    if keep_browser_open is True:
        args.append("--keep-browser-open")
    elif keep_browser_open is False:
        args.append("--no-keep-browser-open")

    print_header("ChatGPT-Enrichment wird gestartet")
    run_python_script("enrich_multimodal_descriptions.py", args)


def setup_chatgpt_config(force: bool = False) -> None:
    created, target_path = copy_file_if_missing(CHATGPT_CONFIG_EXAMPLE_PATH, CHATGPT_CONFIG_PATH, force=force)
    if target_path is None:
        raise RuntimeError(f"Vorlage fehlt: {repo_relative(CHATGPT_CONFIG_EXAMPLE_PATH)}")
    sanitized, sanitized_path = sanitize_chatgpt_config_paths(target_path)
    ensure_runtime_dirs(sanitized_path)
    moved_artifacts, failed_artifacts = relocate_repo_session_artifacts(sanitized_path)

    print_header("ChatGPT-Setup")
    if created and force:
        print(f"Konfiguration neu geschrieben: {repo_relative(target_path)}")
    elif created:
        print(f"Konfiguration angelegt: {repo_relative(target_path)}")
    else:
        print(f"Konfiguration bereits vorhanden: {repo_relative(target_path)}")
    if sanitized:
        print(f"- Unsichere Session-Pfade in der Konfig wurden auf `.runtime` umgestellt: {repo_relative(sanitized_path)}")
    if moved_artifacts:
        print("- Repo-lokale Session-Artefakte wurden verschoben:")
        for source, destination in moved_artifacts:
            print(f"  - {repo_relative(source)} -> {repo_relative(destination)}")
    if failed_artifacts:
        print("- Konnte einzelne Legacy-Artefakte nicht verschieben:")
        for source, error in failed_artifacts:
            print(f"  - {repo_relative(source)}: {error}")

    config = load_config(sanitized_path)
    print(f"- Profilordner: {config.user_data_dir}")
    print(f"- Cookie-Datei: {config.cookies_file}")
    print(f"- Selektoren: {len(config.selector_files)} Datei(en)")
    print("- Wenn noch keine Session vorhanden ist, oeffnet sich beim ersten Enrichment ein normales Browserfenster zum Login.")
    print("- Nach erfolgreichem Login dieses Fenster wieder schliessen, damit die Automation das Profil weiterverwenden kann.")


def run_everything(
    with_chatgpt: bool,
    chatgpt_limit: int | None,
    chatgpt_language: str,
    chatgpt_max_assets_per_chat: int,
    chatgpt_dry_run: bool,
    manual_login: bool,
    config_path: str | Path | None,
    reprocess_existing: bool,
    keep_browser_open: bool | None,
    with_split: bool,
) -> None:
    print_header("Projektcheck")
    results = build_doctor_results(config_path)
    for result in results:
        print_check(result)
    print_next_steps(results)

    run_prepare_dataset()

    enrichment_target_paths: list[Path] = []
    if with_chatgpt:
        setup_chatgpt_config(force=False)
        enrichment_target_paths = collect_enrichment_target_paths(
            skip_existing_llm=not reprocess_existing,
            limit=chatgpt_limit,
        )
        run_enrichment(
            limit=chatgpt_limit,
            language=chatgpt_language,
            max_assets_per_chat=chatgpt_max_assets_per_chat,
            dry_run=chatgpt_dry_run,
            manual_login=manual_login,
            config_path=config_path,
            reprocess_existing=reprocess_existing,
            keep_browser_open=keep_browser_open,
        )
        if not chatgpt_dry_run and enrichment_target_paths:
            missing_llm = count_missing_llm_enrichment(enrichment_target_paths)
            if missing_llm:
                completed_llm = len(enrichment_target_paths) - missing_llm
                exportable_llm = count_exportable_llm_assets(
                    min_quality="medium",
                    allow_review_required=False,
                )
                if exportable_llm <= 0:
                    raise RuntimeError(
                        "ChatGPT-Enrichment blieb unvollstaendig "
                        f"({completed_llm}/{len(enrichment_target_paths)} Assets mit LLM-Daten), "
                        "und es gibt keine exportierbaren LLM-Assets. "
                        "Auto-Export wird abgebrochen, damit ein vorheriger Trainingsexport erhalten bleibt. "
                        f"Details stehen unter {repo_relative(PROCESSED_CHATGPT_RUNS_DIR)}."
                    )
                print(
                    "ChatGPT-Enrichment blieb unvollstaendig "
                    f"({completed_llm}/{len(enrichment_target_paths)} Assets mit LLM-Daten). "
                    f"{exportable_llm} exportierbare LLM-Assets werden jetzt trotzdem exportiert. "
                    "Beim naechsten Lauf werden fehlende Assets automatisch weiterverarbeitet, "
                    "solange bestehende LLM-Daten nicht explizit neu verarbeitet werden. "
                    f"Details stehen unter {repo_relative(PROCESSED_CHATGPT_RUNS_DIR)}.",
                    flush=True,
                )

    run_auto_export_training_pairs(with_chatgpt=with_chatgpt)

    if with_split:
        if CURATED_FILE.exists():
            run_training_split()
        else:
            print("\nTrain/Eval-Split wird uebersprungen, weil keine kuratierte Trainingsdatei vorhanden ist.")


def interactive_menu() -> None:
    while True:
        print_header("Einfacher Start")
        print("1. Projektcheck")
        print("2. ChatGPT vorbereiten")
        print("3. Dataset aus PDFs bauen")
        print("4. Asset-Beschreibungen mit ChatGPT anreichern")
        print("5. Kompakte Training-Paare exportieren")
        print("6. Train/Eval-Split bauen")
        print("7. Alles automatisch")
        print("0. Beenden")
        choice = input("\nBitte Zahl eingeben: ").strip()

        try:
            if choice == "1":
                results = build_doctor_results()
                for result in results:
                    print_check(result)
                print_next_steps(results)
            elif choice == "2":
                reset_config = prompt_yes_no("Soll die ChatGPT-Konfig notfalls frisch aus der Vorlage erzeugt werden?", False)
                setup_chatgpt_config(force=reset_config)
            elif choice == "3":
                run_prepare_dataset()
            elif choice == "4":
                if not count_real_files(ROOT, MULTIMODAL_ASSET_GLOB):
                    build_first = prompt_yes_no("Es gibt noch keine Asset-JSONs. Soll ich zuerst das Dataset bauen?", True)
                    if build_first:
                        run_prepare_dataset()
                setup_chatgpt_config(force=False)
                limit = prompt_optional_limit()
                language = input("Sprache fuer die Beschreibungen [de/en], Enter = de: ").strip().lower() or "de"
                if language not in {"de", "en"}:
                    language = "de"
                dry_run = prompt_yes_no("Nur testen ohne die JSON-Dateien zu veraendern?", False)
                manual_login = prompt_yes_no("Darf bei Bedarf ein Browser fuer manuellen Login aufgehen?", True)
                reprocess_existing = prompt_yes_no("Sollen bereits angereicherte Assets neu verarbeitet werden?", False)
                keep_browser_open = prompt_yes_no("Soll das Automations-Browserfenster nach dem Lauf offen bleiben?", True)
                run_enrichment(
                    limit=limit,
                    language=language,
                    max_assets_per_chat=DEFAULT_MAX_ASSETS_PER_CHAT,
                    dry_run=dry_run,
                    manual_login=manual_login,
                    config_path=None,
                    reprocess_existing=reprocess_existing,
                    keep_browser_open=keep_browser_open,
                )
            elif choice == "5":
                run_export_training_pairs()
            elif choice == "6":
                run_training_split()
            elif choice == "7":
                with_chatgpt = prompt_yes_no("Soll nach dem Dataset-Bau auch direkt ChatGPT-Enrichment laufen?", True)
                chatgpt_limit = None
                chatgpt_language = "de"
                chatgpt_max_assets_per_chat = DEFAULT_MAX_ASSETS_PER_CHAT
                chatgpt_dry_run = False
                manual_login = True
                reprocess_existing = False
                keep_browser_open = None
                if with_chatgpt:
                    setup_chatgpt_config(force=False)
                    chatgpt_limit = prompt_optional_limit()
                    chatgpt_language = input("Sprache fuer ChatGPT [de/en], Enter = de: ").strip().lower() or "de"
                    if chatgpt_language not in {"de", "en"}:
                        chatgpt_language = "de"
                    chatgpt_dry_run = prompt_yes_no("Nur testen ohne Dateien zu aendern?", False)
                    manual_login = prompt_yes_no("Darf bei Bedarf ein Browser fuer manuellen Login aufgehen?", True)
                    reprocess_existing = prompt_yes_no("Sollen bereits angereicherte Assets neu verarbeitet werden?", False)
                    keep_browser_open = prompt_yes_no(
                        "Soll das Automations-Browserfenster nach dem Lauf offen bleiben?",
                        True,
                    )
                with_split = prompt_yes_no("Soll ich am Ende auch den Train/Eval-Split bauen, falls kuratierte Daten existieren?", False)
                run_everything(
                    with_chatgpt=with_chatgpt,
                    chatgpt_limit=chatgpt_limit,
                    chatgpt_language=chatgpt_language,
                    chatgpt_max_assets_per_chat=chatgpt_max_assets_per_chat,
                    chatgpt_dry_run=chatgpt_dry_run,
                    manual_login=manual_login,
                    config_path=None,
                    reprocess_existing=reprocess_existing,
                    keep_browser_open=keep_browser_open,
                    with_split=with_split,
                )
            elif choice == "0":
                return
            else:
                print("Bitte eine gueltige Zahl waehlen.")
        except Exception as exc:
            print(f"\nFehler: {exc}")

        input("\nWeiter mit Enter...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Einfacher, gefuehrter Einstieg fuer Dataset-Bau, ChatGPT-Enrichment und Train/Eval-Split."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("doctor", help="Projektcheck ausgeben.")

    setup_parser = subparsers.add_parser("setup-chatgpt", help="ChatGPT-Konfig aus der Vorlage anlegen.")
    setup_parser.add_argument("--reset-config", action="store_true", help="Vorhandene ChatGPT-Konfig aus der Vorlage neu schreiben.")

    bootstrap_login_parser = subparsers.add_parser(
        "bootstrap-chatgpt-login",
        help="ChatGPT-Login im Projektprofil oeffnen und Session-Cookies pruefen.",
    )
    bootstrap_login_parser.add_argument("--config", default=None, help="Optionaler Pfad zu ChatGPT/config.json.")

    set_cookie_parser = subparsers.add_parser("set-cookie", help="Ein einzelnes ChatGPT-Cookie in die Cookie-Datei schreiben.")
    set_cookie_parser.add_argument("--name", required=True, help="Cookie-Name.")
    set_cookie_parser.add_argument("--value", required=True, help="Cookie-Wert.")
    set_cookie_parser.add_argument("--domain", default=".chatgpt.com", help="Cookie-Domain. Standard: .chatgpt.com")
    set_cookie_parser.add_argument("--path", default="/", help="Cookie-Pfad. Standard: /")
    set_cookie_parser.add_argument("--expiry", type=int, default=None, help="Optionaler Unix-Timestamp fuer den Ablaufzeitpunkt.")
    set_cookie_parser.add_argument("--same-site", choices=["Strict", "Lax", "None"], default=None, help="Optionaler SameSite-Wert.")
    set_cookie_parser.add_argument("--secure", action=argparse.BooleanOptionalAction, default=True, help="Secure-Flag setzen.")
    set_cookie_parser.add_argument("--http-only", action=argparse.BooleanOptionalAction, default=None, help="HttpOnly-Flag setzen.")
    set_cookie_parser.add_argument("--replace", action="store_true", help="Bestehende Cookie-Datei ersetzen statt Cookies zusammenzufuehren.")
    set_cookie_parser.add_argument("--config", default=None, help="Optionaler Pfad zu ChatGPT/config.json.")

    import_cookies_parser = subparsers.add_parser("import-cookies", help="ChatGPT-Cookies aus einer JSON-Datei importieren.")
    import_cookies_parser.add_argument("--source", required=True, help="JSON-Datei mit {'cookies': [...]} oder direkter Cookie-Liste.")
    import_cookies_parser.add_argument("--replace", action="store_true", help="Bestehende Cookie-Datei ersetzen statt Cookies zusammenzufuehren.")
    import_cookies_parser.add_argument("--config", default=None, help="Optionaler Pfad zu ChatGPT/config.json.")

    subparsers.add_parser("prepare", help="Dataset aus den Rohdaten bauen.")

    enrich_parser = subparsers.add_parser("enrich", help="Asset-Beschreibungen mit ChatGPT anreichern.")
    enrich_parser.add_argument("--limit", type=int, default=None, help="Maximale Anzahl an Assets. Standard: alle.")
    enrich_parser.add_argument("--language", choices=["de", "en"], default="de", help="Zielsprache fuer die Beschreibungen.")
    enrich_parser.add_argument(
        "--max-assets-per-chat",
        type=int,
        default=DEFAULT_MAX_ASSETS_PER_CHAT,
        help="Wie viele Assets pro ChatGPT-Chat verarbeitet werden, bevor ein neuer Chat startet. 0 = ein Chat fuer den ganzen Lauf.",
    )
    enrich_parser.add_argument("--dry-run", action="store_true", help="Browserlauf ohne Zurueckschreiben der JSON-Dateien.")
    enrich_parser.add_argument(
        "--manual-login",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Erlaubt einen manuellen Browser-Login, wenn keine Session vorhanden ist.",
    )
    enrich_parser.add_argument(
        "--keep-browser-open",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Laesst das Automations-Browserfenster nach dem Lauf offen.",
    )
    enrich_parser.add_argument("--config", default=None, help="Optionaler Pfad zu ChatGPT/config.json.")
    enrich_parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Verarbeitet auch bereits angereicherte Assets erneut.",
    )

    export_pairs_parser = subparsers.add_parser(
        "export-training-pairs",
        help="Kompakte Bild+JSON-Paare fuer multimodales Fine-Tuning exportieren.",
    )
    export_pairs_parser.add_argument(
        "--min-quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimale Qualitaet fuer den Export.",
    )
    export_pairs_parser.add_argument(
        "--require-llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Nur bereits per ChatGPT angereicherte Assets exportieren.",
    )
    export_pairs_parser.add_argument(
        "--allow-review-required",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auch Assets exportieren, die noch auf manueller Review stehen.",
    )

    subparsers.add_parser("split", help="Train/Eval-Split aus der kuratierten Datei bauen.")

    all_parser = subparsers.add_parser("all", help="Gefuehrten Komplettlauf ausfuehren.")
    all_parser.add_argument("--with-chatgpt", action="store_true", help="Nach dem Dataset-Bau auch ChatGPT-Enrichment ausfuehren.")
    all_parser.add_argument("--chatgpt-limit", type=int, default=None, help="Maximale Anzahl an Assets fuer das ChatGPT-Enrichment.")
    all_parser.add_argument("--chatgpt-language", choices=["de", "en"], default="de", help="Zielsprache fuer das ChatGPT-Enrichment.")
    all_parser.add_argument(
        "--chatgpt-max-assets-per-chat",
        type=int,
        default=DEFAULT_MAX_ASSETS_PER_CHAT,
        help="Wie viele Assets pro ChatGPT-Chat verarbeitet werden, bevor ein neuer Chat startet. 0 = ein Chat fuer den ganzen Lauf.",
    )
    all_parser.add_argument("--chatgpt-dry-run", action="store_true", help="ChatGPT-Enrichment ohne Datei-Updates.")
    all_parser.add_argument(
        "--manual-login",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Erlaubt einen manuellen Browser-Login fuer ChatGPT.",
    )
    all_parser.add_argument(
        "--keep-browser-open",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Laesst das Automations-Browserfenster nach dem Lauf offen.",
    )
    all_parser.add_argument("--config", default=None, help="Optionaler Pfad zu ChatGPT/config.json.")
    all_parser.add_argument(
        "--reprocess-existing",
        action="store_true",
        help="Verarbeitet auch bereits angereicherte Assets erneut.",
    )
    all_parser.add_argument("--with-split", action="store_true", help="Am Ende auch den Train/Eval-Split bauen.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.command:
        interactive_menu()
        return

    if args.command == "doctor":
        results = build_doctor_results()
        for result in results:
            print_check(result)
        print_next_steps(results)
        return

    if args.command == "setup-chatgpt":
        setup_chatgpt_config(force=bool(args.reset_config))
        return

    if args.command == "bootstrap-chatgpt-login":
        bootstrap_chatgpt_login(config_path=args.config)
        return

    if args.command == "set-cookie":
        cookie = build_cookie_from_args(args)
        written = write_chatgpt_cookies([cookie], replace=bool(args.replace), config_path=args.config)
        print(f"{written} Cookie(s) in die ChatGPT-Cookie-Datei geschrieben.")
        return

    if args.command == "import-cookies":
        cookies = load_cookie_source(Path(args.source).resolve())
        if not cookies:
            raise RuntimeError(f"Keine gueltigen Cookies in {args.source} gefunden.")
        written = write_chatgpt_cookies(cookies, replace=bool(args.replace), config_path=args.config)
        print(f"{written} Cookie(s) in die ChatGPT-Cookie-Datei geschrieben.")
        return

    if args.command == "prepare":
        run_prepare_dataset()
        return

    if args.command == "enrich":
        setup_chatgpt_config(force=False)
        run_enrichment(
            limit=args.limit,
            language=args.language,
            max_assets_per_chat=args.max_assets_per_chat,
            dry_run=args.dry_run,
            manual_login=args.manual_login,
            config_path=args.config,
            reprocess_existing=args.reprocess_existing,
            keep_browser_open=args.keep_browser_open,
        )
        return

    if args.command == "export-training-pairs":
        run_export_training_pairs(
            min_quality=args.min_quality,
            require_llm=args.require_llm,
            allow_review_required=args.allow_review_required,
        )
        return

    if args.command == "split":
        run_training_split()
        return

    if args.command == "all":
        run_everything(
            with_chatgpt=args.with_chatgpt,
            chatgpt_limit=args.chatgpt_limit,
            chatgpt_language=args.chatgpt_language,
            chatgpt_max_assets_per_chat=args.chatgpt_max_assets_per_chat,
            chatgpt_dry_run=args.chatgpt_dry_run,
            manual_login=args.manual_login,
            config_path=args.config,
            reprocess_existing=args.reprocess_existing,
            keep_browser_open=args.keep_browser_open,
            with_split=args.with_split,
        )
        return

    raise RuntimeError(f"Unbekannter Befehl: {args.command}")


if __name__ == "__main__":
    main()
