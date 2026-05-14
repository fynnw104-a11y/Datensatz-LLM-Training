from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHATGPT_DIR = ROOT / "ChatGPT"
DEFAULT_CONFIG_PATH = CHATGPT_DIR / "config.json"
DEFAULT_BASE_URL = "https://chatgpt.com/"
DEFAULT_LOGIN_URL = "https://chatgpt.com/auth/login"
DEFAULT_RUNTIME_DIR = ROOT / ".runtime" / "chatgpt"
DEFAULT_USER_DATA_DIR = DEFAULT_RUNTIME_DIR / "browser_profile"
DEFAULT_COOKIES_FILE = DEFAULT_RUNTIME_DIR / "cookies" / "ChatGPT.json"


def _resolve_candidate(base_dir: Path, raw_value: str | None, fallback: Path | None = None) -> Path | None:
    if not raw_value:
        return fallback

    raw_path = Path(os.path.expanduser(os.path.expandvars(raw_value)))
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                (base_dir / raw_path).resolve(),
                (ROOT / raw_path).resolve(),
                (CHATGPT_DIR / raw_path).resolve(),
            ]
        )
        if raw_path.name:
            candidates.append((CHATGPT_DIR / raw_path.name).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else fallback


def _resolve_optional_path(base_dir: Path, raw_value: str | None, fallback: Path | None = None) -> Path | None:
    if raw_value and raw_value.strip():
        return _resolve_candidate(base_dir, raw_value, fallback=fallback)
    return fallback


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _prefer_safe_runtime_path(candidate: Path | None, fallback: Path | None) -> Path | None:
    if candidate is None:
        return fallback
    if fallback is None:
        return candidate
    if _path_is_within(candidate, CHATGPT_DIR):
        return fallback
    return candidate


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class ChatGPTAutomationConfig:
    config_path: Path
    chatgpt_dir: Path
    browser: str
    headless: bool
    stealth: bool
    keep_browser_open: bool
    window_size: str
    user_agent: str
    base_url: str
    login_url: str
    user_data_dir: Path | None
    cookies_file: Path | None
    active_selector_file: Path | None
    selector_files: tuple[Path, ...]
    browser_executable: Path | None
    driver_path: Path | None
    page_load_timeout: float
    send_button_timeout_seconds: float
    response_timeout_seconds: float
    response_idle_seconds: float
    prompt_settle_seconds: float
    manual_login_timeout_seconds: float

    def with_overrides(self, **overrides: object) -> "ChatGPTAutomationConfig":
        payload = self.__dict__.copy()
        payload.update(overrides)
        return ChatGPTAutomationConfig(**payload)


def load_config(config_path: str | Path | None = None) -> ChatGPTAutomationConfig:
    resolved_config_path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH.resolve()
    config_dir = resolved_config_path.parent
    payload: dict[str, object] = {}
    if resolved_config_path.exists():
        payload = json.loads(resolved_config_path.read_text(encoding="utf-8"))

    browser = str(os.getenv("CHATGPT_BROWSER", payload.get("browser", "chrome"))).strip().lower() or "chrome"
    headless = _bool_env("CHATGPT_HEADLESS", bool(payload.get("headless", False)))
    stealth = _bool_env("CHATGPT_STEALTH", bool(payload.get("stealth", True)))
    keep_browser_open = _bool_env("CHATGPT_KEEP_BROWSER_OPEN", bool(payload.get("keep_browser_open", False)))
    window_size = str(os.getenv("CHATGPT_WINDOW_SIZE", payload.get("window_size", "1400,1000"))).strip()
    user_agent = str(os.getenv("CHATGPT_USER_AGENT", payload.get("user_agent", ""))).strip()
    base_url = str(os.getenv("CHATGPT_BASE_URL", payload.get("url", DEFAULT_BASE_URL))).strip() or DEFAULT_BASE_URL
    login_url = str(os.getenv("CHATGPT_LOGIN_URL", payload.get("login_url", DEFAULT_LOGIN_URL))).strip()
    raw_user_data_dir = os.getenv("CHATGPT_USER_DATA_DIR") or str(payload.get("user_data_dir", "")).strip()
    raw_browser_executable = os.getenv("CHATGPT_BROWSER_EXECUTABLE") or str(
        payload.get("browser_executable", "")
    ).strip()
    raw_driver_path = os.getenv("CHATGPT_DRIVER_PATH") or str(payload.get("driver_path", "")).strip()
    raw_active_selector = str(payload.get("active_selector_file", "")).strip()
    raw_cookies_file = os.getenv("CHATGPT_COOKIES_FILE") or str(payload.get("cookies_file", "")).strip()

    user_data_dir = _prefer_safe_runtime_path(
        _resolve_optional_path(config_dir, raw_user_data_dir, fallback=DEFAULT_USER_DATA_DIR),
        DEFAULT_USER_DATA_DIR,
    )
    cookies_file = _prefer_safe_runtime_path(
        _resolve_optional_path(config_dir, raw_cookies_file, fallback=DEFAULT_COOKIES_FILE),
        DEFAULT_COOKIES_FILE,
    )
    active_selector_file = _resolve_optional_path(config_dir, raw_active_selector)
    browser_executable = _resolve_optional_path(config_dir, raw_browser_executable)
    driver_path = _resolve_optional_path(config_dir, raw_driver_path)

    selector_paths: list[Path] = []
    for selector_path in sorted(CHATGPT_DIR.rglob("selectors.json")):
        selector_paths.append(selector_path.resolve())
    if active_selector_file and active_selector_file not in selector_paths:
        selector_paths.insert(0, active_selector_file)

    deduped_selectors: list[Path] = []
    seen: set[Path] = set()
    for path in selector_paths:
        if path in seen:
            continue
        seen.add(path)
        deduped_selectors.append(path)

    return ChatGPTAutomationConfig(
        config_path=resolved_config_path,
        chatgpt_dir=CHATGPT_DIR,
        browser=browser,
        headless=headless,
        stealth=stealth,
        keep_browser_open=keep_browser_open,
        window_size=window_size,
        user_agent=user_agent,
        base_url=base_url,
        login_url=login_url,
        user_data_dir=user_data_dir,
        cookies_file=cookies_file,
        active_selector_file=active_selector_file,
        selector_files=tuple(deduped_selectors),
        browser_executable=browser_executable,
        driver_path=driver_path,
        page_load_timeout=_float_env("CHATGPT_PAGE_LOAD_TIMEOUT", 60.0),
        send_button_timeout_seconds=_float_env("CHATGPT_SEND_BUTTON_TIMEOUT", 45.0),
        response_timeout_seconds=_float_env("CHATGPT_RESPONSE_TIMEOUT", 180.0),
        response_idle_seconds=_float_env("CHATGPT_RESPONSE_IDLE_SECONDS", 6.0),
        prompt_settle_seconds=_float_env("CHATGPT_PROMPT_SETTLE_SECONDS", 1.0),
        manual_login_timeout_seconds=_float_env("CHATGPT_MANUAL_LOGIN_TIMEOUT", 300.0),
    )
