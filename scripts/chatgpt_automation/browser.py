from __future__ import annotations

import json
import platform
import shutil
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any

from .config import ChatGPTAutomationConfig

SELENIUM_IMPORT_ERROR: Exception | None = None
SELENIUM_STEALTH_IMPORT_ERROR: Exception | None = None
try:
    from selenium import webdriver
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.edge.service import Service as EdgeService
except ImportError as exc:  # pragma: no cover - depends on local machine state
    SELENIUM_IMPORT_ERROR = exc
    webdriver = None  # type: ignore[assignment]
    WebDriverException = Exception  # type: ignore[assignment]
    ChromeOptions = object  # type: ignore[assignment]
    ChromeService = object  # type: ignore[assignment]
    EdgeOptions = object  # type: ignore[assignment]
    EdgeService = object  # type: ignore[assignment]

try:
    from selenium_stealth import stealth
except ImportError as exc:  # pragma: no cover - depends on local machine state
    SELENIUM_STEALTH_IMPORT_ERROR = exc
    stealth = None  # type: ignore[assignment]


def _missing_browser_dependencies() -> list[str]:
    missing: list[str] = []
    if SELENIUM_IMPORT_ERROR is not None:
        missing.append("selenium")
    if SELENIUM_STEALTH_IMPORT_ERROR is not None:
        missing.append("selenium-stealth")
    return missing


def require_selenium() -> None:
    missing_dependencies = _missing_browser_dependencies()
    if missing_dependencies:
        missing_text = ", ".join(missing_dependencies)
        raise RuntimeError(
            f"Browser automation dependencies are missing ({missing_text}). Install dependencies with "
            "`pip install -r requirements.txt` before running the ChatGPT browser automation."
        ) from (SELENIUM_IMPORT_ERROR or SELENIUM_STEALTH_IMPORT_ERROR)


def discover_browser_executable(browser: str) -> Path | None:
    candidates: list[Path] = []
    system = platform.system().lower()
    if browser == "edge" and system == "windows":
        candidates.extend(
            [
                Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
                Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            ]
        )
    elif browser != "edge" and system == "windows":
        candidates.extend(
            [
                Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
                Path(r"C:\Program Files\Chromium\Application\chrome.exe"),
            ]
        )
    elif browser == "edge" and system == "darwin":
        candidates.append(Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"))
    elif browser != "edge" and system == "darwin":
        candidates.extend(
            [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    path_commands = (
        ("msedge.exe", "msedge", "microsoft-edge", "microsoft-edge-stable")
        if browser == "edge"
        else (
            "chrome.exe",
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "chrome",
        )
    )
    for path_command in path_commands:
        resolved = shutil.which(path_command)
        if resolved:
            return Path(resolved)
    return None


def _build_common_options(options: Any, config: ChatGPTAutomationConfig, use_profile: bool) -> Any:
    options.add_argument(f"--window-size={config.window_size}")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-first-run")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-notifications")
    if config.headless:
        options.add_argument("--headless=new")
    if config.user_agent:
        options.add_argument(f"--user-agent={config.user_agent}")
    if use_profile and config.user_data_dir:
        options.add_argument(f"--user-data-dir={config.user_data_dir}")
    if config.keep_browser_open and hasattr(options, "add_experimental_option"):
        options.add_experimental_option("detach", True)
    if config.stealth:
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
    return options


def _default_stealth_platform() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "Win32"
    if system == "darwin":
        return "MacIntel"
    return "Linux x86_64"


def _build_stealth_kwargs(config: ChatGPTAutomationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "languages": ["de-DE", "de", "en-US", "en"],
        "vendor": "Google Inc.",
        "platform": _default_stealth_platform(),
        "webgl_vendor": "Intel Inc.",
        "renderer": "Intel Iris OpenGL Engine",
        "fix_hairline": True,
    }
    if config.user_agent:
        kwargs["user_agent"] = config.user_agent
    return kwargs


def _apply_selenium_stealth(driver: Any, config: ChatGPTAutomationConfig) -> None:
    if not config.stealth:
        return
    try:
        stealth(driver, **_build_stealth_kwargs(config))
    except Exception as exc:
        raise RuntimeError("Could not apply selenium-stealth to the browser session.") from exc


def create_webdriver(config: ChatGPTAutomationConfig, use_profile: bool = True) -> Any:
    require_selenium()
    browser = config.browser
    executable = config.browser_executable or discover_browser_executable(browser)

    if browser == "edge":
        options = _build_common_options(EdgeOptions(), config, use_profile)
        if executable:
            options.binary_location = str(executable)
        service = EdgeService(executable_path=str(config.driver_path)) if config.driver_path else EdgeService()
        driver = webdriver.Edge(options=options, service=service)
    else:
        options = _build_common_options(ChromeOptions(), config, use_profile)
        if executable:
            options.binary_location = str(executable)
        service = ChromeService(executable_path=str(config.driver_path)) if config.driver_path else ChromeService()
        driver = webdriver.Chrome(options=options, service=service)

    try:
        _apply_selenium_stealth(driver, config)
    except Exception:
        with suppress(Exception):
            driver.quit()
        raise

    driver.set_page_load_timeout(config.page_load_timeout)
    return driver


def build_manual_login_browser_command(config: ChatGPTAutomationConfig, url: str | None = None) -> list[str]:
    executable = config.browser_executable or discover_browser_executable(config.browser)
    if executable is None:
        raise RuntimeError(f"Could not find a local {config.browser} executable for the manual login bootstrap.")
    if config.user_data_dir is None:
        raise RuntimeError("Manual login bootstrap requires a user_data_dir in the ChatGPT config.")

    config.user_data_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(executable),
        f"--user-data-dir={config.user_data_dir.resolve()}",
        "--new-window",
        "--no-first-run",
    ]
    if config.window_size:
        command.append(f"--window-size={config.window_size}")

    target_url = (url or config.login_url or config.base_url).strip()
    if target_url:
        command.append(target_url)
    return command


def launch_browser_for_manual_login(config: ChatGPTAutomationConfig, url: str | None = None) -> subprocess.Popen[Any]:
    command = build_manual_login_browser_command(config, url=url)
    return subprocess.Popen(command)


def build_driver_with_fallback(config: ChatGPTAutomationConfig) -> Any:
    require_selenium()
    errors: list[str] = []

    if config.user_data_dir and config.user_data_dir.exists():
        try:
            return create_webdriver(config, use_profile=True)
        except (RuntimeError, WebDriverException) as exc:  # pragma: no cover - depends on local browser state
            errors.append(f"profile mode failed: {exc}")

    try:
        return create_webdriver(config, use_profile=False)
    except (RuntimeError, WebDriverException) as exc:  # pragma: no cover - depends on local browser state
        errors.append(f"clean mode failed: {exc}")
        raise RuntimeError("Could not start a Selenium browser session.\n" + "\n".join(errors)) from exc


def normalize_cookie_payload(payload: Any) -> list[dict[str, Any]]:
    cookies = payload.get("cookies", payload) if isinstance(payload, dict) else payload
    if not isinstance(cookies, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in cookies:
        if not isinstance(item, dict):
            continue
        cookie: dict[str, Any] = {
            "name": item.get("name"),
            "value": item.get("value"),
            "path": item.get("path", "/"),
        }
        domain = item.get("domain")
        if isinstance(domain, str) and domain.strip():
            cookie["domain"] = domain
        expiry = item.get("expiry", item.get("expirationDate"))
        if isinstance(expiry, (int, float)):
            cookie["expiry"] = int(expiry)
        same_site = item.get("sameSite", item.get("same_site"))
        if isinstance(same_site, str) and same_site.strip():
            cookie["sameSite"] = same_site
        secure = item.get("secure")
        if isinstance(secure, bool):
            cookie["secure"] = secure
        http_only = item.get("httpOnly", item.get("http_only"))
        if isinstance(http_only, bool):
            cookie["httpOnly"] = http_only
        if cookie.get("name") and cookie.get("value") is not None:
            normalized.append(cookie)
    return normalized


def load_cookie_payload(cookie_file: Path | None) -> list[dict[str, Any]]:
    if cookie_file is None or not cookie_file.exists():
        return []
    payload = json.loads(cookie_file.read_text(encoding="utf-8"))
    return normalize_cookie_payload(payload)


def _cookie_merge_key(cookie: dict[str, Any]) -> tuple[str, str, str]:
    domain = str(cookie.get("domain", "") or "").lstrip(".").lower()
    path = str(cookie.get("path", "/") or "/")
    name = str(cookie.get("name", "") or "")
    return domain, path, name


def write_cookie_payload(cookie_file: Path, cookies: list[dict[str, Any]]) -> int:
    normalized = normalize_cookie_payload(cookies)
    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    cookie_file.write_text(json.dumps({"cookies": normalized}, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(normalized)


def set_cookie_payload(cookie_file: Path | None, cookies: list[dict[str, Any]], replace: bool = False) -> int:
    if cookie_file is None:
        return 0

    new_cookies = normalize_cookie_payload(cookies)
    if replace:
        return write_cookie_payload(cookie_file, new_cookies)

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cookie in load_cookie_payload(cookie_file):
        merged[_cookie_merge_key(cookie)] = cookie
    for cookie in new_cookies:
        merged[_cookie_merge_key(cookie)] = cookie
    return write_cookie_payload(cookie_file, list(merged.values()))
