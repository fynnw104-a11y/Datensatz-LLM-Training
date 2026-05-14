from __future__ import annotations

import sys
import shutil
import unittest
import uuid
import json
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chatgpt_automation import browser
from chatgpt_automation.config import load_config


class ChatGPTBrowserTests(unittest.TestCase):
    def test_build_stealth_kwargs_keeps_configured_user_agent(self) -> None:
        config = load_config().with_overrides(user_agent="CustomAgent/1.0")

        kwargs = browser._build_stealth_kwargs(config)

        self.assertEqual(kwargs["user_agent"], "CustomAgent/1.0")
        self.assertEqual(kwargs["vendor"], "Google Inc.")
        self.assertIn("languages", kwargs)

    def test_require_selenium_mentions_selenium_stealth_when_missing(self) -> None:
        original_selenium_error = browser.SELENIUM_IMPORT_ERROR
        original_stealth_error = browser.SELENIUM_STEALTH_IMPORT_ERROR
        try:
            browser.SELENIUM_IMPORT_ERROR = ImportError("selenium missing")
            browser.SELENIUM_STEALTH_IMPORT_ERROR = ImportError("selenium-stealth missing")

            with self.assertRaises(RuntimeError) as context:
                browser.require_selenium()

            self.assertIn("selenium-stealth", str(context.exception))
            self.assertIn("pip install -r requirements.txt", str(context.exception))
        finally:
            browser.SELENIUM_IMPORT_ERROR = original_selenium_error
            browser.SELENIUM_STEALTH_IMPORT_ERROR = original_stealth_error

    def test_build_manual_login_browser_command_uses_profile_and_plain_browser_flags(self) -> None:
        profile_dir = ROOT / ".tmp" / "test_chatgpt_manual_login_profile"
        config = load_config().with_overrides(
            browser_executable=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            user_data_dir=profile_dir,
        )

        command = browser.build_manual_login_browser_command(config, url="https://chatgpt.com/auth/login")

        self.assertEqual(command[0], r"C:\Program Files\Google\Chrome\Application\chrome.exe")
        self.assertIn(f"--user-data-dir={profile_dir.resolve()}", command)
        self.assertIn("--new-window", command)
        self.assertNotIn("--disable-blink-features=AutomationControlled", command)
        self.assertEqual(command[-1], "https://chatgpt.com/auth/login")

    def test_build_manual_login_browser_command_requires_user_data_dir(self) -> None:
        config = load_config().with_overrides(
            browser_executable=Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            user_data_dir=None,
        )

        with self.assertRaises(RuntimeError) as context:
            browser.build_manual_login_browser_command(config)

        self.assertIn("user_data_dir", str(context.exception))

    def test_discover_browser_executable_finds_linux_chrome_from_path(self) -> None:
        def fake_which(command: str) -> str | None:
            return "/usr/bin/google-chrome" if command == "google-chrome" else None

        with (
            mock.patch.object(browser.platform, "system", return_value="Linux"),
            mock.patch.object(browser.shutil, "which", side_effect=fake_which),
        ):
            self.assertEqual(browser.discover_browser_executable("chrome"), Path("/usr/bin/google-chrome"))

    def test_discover_browser_executable_finds_linux_edge_from_path(self) -> None:
        def fake_which(command: str) -> str | None:
            return "/usr/bin/microsoft-edge" if command == "microsoft-edge" else None

        with (
            mock.patch.object(browser.platform, "system", return_value="Linux"),
            mock.patch.object(browser.shutil, "which", side_effect=fake_which),
        ):
            self.assertEqual(browser.discover_browser_executable("edge"), Path("/usr/bin/microsoft-edge"))

    def test_normalize_cookie_payload_accepts_wrapped_or_plain_cookie_lists(self) -> None:
        payload = {
            "cookies": [
                {
                    "name": "session",
                    "value": "abc",
                    "domain": ".chatgpt.com",
                    "sameSite": "Lax",
                    "expirationDate": 1912345678,
                    "secure": True,
                    "http_only": True,
                }
            ]
        }

        wrapped = browser.normalize_cookie_payload(payload)
        plain = browser.normalize_cookie_payload(payload["cookies"])

        self.assertEqual(wrapped, plain)
        self.assertEqual(wrapped[0]["name"], "session")
        self.assertEqual(wrapped[0]["path"], "/")
        self.assertEqual(wrapped[0]["expiry"], 1912345678)
        self.assertTrue(wrapped[0]["secure"])
        self.assertTrue(wrapped[0]["httpOnly"])

    def test_set_cookie_payload_merges_by_domain_path_and_name(self) -> None:
        temp_root = ROOT / ".tmp" / f"test_set_cookie_payload_{uuid.uuid4().hex}"
        cookie_file = temp_root / "cookies.json"
        try:
            temp_root.mkdir(parents=True, exist_ok=True)
            cookie_file.write_text(
                json.dumps(
                    {
                        "cookies": [
                            {"name": "session", "value": "old", "domain": ".chatgpt.com", "path": "/"},
                            {"name": "other", "value": "kept", "domain": ".chatgpt.com", "path": "/"},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            written = browser.set_cookie_payload(
                cookie_file,
                [{"name": "session", "value": "new", "domain": ".chatgpt.com", "path": "/"}],
            )

            self.assertEqual(written, 2)
            cookies_by_name = {cookie["name"]: cookie["value"] for cookie in browser.load_cookie_payload(cookie_file)}
            self.assertEqual(cookies_by_name, {"session": "new", "other": "kept"})
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
