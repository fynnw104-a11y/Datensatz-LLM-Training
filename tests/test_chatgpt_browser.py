from __future__ import annotations

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
