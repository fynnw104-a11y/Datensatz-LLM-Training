import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chatgpt_automation.config import (
    CHATGPT_DIR,
    DEFAULT_COOKIES_FILE,
    DEFAULT_USER_DATA_DIR,
    _path_is_within,
    _prefer_safe_runtime_path,
)


class ChatGPTConfigTests(unittest.TestCase):
    def test_path_is_within_detects_repo_subpath(self) -> None:
        candidate = CHATGPT_DIR / "browser_profile" / "Default"
        self.assertTrue(_path_is_within(candidate, CHATGPT_DIR))

    def test_prefer_safe_runtime_path_redirects_repo_local_profile(self) -> None:
        candidate = CHATGPT_DIR / "browser_profile"
        self.assertEqual(_prefer_safe_runtime_path(candidate, DEFAULT_USER_DATA_DIR), DEFAULT_USER_DATA_DIR)

    def test_prefer_safe_runtime_path_keeps_external_path(self) -> None:
        candidate = Path.home() / "safe-chatgpt-profile"
        self.assertEqual(_prefer_safe_runtime_path(candidate, DEFAULT_COOKIES_FILE), candidate)


if __name__ == "__main__":
    unittest.main()
