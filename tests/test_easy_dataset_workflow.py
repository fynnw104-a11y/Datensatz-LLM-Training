import sys
import shutil
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from easy_dataset_workflow import copy_file_if_missing, count_real_files


class EasyWorkflowTests(unittest.TestCase):
    def test_count_real_files_ignores_gitkeep(self) -> None:
        temp_root = ROOT / ".tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        root = temp_root / f"test_count_real_files_{uuid.uuid4().hex}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            (root / ".gitkeep").write_text("", encoding="utf-8")
            (root / "one.pdf").write_text("x", encoding="utf-8")
            self.assertEqual(count_real_files(root, "*.pdf"), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_copy_file_if_missing_creates_target(self) -> None:
        temp_root = ROOT / ".tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        tmp = temp_root / f"test_copy_file_if_missing_{uuid.uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            source = tmp / "source.json"
            target = tmp / "nested" / "target.json"
            source.write_text('{"ok": true}', encoding="utf-8")

            created, written_target = copy_file_if_missing(source, target)

            self.assertTrue(created)
            self.assertEqual(written_target, target)
            self.assertEqual(target.read_text(encoding="utf-8"), '{"ok": true}')
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
