import sys
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import easy_dataset_workflow as workflow
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

    def test_run_everything_exports_training_pairs_and_relaxes_review_filter_after_empty_strict_export(self) -> None:
        export_manifests = [
            {"exported_pairs": 0, "skipped": {"review_required": 2}},
            {"exported_pairs": 1, "skipped": {}},
        ]

        with (
            mock.patch.object(workflow, "build_doctor_results", return_value=[]),
            mock.patch.object(workflow, "print_check"),
            mock.patch.object(workflow, "print_next_steps"),
            mock.patch.object(workflow, "run_prepare_dataset"),
            mock.patch.object(workflow, "setup_chatgpt_config"),
            mock.patch.object(workflow, "collect_enrichment_target_paths", return_value=[]),
            mock.patch.object(workflow, "run_enrichment"),
            mock.patch.object(workflow, "count_missing_llm_enrichment", return_value=0),
            mock.patch.object(workflow, "run_training_split"),
            mock.patch.object(workflow, "run_export_training_pairs", side_effect=export_manifests) as export_mock,
        ):
            workflow.run_everything(
                with_chatgpt=True,
                chatgpt_limit=5,
                chatgpt_language="de",
                chatgpt_max_assets_per_chat=10,
                chatgpt_dry_run=False,
                manual_login=True,
                config_path=None,
                reprocess_existing=False,
                keep_browser_open=None,
                with_split=False,
            )

        self.assertEqual(export_mock.call_count, 2)
        self.assertEqual(
            export_mock.call_args_list,
            [
                mock.call(min_quality="medium", require_llm=True, allow_review_required=False),
                mock.call(min_quality="medium", require_llm=True, allow_review_required=True),
            ],
        )

    def test_run_everything_exports_without_llm_requirement_when_chatgpt_is_disabled(self) -> None:
        with (
            mock.patch.object(workflow, "build_doctor_results", return_value=[]),
            mock.patch.object(workflow, "print_check"),
            mock.patch.object(workflow, "print_next_steps"),
            mock.patch.object(workflow, "run_prepare_dataset"),
            mock.patch.object(workflow, "setup_chatgpt_config") as setup_mock,
            mock.patch.object(workflow, "collect_enrichment_target_paths", return_value=[]),
            mock.patch.object(workflow, "run_enrichment") as enrich_mock,
            mock.patch.object(workflow, "count_missing_llm_enrichment", return_value=0),
            mock.patch.object(workflow, "run_training_split"),
            mock.patch.object(
                workflow,
                "run_export_training_pairs",
                return_value={"exported_pairs": 2, "skipped": {}},
            ) as export_mock,
        ):
            workflow.run_everything(
                with_chatgpt=False,
                chatgpt_limit=None,
                chatgpt_language="de",
                chatgpt_max_assets_per_chat=10,
                chatgpt_dry_run=False,
                manual_login=True,
                config_path=None,
                reprocess_existing=False,
                keep_browser_open=None,
                with_split=False,
            )

        setup_mock.assert_not_called()
        enrich_mock.assert_not_called()
        export_mock.assert_called_once_with(
            min_quality="medium",
            require_llm=False,
            allow_review_required=False,
        )

    def test_run_auto_export_training_pairs_does_not_relax_review_filter_when_llm_is_missing(self) -> None:
        manifest = {
            "exported_pairs": 0,
            "skipped": {
                "review_required": 1,
                "missing_llm_enrichment": 1,
            },
        }

        with mock.patch.object(workflow, "run_export_training_pairs", return_value=manifest) as export_mock:
            result = workflow.run_auto_export_training_pairs(with_chatgpt=True)

        self.assertEqual(result, manifest)
        export_mock.assert_called_once_with(
            min_quality="medium",
            require_llm=True,
            allow_review_required=False,
        )

    def test_run_everything_exports_completed_pairs_after_incomplete_enrichment(self) -> None:
        with (
            mock.patch.object(workflow, "build_doctor_results", return_value=[]),
            mock.patch.object(workflow, "print_check"),
            mock.patch.object(workflow, "print_next_steps"),
            mock.patch.object(workflow, "run_prepare_dataset"),
            mock.patch.object(workflow, "setup_chatgpt_config"),
            mock.patch.object(workflow, "collect_enrichment_target_paths", return_value=[Path("a.json"), Path("b.json")]),
            mock.patch.object(workflow, "run_enrichment"),
            mock.patch.object(workflow, "count_missing_llm_enrichment", return_value=1),
            mock.patch.object(workflow, "count_exportable_llm_assets", return_value=1),
            mock.patch.object(
                workflow,
                "run_export_training_pairs",
                return_value={"exported_pairs": 1, "skipped": {"missing_llm_enrichment": 1}},
            ) as export_mock,
        ):
            workflow.run_everything(
                with_chatgpt=True,
                chatgpt_limit=None,
                chatgpt_language="de",
                chatgpt_max_assets_per_chat=10,
                chatgpt_dry_run=False,
                manual_login=True,
                config_path=None,
                reprocess_existing=False,
                keep_browser_open=None,
                with_split=False,
            )

        export_mock.assert_called_once_with(
            min_quality="medium",
            require_llm=True,
            allow_review_required=False,
        )

    def test_run_everything_aborts_incomplete_enrichment_when_no_llm_assets_are_exportable(self) -> None:
        with (
            mock.patch.object(workflow, "build_doctor_results", return_value=[]),
            mock.patch.object(workflow, "print_check"),
            mock.patch.object(workflow, "print_next_steps"),
            mock.patch.object(workflow, "run_prepare_dataset"),
            mock.patch.object(workflow, "setup_chatgpt_config"),
            mock.patch.object(workflow, "collect_enrichment_target_paths", return_value=[Path("a.json"), Path("b.json")]),
            mock.patch.object(workflow, "run_enrichment"),
            mock.patch.object(workflow, "count_missing_llm_enrichment", return_value=2),
            mock.patch.object(workflow, "count_exportable_llm_assets", return_value=0),
            mock.patch.object(workflow, "run_export_training_pairs") as export_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "keine exportierbaren LLM-Assets"):
                workflow.run_everything(
                    with_chatgpt=True,
                    chatgpt_limit=None,
                    chatgpt_language="de",
                    chatgpt_max_assets_per_chat=10,
                    chatgpt_dry_run=False,
                    manual_login=True,
                    config_path=None,
                    reprocess_existing=False,
                    keep_browser_open=None,
                    with_split=False,
                )

        export_mock.assert_not_called()

    def test_has_chatgpt_auth_cookie_detects_session_cookie_names(self) -> None:
        self.assertTrue(
            workflow.has_chatgpt_auth_cookie(
                [
                    ".chatgpt.com:cf_clearance",
                    "chatgpt.com:__Secure-next-auth.session-token",
                ]
            )
        )
        self.assertFalse(
            workflow.has_chatgpt_auth_cookie(
                [
                    ".chatgpt.com:cf_clearance",
                    "chatgpt.com:__Host-next-auth.csrf-token",
                ]
            )
        )


if __name__ == "__main__":
    unittest.main()
