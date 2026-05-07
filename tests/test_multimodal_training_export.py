import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export_multimodal_training_pairs import build_training_response, export_training_pairs, should_export_annotation


class MultimodalTrainingExportTests(unittest.TestCase):
    def test_build_training_response_prefers_llm_but_only_uses_visible_normalized_fields(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "caption": "Fallback caption",
            "summary": "Fallback summary",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "short_caption": "BTCUSDT H1 bearish chart",
                    "visual_summary": "A bearish BTCUSDT chart with highlighted BOS zones.",
                    "visible_text": "Bitcoin / TetherUS, 1h, BINANCE",
                    "key_visual_elements": ["candlesticks", "price scale", "BOS labels"],
                    "limitations": ["small header text"],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "normalized_fields": {
                            "primary_symbol": "BTCUSDT",
                            "instrument_name": "Bitcoin / TetherUS",
                            "venue": "BINANCE",
                            "symbols": ["BTCUSDT"],
                            "timeframes": ["H1"],
                        },
                        "clean_text": "Visible labels: Bitcoin / TetherUS, 1h, BINANCE",
                    }
                },
                "provenance": {"quality": {"annotation_quality": "high"}},
            },
        }

        response = build_training_response(annotation)

        self.assertEqual(response["asset_type"], "chart")
        self.assertEqual(response["primary_symbol"], "BTCUSDT")
        self.assertEqual(response["venue"], "BINANCE")
        self.assertEqual(response["timeframes"], ["H1"])
        self.assertEqual(response["short_caption"], "BTCUSDT H1 bearish chart")
        self.assertEqual(response["visual_summary"], "A bearish BTCUSDT chart with highlighted BOS zones.")
        self.assertEqual(response["visible_text"], "Bitcoin / TetherUS, 1h, BINANCE")
        self.assertEqual(response["key_visual_elements"], ["candlesticks", "price scale", "BOS labels"])

    def test_should_export_annotation_filters_missing_llm_and_low_quality(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "caption": "Caption",
            "summary": "Summary",
            "review_required": False,
            "target_json": {
                "description": {"short_caption": "Caption", "visual_summary": "Summary"},
                "provenance": {"quality": {"annotation_quality": "low"}},
                "observed": {"visible_in_crop": {"normalized_fields": {}}},
            },
        }

        should_export, reason = should_export_annotation(
            annotation=annotation,
            min_quality="medium",
            require_llm=True,
            allow_review_required=False,
        )

        self.assertFalse(should_export)
        self.assertEqual(reason, "missing_llm_enrichment")

    def test_export_training_pairs_writes_compact_pair_folder_and_index(self) -> None:
        temp_root = ROOT / ".tmp" / f"test_training_export_{uuid.uuid4().hex}"
        source_dir = temp_root / "source_pairs"
        output_dir = temp_root / "training_pairs"
        index_path = temp_root / "training_pairs.jsonl"
        manifest_path = temp_root / "training_pairs_manifest.json"
        source_dir.mkdir(parents=True, exist_ok=True)

        image_path = source_dir / "asset_01.jpg"
        annotation_path = source_dir / "asset_01.json"
        image_path.write_bytes(b"fake-image")
        annotation_path.write_text(
            json.dumps(
                {
                    "id": "asset-1",
                    "pair_type": "visual_asset",
                    "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                    "asset_type": "chart",
                    "source_pdf": "03.10.pdf",
                    "page_number": 1,
                    "asset_index": 1,
                    "review_required": False,
                    "llm_enrichment": {
                        "structured_response": {
                            "short_caption": "BTCUSDT H1 chart",
                            "visual_summary": "A chart with candles and BOS labels.",
                            "visible_text": "Bitcoin / TetherUS, 1h, BINANCE",
                            "key_visual_elements": ["candlesticks", "BOS labels"],
                            "limitations": [],
                            "confidence": "high",
                        }
                    },
                    "target_json": {
                        "observed": {
                            "visible_in_crop": {
                                "normalized_fields": {
                                    "primary_symbol": "BTCUSDT",
                                    "instrument_name": "Bitcoin / TetherUS",
                                    "venue": "BINANCE",
                                    "symbols": ["BTCUSDT"],
                                    "timeframes": ["H1"],
                                },
                                "clean_text": "Visible labels: Bitcoin / TetherUS, 1h, BINANCE",
                            }
                        },
                        "provenance": {"quality": {"annotation_quality": "high"}},
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            manifest = export_training_pairs(
                glob_pattern=str((source_dir / "*.json").relative_to(ROOT)).replace("\\", "/"),
                output_dir=output_dir,
                index_path=index_path,
                manifest_path=manifest_path,
                min_quality="medium",
                require_llm=True,
                allow_review_required=False,
            )

            exported_json = output_dir / "asset_01.json"
            exported_image = output_dir / "asset_01.jpg"
            self.assertTrue(exported_json.exists())
            self.assertTrue(exported_image.exists())
            self.assertTrue((output_dir / "_instruction.txt").exists())
            self.assertEqual(manifest["exported_pairs"], 1)

            payload = json.loads(exported_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_type"], "strict_image_to_json")
            self.assertEqual(payload["response"]["primary_symbol"], "BTCUSDT")
            self.assertEqual(payload["response"]["short_caption"], "BTCUSDT H1 chart")

            index_rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(index_rows), 1)
            self.assertTrue(index_rows[0]["json_path"].endswith("asset_01.json"))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
