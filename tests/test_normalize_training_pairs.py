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

from normalize_multimodal_training_pairs import normalize_training_pairs


class NormalizeTrainingPairsTests(unittest.TestCase):
    def test_normalize_training_pairs_rewrites_existing_payloads_and_index(self) -> None:
        temp_root = ROOT / ".tmp" / f"test_normalize_training_pairs_{uuid.uuid4().hex}"
        source_dir = temp_root / "source_pairs"
        output_dir = temp_root / "training_pairs"
        index_path = temp_root / "training_pairs.jsonl"
        manifest_path = temp_root / "training_pairs_manifest.json"
        source_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        annotation_path = source_dir / "asset_01.json"
        training_json_path = output_dir / "asset_01.json"

        annotation_path.write_text(
            json.dumps(
                {
                    "id": "asset-1",
                    "pair_type": "visual_asset",
                    "image_path": "data/processed/multimodal/pairs/asset_01.jpg",
                    "asset_type": "chart",
                    "source_pdf": "Dezember.pdf",
                    "page_number": 1,
                    "asset_index": 1,
                    "review_required": False,
                    "ocr_text": "Euro / US-Dollar · 5 · FXCM",
                    "llm_enrichment": {
                        "structured_response": {
                            "short_caption": "old caption",
                            "visual_summary": "old summary",
                            "visible_text": "Euro / US-Dollar · 5 · FXCM",
                            "key_visual_elements": ["candlesticks", "highlighted circles", "directional arrows"],
                            "limitations": ["exact price values are difficult to read precisely"],
                            "confidence": "high",
                        }
                    },
                    "target_json": {
                        "observed": {
                            "visible_in_crop": {
                                "normalized_fields": {
                                    "primary_symbol": "EURUSD",
                                    "instrument_name": "Euro / US-Dollar · 5 · FXCM",
                                    "venue": "FXCM",
                                    "symbols": ["EURUSD"],
                                    "timeframes": ["M5"],
                                },
                                "clean_text": "Visible labels: Euro / US-Dollar · 5 · FXCM",
                                "visual_elements": ["chart_panel", "price_axis_or_scale", "timeframe_label"],
                            }
                        },
                        "provenance": {
                            "quality": {"annotation_quality": "high"},
                            "field_sources": {
                                "primary_symbol": "llm_enrichment",
                                "instrument_name": "llm_enrichment",
                                "venue": "llm_enrichment",
                                "symbols": "llm_enrichment",
                                "timeframes": "llm_enrichment",
                            },
                        },
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        training_json_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "task_type": "strict_image_to_json",
                    "response": {
                        "asset_type": "chart",
                        "primary_symbol": "EURUSD",
                        "instrument_name": "Euro / US-Dollar · 5 · FXCM",
                        "venue": "FXCM",
                        "short_caption": "Annotated EURUSD  chart with highlighted swing markers",
                        "visual_summary": "TradingView chart for EURUSD.",
                        "visible_text": "Euro / US-Dollar · 5 · FXCM",
                        "key_visual_elements": ["candlestick chart"],
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        index_path.write_text(
            json.dumps(
                {
                    "id": "asset-1",
                    "json_path": str(training_json_path.relative_to(ROOT)).replace("\\", "/"),
                    "source_annotation_path": str(annotation_path.relative_to(ROOT)).replace("\\", "/"),
                    "exported_asset_type": "chart",
                    "exported_primary_symbol": "EURUSD",
                    "exported_timeframes": [],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path.write_text(json.dumps({"exported_pairs": 1}, ensure_ascii=False), encoding="utf-8")

        try:
            manifest = normalize_training_pairs(output_dir=output_dir, index_path=index_path, manifest_path=manifest_path)

            payload = json.loads(training_json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["response"]["instrument_name"], "Euro / US-Dollar")
            self.assertEqual(payload["response"]["timeframes"], ["M5"])
            self.assertEqual(payload["response"]["visible_text"], "Euro / US-Dollar - 5 - FXCM")
            self.assertEqual(
                payload["response"]["short_caption"],
                "Annotated EURUSD M5 chart with highlighted swing markers and directional arrows",
            )

            index_rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(index_rows[0]["exported_timeframes"], ["M5"])
            self.assertEqual(index_rows[0]["exported_primary_symbol"], "EURUSD")

            self.assertEqual(manifest["normalized_pairs"], 1)
            self.assertEqual(manifest["index_rows"], 1)
            self.assertEqual(manifest["exported_pairs"], 1)
            self.assertEqual(manifest["normalizer"], "strict_visible_grounding_consistency_v1")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
