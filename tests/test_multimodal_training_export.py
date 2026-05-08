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
    def test_build_training_response_uses_visible_label_grounding_and_repairs_conflicts(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "figure",
            "page_type": "unknown",
            "caption": "Fallback candlestick chart",
            "summary": "Fallback summary",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "short_caption": "EURUSD H1 bearish chart",
                    "visual_summary": "A bearish EURUSD chart with highlighted BOS zones.",
                    "visible_text": "Euro / US-Dollar, 1h, FXCM",
                    "key_visual_elements": ["candlesticks", "price scale", "highlighted circles"],
                    "limitations": [
                        "annotation meaning is inferred rather than explicitly labeled",
                        "exact price values are difficult to read precisely",
                    ],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "normalized_fields": {
                            "primary_symbol": "EURUSD",
                            "instrument_name": "Euro / US-Dollar",
                            "venue": "FXCM",
                            "symbols": ["EURUSD"],
                            "timeframes": ["M5"],
                        },
                        "clean_text": "Asset type: chart\nTimeframes: H1\nVisible labels: Euro / US-Dollar - 5 - FXCM",
                        "visual_elements": ["generic_figure", "timeframe_label", "price_axis_or_scale"],
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
        }

        response = build_training_response(annotation)

        self.assertEqual(response["asset_type"], "chart")
        self.assertEqual(response["primary_symbol"], "EURUSD")
        self.assertEqual(response["instrument_name"], "Euro / US-Dollar")
        self.assertEqual(response["venue"], "FXCM")
        self.assertEqual(response["timeframes"], ["M5"])
        self.assertEqual(response["instrument"]["primary_symbol"], "EURUSD")
        self.assertEqual(response["instrument"]["timeframes"], ["M5"])
        self.assertEqual(response["visible_text"], "Euro / US-Dollar - 5 - FXCM")
        self.assertEqual(response["short_caption"], "Annotated EURUSD M5 chart with highlighted swing markers")
        self.assertEqual(
            response["visual_summary"],
            'Chart for EURUSD on M5. Visible elements include candlestick chart, price scale, and highlighted swing markers. Visible label text includes "Euro / US-Dollar - 5 - FXCM".',
        )
        self.assertEqual(response["chart_content"]["chart_type"], "candlestick")
        self.assertNotIn("bearish", response["chart_content"]["visible_market_behavior"].lower())
        self.assertNotIn("bos zones", response["chart_content"]["visible_market_behavior"].lower())
        self.assertTrue(response["chart_content"]["has_price_scale"])
        self.assertEqual(response["annotations"]["swing_markers"], "highlighted local highs or lows visible in the chart crop")
        self.assertEqual(response["visible_text_details"]["main_header"], "Euro / US-Dollar - 5 - FXCM")
        self.assertEqual(response["confidence"]["symbol"], "high")
        self.assertEqual(
            response["key_visual_elements"],
            ["candlestick chart", "price scale", "highlighted swing markers", "timeframe label"],
        )
        self.assertEqual(response["limitations"], ["exact price values are difficult to read precisely"])

    def test_build_training_response_keeps_visible_bos_label_when_grounded(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "short_caption": "BTCUSDT H1 chart with BOS label",
                    "visual_summary": "BTCUSDT H1 candlestick chart with a visible BOS label.",
                    "visible_text": "Bitcoin / TetherUS, 1h, BINANCE, BOS",
                    "key_visual_elements": ["candlesticks", "BOS labels", "price scale"],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "Visible labels: Bitcoin / TetherUS, 1h, BINANCE, BOS",
                        "visual_elements": ["chart_panel", "price_axis_or_scale"],
                    }
                },
                "provenance": {"quality": {"annotation_quality": "high"}},
            },
        }

        response = build_training_response(annotation)

        self.assertIn("BOS label", response["key_visual_elements"])
        self.assertTrue(response["chart_content"]["has_bos_label"])
        self.assertEqual(response["annotations"]["bos_label"], "visible BOS label or break-of-structure annotation")
        self.assertEqual(response["visual_summary"], "BTCUSDT H1 candlestick chart with a visible BOS label.")

    def test_build_training_response_does_not_reject_price_numbers_as_timeframes(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "short_caption": "EURUSD M5 chart near 1.0850",
                    "visual_summary": "EURUSD M5 candlestick chart with visible price action near 1.0850.",
                    "visible_text": "Euro / US-Dollar, 5, FXCM",
                    "key_visual_elements": ["candlesticks", "price scale"],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "Visible labels: Euro / US-Dollar, 5, FXCM",
                        "visual_elements": ["chart_panel", "price_axis_or_scale"],
                    }
                },
                "provenance": {"quality": {"annotation_quality": "high"}},
            },
        }

        response = build_training_response(annotation)

        self.assertEqual(response["short_caption"], "EURUSD M5 chart near 1.0850")
        self.assertEqual(
            response["visual_summary"],
            "EURUSD M5 candlestick chart with visible price action near 1.0850.",
        )

    def test_build_training_response_keeps_risk_reward_tool_as_trading_tool(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "short_caption": "EURUSD M5 chart with short position risk/reward tool",
                    "visual_summary": "EURUSD M5 candlestick chart with a visible short position risk/reward tool.",
                    "visible_text": "Euro / US-Dollar, 5, FXCM",
                    "key_visual_elements": [
                        "candlesticks",
                        "short position risk/reward tool",
                        "gray reward rectangle",
                        "price scale",
                    ],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "Visible labels: Euro / US-Dollar, 5, FXCM",
                        "visual_elements": ["chart_panel", "price_axis_or_scale"],
                    }
                },
                "provenance": {"quality": {"annotation_quality": "high"}},
            },
        }

        response = build_training_response(annotation)

        self.assertIn("short position tool", response["key_visual_elements"])
        self.assertTrue(response["chart_content"]["has_risk_reward_tool"])
        self.assertEqual(response["chart_content"]["position_tool_direction"], "short")
        self.assertEqual(
            response["visual_summary"],
            "EURUSD M5 candlestick chart with a visible short position risk/reward tool.",
        )
        self.assertEqual(
            response["annotations"]["risk_reward_position_tool"],
            "visible short position risk/reward tool with entry, risk, and reward regions",
        )

    def test_build_training_response_preserves_general_annotation_types(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "short_caption": "EURUSD M5 chart with labels, range box, and trendline",
                    "visual_summary": "EURUSD M5 candlestick chart with visible text labels, a range box, horizontal price line, and diagonal trendline.",
                    "visible_text": "Euro / US-Dollar, 5, FXCM",
                    "key_visual_elements": [
                        "candlesticks",
                        "text annotation label",
                        "horizontal support line",
                        "vertical time marker",
                        "diagonal trendline",
                        "range box",
                        "supply zone",
                        "price label",
                    ],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "Visible labels: Euro / US-Dollar, 5, FXCM",
                        "visual_elements": ["chart_panel", "price_axis_or_scale"],
                    }
                },
                "provenance": {"quality": {"annotation_quality": "high"}},
            },
        }

        response = build_training_response(annotation)

        self.assertIn("text annotation label", response["key_visual_elements"])
        self.assertIn("horizontal price line", response["key_visual_elements"])
        self.assertIn("vertical time marker", response["key_visual_elements"])
        self.assertIn("trendline or diagonal connector", response["key_visual_elements"])
        self.assertIn("range box", response["key_visual_elements"])
        self.assertIn("supply/demand zone", response["key_visual_elements"])
        self.assertTrue(response["chart_content"]["has_text_annotations"])
        self.assertTrue(response["chart_content"]["has_horizontal_price_lines"])
        self.assertTrue(response["chart_content"]["has_vertical_time_markers"])
        self.assertTrue(response["chart_content"]["has_trendlines"])
        self.assertTrue(response["chart_content"]["has_range_boxes"])
        self.assertTrue(response["chart_content"]["has_supply_demand_zones"])
        self.assertIn("text_annotations", response["annotations"])
        self.assertIn("horizontal_price_lines", response["annotations"])
        self.assertIn("vertical_time_markers", response["annotations"])
        self.assertIn("trendlines", response["annotations"])
        self.assertIn("range_boxes", response["annotations"])
        self.assertIn("supply_demand_zones", response["annotations"])

    def test_build_training_response_does_not_treat_vertical_time_marker_as_swing_marker(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "visible_text": "Euro / US-Dollar, 5, FXCM",
                    "key_visual_elements": ["candlesticks", "vertical time marker"],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "Visible labels: Euro / US-Dollar, 5, FXCM",
                        "visual_elements": ["chart_panel"],
                    }
                },
                "provenance": {"quality": {"annotation_quality": "high"}},
            },
        }

        response = build_training_response(annotation)

        self.assertTrue(response["chart_content"]["has_vertical_time_markers"])
        self.assertFalse(response["chart_content"]["has_swing_markers"])
        self.assertIn("vertical_time_markers", response["annotations"])
        self.assertNotIn("swing_markers", response["annotations"])

    def test_build_training_response_treats_price_label_as_text_annotation_in_chart_content(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "visible_text": "Euro / US-Dollar, 5, FXCM",
                    "key_visual_elements": ["candlesticks", "price label"],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "Visible labels: Euro / US-Dollar, 5, FXCM",
                        "visual_elements": ["chart_panel"],
                    }
                },
                "provenance": {"quality": {"annotation_quality": "high"}},
            },
        }

        response = build_training_response(annotation)

        self.assertTrue(response["chart_content"]["has_text_annotations"])
        self.assertIn("text_annotations", response["annotations"])

    def test_build_training_response_keeps_grounded_supply_demand_zone_summary(self) -> None:
        annotation = {
            "pair_type": "visual_asset",
            "asset_type": "chart",
            "review_required": False,
            "llm_enrichment": {
                "structured_response": {
                    "short_caption": "EURUSD M5 chart with supply zone",
                    "visual_summary": "EURUSD M5 candlestick chart with a visible supply zone.",
                    "visible_text": "Euro / US-Dollar, 5, FXCM",
                    "key_visual_elements": ["candlesticks", "supply zone", "price scale"],
                    "confidence": "high",
                }
            },
            "target_json": {
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "Visible labels: Euro / US-Dollar, 5, FXCM",
                        "visual_elements": ["chart_panel", "price_axis_or_scale"],
                    }
                },
                "provenance": {"quality": {"annotation_quality": "high"}},
            },
        }

        response = build_training_response(annotation)

        self.assertIn("supply/demand zone", response["key_visual_elements"])
        self.assertEqual(response["visual_summary"], "EURUSD M5 candlestick chart with a visible supply zone.")
        self.assertTrue(response["chart_content"]["has_supply_demand_zones"])

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
                                "visual_elements": ["chart_panel", "price_axis_or_scale"],
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
            self.assertEqual(manifest["grounding_profile"], "strict_visible_grounding")

            payload = json.loads(exported_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], "1.1")
            self.assertEqual(payload["task_type"], "strict_image_to_json")
            self.assertEqual(payload["response"]["primary_symbol"], "BTCUSDT")
            self.assertEqual(payload["response"]["instrument"]["primary_symbol"], "BTCUSDT")
            self.assertEqual(payload["response"]["short_caption"], "BTCUSDT H1 chart")
            self.assertEqual(payload["response"]["visible_text"], "Bitcoin / TetherUS - 1h - BINANCE")
            self.assertEqual(payload["response"]["chart_content"]["chart_type"], "candlestick")
            self.assertIn("confidence", payload["response"])

            index_rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(index_rows), 1)
            self.assertTrue(index_rows[0]["json_path"].endswith("asset_01.json"))
            self.assertEqual(index_rows[0]["exported_asset_type"], "chart")
            self.assertEqual(index_rows[0]["exported_primary_symbol"], "BTCUSDT")
            self.assertEqual(index_rows[0]["exported_timeframes"], ["H1"])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
