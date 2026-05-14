from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
import unicodedata

from text_normalization import repair_common_mojibake

ROOT = Path(__file__).resolve().parents[1]
RAW_PDFS_DIR = ROOT / "data" / "raw" / "pdfs"
RAW_NODES_DIR = ROOT / "data" / "raw" / "nodes"
PROCESSED_DIR = ROOT / "data" / "processed"
MULTIMODAL_DIR = PROCESSED_DIR / "multimodal"
MULTIMODAL_PAIRS_DIR = MULTIMODAL_DIR / "pairs"
MULTIMODAL_IMAGES_DIR = MULTIMODAL_DIR / "images"
MULTIMODAL_ANNOTATIONS_DIR = MULTIMODAL_DIR / "annotations"
PROJECT_TESSDATA_DIR = ROOT / ".tessdata"

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".yaml", ".yml"}
STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv"}
PDF_SUFFIX = ".pdf"

PDF_RENDER_SCALE = float(os.getenv("PDF_RENDER_SCALE", "2.0"))
ASSET_TARGET_LONG_EDGE_PX = int(os.getenv("ASSET_TARGET_LONG_EDGE_PX", "2200"))
ASSET_MAX_RENDER_SCALE = float(os.getenv("ASSET_MAX_RENDER_SCALE", "4.0"))
OCR_ENABLED = os.getenv("ENABLE_OCR", "1").strip().lower() not in {"0", "false", "no"}
OCR_LANGUAGE = os.getenv("TESSERACT_LANG", "eng")
ASSET_MIN_AREA_RATIO = float(os.getenv("ASSET_MIN_AREA_RATIO", "0.02"))
ASSET_MIN_WIDTH_RATIO = float(os.getenv("ASSET_MIN_WIDTH_RATIO", "0.2"))
ASSET_MIN_HEIGHT_RATIO = float(os.getenv("ASSET_MIN_HEIGHT_RATIO", "0.12"))
ASSET_CONTEXT_MARGIN = float(os.getenv("ASSET_CONTEXT_MARGIN", "42"))
ASSET_RENDER_PADDING = float(os.getenv("ASSET_RENDER_PADDING", "10"))
ASSET_MAX_CONTEXT_BLOCKS = int(os.getenv("ASSET_MAX_CONTEXT_BLOCKS", "6"))
DRAWING_CLUSTER_GAP = float(os.getenv("DRAWING_CLUSTER_GAP", "14"))
TEXT_GAP_MIN_RATIO = float(os.getenv("TEXT_GAP_MIN_RATIO", "0.16"))
SUMMARY_CHAR_LIMIT = 360
LLM_ENRICHMENT_PRESERVED_FIELDS = (
    "caption",
    "summary",
    "description",
    "clean_text",
    "extraction_methods",
    "primary_symbol",
    "instrument_name",
    "venue",
    "symbols",
    "timeframes",
    "review_required",
    "llm_enrichment",
)
LLM_TARGET_DESCRIPTION_FIELDS = (
    "short_caption",
    "visual_summary",
    "context_augmented_summary",
    "key_visual_elements",
    "limitations",
)

BBox = tuple[float, float, float, float]

WINDOWS_TESSERACT_CANDIDATES = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
]
TESSDATA_CANDIDATES = [
    Path("/usr/share/tesseract-ocr/5/tessdata"),
    Path("/usr/share/tesseract-ocr/4.00/tessdata"),
    Path("/usr/share/tessdata"),
    Path("/opt/homebrew/share/tessdata"),
    Path("/usr/local/share/tessdata"),
]

FOREX_CODES = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "AUD",
    "NZD",
    "CAD",
    "CHF",
}
KNOWN_GLOBAL_SYMBOLS = {
    "BTCUSD",
    "BTCUSDT",
    "ETHUSD",
    "ETHUSDT",
    "XAUUSD",
    "XAGUSD",
    "US30",
    "NAS100",
    "SPX500",
    "GER40",
    "UK100",
    "WTI",
    "BRENT",
}

KNOWN_VENUES = {
    "BINANCE",
    "BYBIT",
    "BITGET",
    "COINBASE",
    "KRAKEN",
    "OKX",
}

BASE_ASSET_ALIASES = {
    "bitcoin": "BTC",
    "bitokn": "BTC",
    "btc": "BTC",
    "ethereum": "ETH",
    "ether": "ETH",
    "eth": "ETH",
    "gold": "XAU",
    "silver": "XAG",
    "nasdaq": "NAS100",
    "dow": "US30",
    "sp500": "SPX500",
}

QUOTE_ASSET_ALIASES = {
    "tether": "USDT",
    "tethers": "USDT",
    "tetherus": "USDT",
    "tetherusd": "USDT",
    "usdt": "USDT",
    "usd": "USD",
    "dollar": "USD",
    "dollars": "USD",
}

SYMBOL_DISPLAY_NAMES = {
    "BTCUSDT": "Bitcoin / TetherUS",
    "BTCUSD": "Bitcoin / US Dollar",
    "ETHUSDT": "Ethereum / TetherUS",
    "ETHUSD": "Ethereum / US Dollar",
    "XAUUSD": "Gold / US Dollar",
    "XAGUSD": "Silver / US Dollar",
}

TIMEFRAME_MAP = {
    "1m": "M1",
    "3m": "M3",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "45m": "M45",
    "1h": "H1",
    "2h": "H2",
    "4h": "H4",
    "6h": "H6",
    "8h": "H8",
    "12h": "H12",
    "1d": "D1",
    "1w": "W1",
    "1mo": "MN1",
}

CONCEPT_KEYWORDS = {
    "breakout": ["breakout", "range break", "compression break"],
    "pullback": ["pullback", "retest", "retracement"],
    "trend_following": ["trend", "trendline", "trend following"],
    "mean_reversion": ["mean reversion", "reversion", "overextended"],
    "market_structure": ["higher high", "lower low", "bos", "choch", "market structure"],
    "support_resistance": ["support", "resistance", "s/r", "level"],
    "risk_management": ["risk", "risk management", "position size", "risk reward", "risk/reward", "r:r", "r/r"],
    "stop_loss": ["stop loss", "sl", "stop"],
    "take_profit": ["take profit", "tp", "target"],
    "liquidity": ["liquidity", "sweep", "stop hunt"],
    "order_flow": ["order flow", "footprint", "delta", "bid ask"],
    "volume": ["volume", "vpvr", "volume profile"],
    "divergence": ["divergence", "hidden divergence"],
    "reversal": ["reversal", "fade", "turnaround"],
    "range": ["range", "sideways", "consolidation"],
    "volatility": ["volatility", "atr", "expansion", "compression"],
    "momentum": ["momentum", "impulse", "acceleration"],
    "psychology": ["discipline", "psychology", "emotion", "mindset"],
}

DOMAIN_KEYWORDS = {
    "technical_analysis": [
        "candlestick",
        "support",
        "resistance",
        "breakout",
        "trendline",
        "chart",
        "pattern",
    ],
    "risk_management": ["risk", "drawdown", "stop loss", "position size", "exposure"],
    "performance_review": ["pnl", "profit factor", "win rate", "equity curve", "drawdown", "sharpe"],
    "execution": ["entry", "exit", "limit order", "market order", "execution", "fill"],
    "macro_context": ["cpi", "fomc", "rates", "macro", "news", "fed", "ecb"],
    "trading_psychology": ["discipline", "emotion", "mindset", "patience", "fear", "greed"],
}

FIELD_ALIASES = {
    "symbol": ["symbol", "ticker", "asset", "instrument", "market"],
    "timeframe": ["timeframe", "tf", "interval"],
    "direction": ["direction", "side", "bias", "action", "signal"],
    "entry": ["entry", "entry_price", "buy_price", "sell_price", "open_price"],
    "stop_loss": ["stop_loss", "sl", "stop"],
    "take_profit": ["take_profit", "tp", "target", "target_price"],
    "thesis": ["thesis", "reason", "rationale", "setup", "setup_reason"],
    "notes": ["notes", "comment", "comments", "summary", "description"],
    "outcome": ["outcome", "result", "status"],
    "pnl": ["pnl", "profit", "profit_loss", "return", "return_pct", "roi"],
    "start_time": ["start_time", "open_time", "entry_time", "timestamp", "date"],
    "end_time": ["end_time", "close_time", "exit_time"],
}


@dataclass
class Document:
    doc_id: str
    source_type: str
    relative_path: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    structured_payload: dict[str, Any] | None = None


@dataclass(frozen=True)
class LayoutTextBlock:
    bbox: BBox
    text: str
    font_size: float = 0.0


@dataclass(frozen=True)
class LayoutImageBlock:
    bbox: BBox
    ext: str
    image_bytes: bytes
    width: int = 0
    height: int = 0
    xres: int = 0
    yres: int = 0


@dataclass(frozen=True)
class VisualAssetCandidate:
    bbox: BBox
    source: str
    image_block: LayoutImageBlock | None = None


def safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def normalize_whitespace(text: str) -> str:
    text = repair_common_mojibake(text)
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def flatten_value(value: Any, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, dict):
        for key, inner in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(flatten_value(inner, next_prefix))
        return lines
    if isinstance(value, list):
        for index, inner in enumerate(value):
            next_prefix = f"{prefix}[{index}]"
            lines.extend(flatten_value(inner, next_prefix))
        return lines
    scalar = "" if value is None else str(value)
    if prefix:
        lines.append(f"{prefix}: {scalar}")
    else:
        lines.append(scalar)
    return lines


def extract_tags(path: Path) -> list[str]:
    tags: list[str] = []
    for part in path.parts:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", part.strip().lower()).strip("-")
        if cleaned:
            tags.append(cleaned)
    return tags[-6:]


def slugify(value: str) -> str:
    normalized = value.replace("\\", "/").strip("/")
    normalized = re.sub(r"[^a-zA-Z0-9/_-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    return normalized.strip("-")


def root_relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def normalize_image_extension(value: str) -> str:
    normalized = value.strip().lower().lstrip(".")
    extension_map = {
        "jpeg": "jpg",
        "jpg": "jpg",
        "png": "png",
        "webp": "webp",
        "bmp": "bmp",
        "tif": "tif",
        "tiff": "tif",
    }
    return extension_map.get(normalized, "")


def build_asset_pair_id(source_pdf: str, page_number: int, asset_index: int) -> str:
    return stable_id("pdf_asset", source_pdf, str(page_number), str(asset_index))


def build_asset_pair_basename(source_pdf: str, page_number: int, asset_index: int) -> str:
    pdf_slug = slugify(str(Path(source_pdf).with_suffix(""))).replace("/", "__")
    pair_id = build_asset_pair_id(source_pdf, page_number, asset_index)
    return f"{pdf_slug}__p{page_number:04d}__a{asset_index:02d}__{pair_id}"


def build_asset_pair_paths(
    source_pdf: str,
    page_number: int,
    asset_index: int,
    image_extension: str = "png",
) -> tuple[Path, Path]:
    normalized_extension = normalize_image_extension(image_extension) or "png"
    basename = build_asset_pair_basename(source_pdf, page_number, asset_index)
    return (
        MULTIMODAL_PAIRS_DIR / f"{basename}.{normalized_extension}",
        MULTIMODAL_PAIRS_DIR / f"{basename}.json",
    )


def ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", ascii_fold(value).lower())


def fuzzy_lookup(token: str, candidates: dict[str, str], minimum_score: float = 0.72) -> str | None:
    normalized_token = normalize_token(token)
    if not normalized_token:
        return None
    if normalized_token in candidates:
        return candidates[normalized_token]
    if len(normalized_token) < 4:
        return None

    best_match: str | None = None
    best_score = 0.0
    for candidate_key, candidate_value in candidates.items():
        if candidate_key in normalized_token or normalized_token in candidate_key:
            score = 0.92
        else:
            score = SequenceMatcher(a=normalized_token, b=candidate_key).ratio()
        if score > best_score:
            best_score = score
            best_match = candidate_value
    if best_score >= minimum_score:
        return best_match
    return None


def try_extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    try:
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page_index, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_text = normalize_whitespace(page_text)
            if page_text:
                pages.append(f"[page {page_index + 1}]\n{page_text}")
        return "\n\n".join(pages)
    except Exception as exc:  # pragma: no cover - defensive for unstable PDFs
        return f"[pdf_extraction_error] {exc}"


def parse_text_file(path: Path, source_root: Path, source_type: str) -> list[Document]:
    text = normalize_whitespace(safe_read_text(path))
    relative = path.relative_to(source_root).as_posix()
    return [
        Document(
            doc_id=stable_id(source_type, relative),
            source_type=source_type,
            relative_path=relative,
            title=path.stem,
            text=text,
            tags=extract_tags(path.relative_to(source_root)),
        )
    ]


def parse_pdf_file(path: Path) -> list[Document]:
    text = normalize_whitespace(try_extract_pdf(path))
    relative = path.relative_to(RAW_PDFS_DIR).as_posix()
    return [
        Document(
            doc_id=stable_id("pdf", relative),
            source_type="pdf",
            relative_path=relative,
            title=path.stem,
            text=text,
            tags=extract_tags(path.relative_to(RAW_PDFS_DIR)),
        )
    ]


def parse_json_file(path: Path, source_root: Path) -> list[Document]:
    relative = path.relative_to(source_root).as_posix()
    payload = json.loads(safe_read_text(path))
    documents: list[Document] = []

    if isinstance(payload, list):
        for index, item in enumerate(payload):
            item_text = normalize_whitespace("\n".join(flatten_value(item)))
            doc = Document(
                doc_id=stable_id("json", relative, str(index)),
                source_type="node",
                relative_path=relative,
                title=f"{path.stem}_{index}",
                text=item_text,
                tags=extract_tags(path.relative_to(source_root)),
                structured_payload=item if isinstance(item, dict) else None,
                metadata={"record_index": index},
            )
            documents.append(doc)
        return documents

    item_text = normalize_whitespace("\n".join(flatten_value(payload)))
    documents.append(
        Document(
            doc_id=stable_id("json", relative),
            source_type="node",
            relative_path=relative,
            title=path.stem,
            text=item_text,
            tags=extract_tags(path.relative_to(source_root)),
            structured_payload=payload if isinstance(payload, dict) else None,
        )
    )
    return documents


def parse_jsonl_file(path: Path, source_root: Path) -> list[Document]:
    relative = path.relative_to(source_root).as_posix()
    documents: list[Document] = []
    for index, raw_line in enumerate(safe_read_text(path).splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = {"raw_line": line}
        text = normalize_whitespace("\n".join(flatten_value(payload)))
        documents.append(
            Document(
                doc_id=stable_id("jsonl", relative, str(index)),
                source_type="node",
                relative_path=relative,
                title=f"{path.stem}_{index}",
                text=text,
                tags=extract_tags(path.relative_to(source_root)),
                structured_payload=payload if isinstance(payload, dict) else None,
                metadata={"record_index": index},
            )
        )
    return documents


def parse_csv_file(path: Path, source_root: Path) -> list[Document]:
    relative = path.relative_to(source_root).as_posix()
    documents: list[Document] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            cleaned_row = {key: value for key, value in row.items() if key is not None}
            text = normalize_whitespace("\n".join(flatten_value(cleaned_row)))
            documents.append(
                Document(
                    doc_id=stable_id("csv", relative, str(index)),
                    source_type="node",
                    relative_path=relative,
                    title=f"{path.stem}_{index}",
                    text=text,
                    tags=extract_tags(path.relative_to(source_root)),
                    structured_payload=cleaned_row,
                    metadata={"record_index": index},
                )
            )
    return documents


def parse_path(path: Path, source_root: Path, source_type: str) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == PDF_SUFFIX:
        return parse_pdf_file(path)
    if suffix in TEXT_SUFFIXES:
        return parse_text_file(path, source_root, source_type)
    if suffix == ".json":
        return parse_json_file(path, source_root)
    if suffix == ".jsonl":
        return parse_jsonl_file(path, source_root)
    if suffix == ".csv":
        return parse_csv_file(path, source_root)
    return []


def collect_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(RAW_PDFS_DIR.rglob("*")):
        if path.is_file():
            documents.extend(parse_path(path, RAW_PDFS_DIR, "pdf"))
    for path in sorted(RAW_NODES_DIR.rglob("*")):
        if path.is_file():
            documents.extend(parse_path(path, RAW_NODES_DIR, "node"))
    return [document for document in documents if document.text]


def chunk_text(text: str, max_words: int = 320, overlap_words: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    step = max_words - overlap_words
    while start < len(words):
        chunk = words[start : start + max_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        start += step
    return chunks


def normalize_record(payload: dict[str, Any]) -> dict[str, Any]:
    lowered = {str(key).lower(): value for key, value in payload.items()}
    normalized: dict[str, Any] = {}
    for target_key, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias in lowered and lowered[alias] not in (None, "", []):
                normalized[target_key] = lowered[alias]
                break

    for passthrough_key in ("strategy", "session", "exchange", "broker"):
        if passthrough_key in lowered and lowered[passthrough_key] not in (None, "", []):
            normalized[passthrough_key] = lowered[passthrough_key]

    return normalized


def estimate_trade_confidence(record: dict[str, Any]) -> float:
    score = 0.0
    if record.get("symbol"):
        score += 0.2
    if record.get("timeframe"):
        score += 0.15
    if record.get("direction"):
        score += 0.2
    if record.get("entry"):
        score += 0.15
    if record.get("thesis") or record.get("notes"):
        score += 0.15
    if record.get("outcome") or record.get("pnl"):
        score += 0.15
    return round(min(score, 1.0), 2)


def build_decision_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    if not record.get("symbol"):
        return None
    if not (record.get("direction") or record.get("entry")):
        return None

    prompt_record = {
        key: value
        for key, value in record.items()
        if key not in {"outcome", "pnl", "end_time"}
    }
    response_record = {
        "action": record.get("direction"),
        "entry": record.get("entry"),
        "stop_loss": record.get("stop_loss"),
        "take_profit": record.get("take_profit"),
        "thesis": record.get("thesis") or record.get("notes"),
    }
    response_record = {key: value for key, value in response_record.items() if value not in (None, "")}
    if not response_record:
        return None

    return [
        {
            "role": "system",
            "content": (
                "You are a trading backtesting assistant. Use only the provided context "
                "and respond with a compact decision summary."
            ),
        },
        {
            "role": "user",
            "content": (
                "Review this historical trade setup and state the intended decision.\n\n"
                f"{json.dumps(prompt_record, ensure_ascii=False, indent=2)}"
            ),
        },
        {
            "role": "assistant",
            "content": json.dumps(response_record, ensure_ascii=False, indent=2),
        },
    ]


def build_documents_jsonl(documents: Iterable[Document]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        rows.append(
            {
                "id": document.doc_id,
                "source_type": document.source_type,
                "source_path": document.relative_path,
                "title": document.title,
                "tags": document.tags,
                "metadata": document.metadata,
                "text": document.text,
            }
        )
    return rows


def build_knowledge_jsonl(documents: Iterable[Document]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        for chunk_index, chunk in enumerate(chunk_text(document.text)):
            rows.append(
                {
                    "id": f"{document.doc_id}_chunk_{chunk_index}",
                    "source_document_id": document.doc_id,
                    "source_type": document.source_type,
                    "source_path": document.relative_path,
                    "title": document.title,
                    "chunk_index": chunk_index,
                    "tags": document.tags,
                    "text": chunk,
                }
            )
    return rows


def build_trade_candidates(documents: Iterable[Document]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        if not document.structured_payload:
            continue
        normalized_record = normalize_record(document.structured_payload)
        confidence = estimate_trade_confidence(normalized_record)
        if confidence < 0.35:
            continue
        messages = build_decision_messages(normalized_record)
        rows.append(
            {
                "id": document.doc_id,
                "source_path": document.relative_path,
                "source_title": document.title,
                "confidence": confidence,
                "review_required": True,
                "normalized_record": normalized_record,
                "candidate_training_example": {
                    "id": document.doc_id,
                    "messages": messages,
                    "metadata": {
                        "task_type": "decision",
                        "source_path": document.relative_path,
                        "symbol": normalized_record.get("symbol"),
                        "timeframe": normalized_record.get("timeframe"),
                        "reviewed": False,
                    },
                }
                if messages
                else None,
            }
        )
    return rows


def import_pymupdf() -> Any | None:
    try:
        import fitz
    except ImportError:
        return None
    return fitz


def import_ocr_modules() -> tuple[Any | None, Any | None]:
    try:
        from PIL import Image
        import pytesseract
    except ImportError:
        return None, None
    return Image, pytesseract


def resolve_tesseract_cmd() -> str | None:
    env_value = os.getenv("TESSERACT_CMD", "").strip()
    if env_value:
        return env_value

    for candidate in WINDOWS_TESSERACT_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    resolved = shutil.which("tesseract")
    return resolved if resolved else None


def resolve_tessdata_prefix() -> str | None:
    env_value = os.getenv("TESSDATA_PREFIX", "").strip()
    if env_value:
        return env_value

    if PROJECT_TESSDATA_DIR.exists():
        return str(PROJECT_TESSDATA_DIR)

    for candidate in TESSDATA_CANDIDATES:
        if candidate.exists():
            return str(candidate)

    resolved_cmd = resolve_tesseract_cmd()
    if resolved_cmd:
        install_tessdata = Path(resolved_cmd).resolve().parent / "tessdata"
        if install_tessdata.exists():
            return str(install_tessdata)
    return None


def configure_tesseract_runtime(pytesseract_module: Any | None = None) -> dict[str, Any]:
    tesseract_cmd = resolve_tesseract_cmd()
    tessdata_prefix = resolve_tessdata_prefix()

    if pytesseract_module is not None and tesseract_cmd:
        pytesseract_module.pytesseract.tesseract_cmd = tesseract_cmd

    if tessdata_prefix:
        os.environ["TESSDATA_PREFIX"] = tessdata_prefix

    return {
        "command": tesseract_cmd,
        "tessdata_prefix": tessdata_prefix,
    }


def detect_ocr_runtime() -> dict[str, Any]:
    if not OCR_ENABLED:
        return {"enabled": False, "available": False, "status": "disabled"}

    _image_module, pytesseract_module = import_ocr_modules()
    if pytesseract_module is None:
        return {
            "enabled": True,
            "available": False,
            "status": "missing_python_packages",
        }

    runtime_config = configure_tesseract_runtime(pytesseract_module)

    try:
        version = str(pytesseract_module.get_tesseract_version())
    except Exception as exc:  # pragma: no cover - depends on local OCR setup
        return {
            "enabled": True,
            "available": False,
            "status": "missing_tesseract_binary",
            "error": str(exc),
            "command": runtime_config["command"],
            "tessdata_prefix": runtime_config["tessdata_prefix"],
        }

    return {
        "enabled": True,
        "available": True,
        "status": "ok",
        "version": version,
        "language": OCR_LANGUAGE,
        "command_override": runtime_config["command"],
        "tessdata_prefix": runtime_config["tessdata_prefix"],
    }


def run_ocr(image_path: Path) -> tuple[str, dict[str, Any]]:
    if not OCR_ENABLED:
        return "", {"enabled": False, "available": False, "status": "disabled"}

    image_module, pytesseract_module = import_ocr_modules()
    if image_module is None or pytesseract_module is None:
        return "", {
            "enabled": True,
            "available": False,
            "status": "missing_python_packages",
        }

    runtime_config = configure_tesseract_runtime(pytesseract_module)

    try:
        from PIL import ImageOps

        with image_module.open(image_path) as image:
            primary_text = pytesseract_module.image_to_string(image, lang=OCR_LANGUAGE, config="--psm 6")

            header_height = max(int(image.height * 0.12), 72)
            header_crop = image.crop((0, 0, image.width, min(image.height, header_height)))
            header_image = ImageOps.autocontrast(header_crop.convert("L"))
            header_image = header_image.resize(
                (max(header_image.width * 2, header_image.width), max(header_image.height * 2, header_image.height))
            )
            header_text = pytesseract_module.image_to_string(header_image, lang=OCR_LANGUAGE, config="--psm 6")

        merged_text = "\n".join(unique_lines([primary_text, header_text]))
        return normalize_whitespace(merged_text), {
            "enabled": True,
            "available": True,
            "status": "ok",
            "language": OCR_LANGUAGE,
            "command": runtime_config["command"],
            "tessdata_prefix": runtime_config["tessdata_prefix"],
        }
    except Exception as exc:  # pragma: no cover - depends on local OCR setup
        return "", {
            "enabled": True,
            "available": False,
            "status": "runtime_error",
            "error": str(exc),
            "command": runtime_config["command"],
            "tessdata_prefix": runtime_config["tessdata_prefix"],
        }


def unique_lines(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for line in lines:
        cleaned = normalize_whitespace(line)
        if not cleaned:
            continue
        if cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        result.append(cleaned)
    return result


def merge_extracted_text(embedded_text: str, ocr_text: str) -> tuple[str, list[str]]:
    methods: list[str] = []
    if embedded_text:
        methods.append("embedded_text")
    if ocr_text:
        methods.append("ocr")

    if not embedded_text and not ocr_text:
        return "", methods
    if embedded_text and not ocr_text:
        return embedded_text, methods
    if ocr_text and not embedded_text:
        return ocr_text, methods

    embedded_lines = embedded_text.splitlines()
    ocr_lines = ocr_text.splitlines()
    merged = unique_lines([*embedded_lines, *ocr_lines])
    return "\n".join(merged), methods


def text_density_bucket(text: str) -> str:
    word_count = len(text.split())
    if word_count < 25:
        return "low"
    if word_count < 120:
        return "medium"
    return "high"


def split_ocr_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9]+", ascii_fold(text))


def extract_pair_symbols(text: str) -> list[str]:
    symbols: set[str] = set()
    candidate_lines = unique_lines(text.splitlines())
    for line in candidate_lines:
        if "/" not in line:
            continue
        parts = re.split(r"\s*/\s*", line, maxsplit=1)
        if len(parts) != 2:
            continue
        left_tokens = split_ocr_tokens(parts[0])
        right_tokens = split_ocr_tokens(parts[1])
        base_symbol = None
        for token in left_tokens:
            base_symbol = fuzzy_lookup(token, BASE_ASSET_ALIASES, minimum_score=0.66)
            if base_symbol:
                break

        quote_symbol = None
        for token in right_tokens[:5]:
            quote_symbol = fuzzy_lookup(token, QUOTE_ASSET_ALIASES, minimum_score=0.66)
            if quote_symbol:
                break

        if base_symbol and quote_symbol:
            symbols.add(base_symbol + quote_symbol)

    if symbols:
        return sorted(symbols)

    tokens = split_ocr_tokens(text)
    for index, token in enumerate(tokens):
        base_symbol = fuzzy_lookup(token, BASE_ASSET_ALIASES, minimum_score=0.68)
        if not base_symbol:
            continue
        for following_token in tokens[index + 1 : index + 5]:
            quote_symbol = fuzzy_lookup(following_token, QUOTE_ASSET_ALIASES, minimum_score=0.68)
            if quote_symbol:
                symbols.add(base_symbol + quote_symbol)
                break
    return sorted(symbols)


def extract_symbols(text: str) -> list[str]:
    if not text:
        return []

    symbols: set[str] = set()
    compact_text = text.upper()
    symbols.update(extract_pair_symbols(text))

    forex_matches = re.findall(r"\b([A-Z]{6})\b", compact_text)
    for token in forex_matches:
        if token[:3] in FOREX_CODES and token[3:] in FOREX_CODES:
            symbols.add(token)

    for token in re.findall(r"\$([A-Z]{1,5})\b", compact_text):
        symbols.add(token)

    for token in re.findall(r"\b([A-Z]{2,6}USDT|[A-Z]{2,6}USD)\b", compact_text):
        if token in KNOWN_GLOBAL_SYMBOLS or token.endswith(("USD", "USDT")):
            symbols.add(token)

    for known_symbol in KNOWN_GLOBAL_SYMBOLS:
        if known_symbol in compact_text:
            symbols.add(known_symbol)

    return sorted(symbols)


def extract_venues(text: str) -> list[str]:
    if not text:
        return []

    venues: set[str] = set()
    compact_text = ascii_fold(text).upper()
    for venue in KNOWN_VENUES:
        if venue in compact_text:
            venues.add(venue)

    venue_aliases = {normalize_token(venue): venue for venue in KNOWN_VENUES}
    for token in split_ocr_tokens(text):
        fuzzy_venue = fuzzy_lookup(token, venue_aliases, minimum_score=0.8)
        if fuzzy_venue:
            venues.add(fuzzy_venue)
    return sorted(venues)


def extract_timeframes(text: str) -> list[str]:
    if not text:
        return []

    normalized = text.lower()
    found: set[str] = set()

    for raw_timeframe, canonical in TIMEFRAME_MAP.items():
        if re.search(rf"\b{re.escape(raw_timeframe)}\b", normalized):
            found.add(canonical)

    shorthand_patterns = [
        ("M1", r"\bm1\b"),
        ("M5", r"\bm5\b"),
        ("M15", r"\bm15\b"),
        ("M15", r"\b1s\b"),
        ("M30", r"\bm30\b"),
        ("H1", r"\bh1\b"),
        ("H1", r"\bih\b"),
        ("H1", r"\blh\b"),
        ("H4", r"\bh4\b"),
        ("D1", r"\bd1\b"),
        ("W1", r"\bw1\b"),
    ]
    for canonical, pattern in shorthand_patterns:
        if re.search(pattern, normalized):
            found.add(canonical)

    return sorted(found)


def derive_asset_market_fields(context_text: str, ocr_text: str, fallback_text: str = "") -> dict[str, Any]:
    merged_text, _unused_methods = merge_extracted_text(context_text, ocr_text)
    analysis_text = fallback_text or merged_text or ocr_text or context_text

    visible_symbols = extract_symbols(ocr_text)
    symbols = visible_symbols or extract_symbols(analysis_text)
    primary_symbol = symbols[0] if symbols else None
    instrument_name = display_name_for_symbol(primary_symbol)

    visible_venues = extract_venues(ocr_text)
    venues = visible_venues or extract_venues(analysis_text)
    venue = venues[0] if venues else None

    visible_timeframes = extract_timeframes(ocr_text)
    timeframes = visible_timeframes or extract_timeframes(analysis_text)

    return {
        "symbols": symbols,
        "primary_symbol": primary_symbol,
        "instrument_name": instrument_name,
        "venue": venue,
        "timeframes": timeframes,
    }


def keyword_matches_normalized_text(normalized_text: str, keyword: str) -> bool:
    normalized_keyword = ascii_fold(keyword).lower().strip()
    if not normalized_text or not normalized_keyword:
        return False
    if not re.search(r"[a-z0-9]", normalized_keyword):
        return normalized_keyword in normalized_text
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])"
    return bool(re.search(pattern, normalized_text))


def extract_keyword_labels(text: str, keyword_map: dict[str, list[str]]) -> list[str]:
    lowered = ascii_fold(text).lower()
    labels = [
        label
        for label, keywords in keyword_map.items()
        if any(keyword_matches_normalized_text(lowered, keyword) for keyword in keywords)
    ]
    return sorted(labels)


def sentence_excerpt(text: str, max_length: int = SUMMARY_CHAR_LIMIT) -> str:
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""
    cleaned = cleaned.replace("\n", " ")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    sentence_candidates = re.split(r"(?<=[.!?])\s+", cleaned)
    excerpt = sentence_candidates[0].strip() if sentence_candidates else cleaned
    if len(excerpt) > max_length:
        excerpt = excerpt[: max_length - 3].rstrip() + "..."
    return excerpt


def is_date_or_time_line(line: str) -> bool:
    lowered = line.strip().lower()
    if not lowered:
        return True
    if re.fullmatch(r"\d{1,2}[.:]\d{2}", lowered):
        return True
    if re.fullmatch(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", lowered):
        return True
    if re.fullmatch(
        r"(montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag),?\s+\d{1,2}\.\s+\w+\s+\d{4}",
        lowered,
    ):
        return True
    return False


def meaningful_text_lines(text: str) -> list[str]:
    return [
        line
        for line in polish_context_lines(text)
        if not is_date_or_time_line(line)
    ]


def select_focus_line(*texts: str) -> str:
    for text in texts:
        for line in meaningful_text_lines(text):
            if len(line) >= 8:
                return line
    for text in texts:
        lines = polish_context_lines(text)
        if lines:
            return lines[0]
    return ""


def is_informative_ocr_line(line: str) -> bool:
    normalized = ascii_fold(line)
    words = re.findall(r"[A-Za-z]{3,}", normalized)
    digits = sum(character.isdigit() for character in normalized)
    letters = sum(character.isalpha() for character in normalized)
    punctuation = sum(not character.isalnum() and not character.isspace() for character in normalized)
    lowered = normalized.lower()
    if "tradingview.com" in lowered or "freigegeben" in lowered:
        return False
    if letters + digits < 8:
        return False
    if not words:
        return False
    if digits > max(letters * 2, 8) and len(words) < 2:
        return False
    if punctuation > max(letters + digits, 1):
        return False
    return True


def select_visible_label_lines(ocr_text: str) -> list[str]:
    selected: list[str] = []
    for line in meaningful_text_lines(ocr_text):
        if not is_informative_ocr_line(line):
            continue
        selected.append(line)
    return selected[:3]


def infer_page_type(text: str, symbols: list[str], timeframes: list[str]) -> tuple[str, float]:
    lowered = text.lower()
    scores = {
        "chart": 0,
        "trade_journal": 0,
        "performance_report": 0,
        "strategy_note": 0,
        "market_analysis": 0,
        "educational_note": 0,
    }

    chart_keywords = ["chart", "candlestick", "support", "resistance", "trendline", "breakout"]
    journal_keywords = ["entry", "exit", "stop loss", "take profit", "trade review", "journal"]
    performance_keywords = ["pnl", "drawdown", "win rate", "profit factor", "sharpe", "equity"]
    strategy_keywords = ["strategy", "rules", "playbook", "checklist", "setup criteria"]
    analysis_keywords = ["analysis", "bias", "outlook", "scenario", "session plan"]
    education_keywords = ["definition", "example", "lesson", "tutorial", "explained"]

    for keyword in chart_keywords:
        if keyword in lowered:
            scores["chart"] += 2
    for keyword in journal_keywords:
        if keyword in lowered:
            scores["trade_journal"] += 2
    for keyword in performance_keywords:
        if keyword in lowered:
            scores["performance_report"] += 2
    for keyword in strategy_keywords:
        if keyword in lowered:
            scores["strategy_note"] += 2
    for keyword in analysis_keywords:
        if keyword in lowered:
            scores["market_analysis"] += 2
    for keyword in education_keywords:
        if keyword in lowered:
            scores["educational_note"] += 2

    if symbols or timeframes:
        scores["chart"] += 1
        scores["market_analysis"] += 1
    if "page" in lowered and len(text.split()) < 20:
        scores["chart"] += 1

    best_page_type = max(scores, key=scores.get)
    best_score = scores[best_page_type]
    if best_score == 0:
        return "unknown", 0.0

    total_score = sum(scores.values()) or 1
    confidence = round(best_score / total_score, 2)
    return best_page_type, confidence


def build_auto_summary(
    page_type: str,
    symbols: list[str],
    timeframes: list[str],
    concepts: list[str],
    text: str,
) -> str:
    parts: list[str] = []
    if page_type != "unknown":
        parts.append(page_type.replace("_", " "))
    if symbols:
        parts.append("symbols: " + ", ".join(symbols[:4]))
    if timeframes:
        parts.append("timeframes: " + ", ".join(timeframes[:4]))
    if concepts:
        parts.append("concepts: " + ", ".join(concepts[:4]))
    excerpt = sentence_excerpt(text, max_length=180)
    if excerpt:
        parts.append(excerpt)
    summary = " | ".join(parts)
    if len(summary) > SUMMARY_CHAR_LIMIT:
        summary = summary[: SUMMARY_CHAR_LIMIT - 3].rstrip() + "..."
    return summary


def build_page_labels(
    page_type: str,
    combined_text: str,
    embedded_text: str,
    ocr_text: str,
    symbols: list[str],
    timeframes: list[str],
) -> dict[str, Any]:
    lowered = combined_text.lower()
    return {
        "has_embedded_text": bool(embedded_text),
        "has_ocr_text": bool(ocr_text),
        "text_density": text_density_bucket(combined_text),
        "likely_chart": page_type == "chart",
        "contains_trade_levels": any(keyword in lowered for keyword in ["entry", "stop", "tp", "target"]),
        "contains_performance_metrics": any(
            keyword in lowered for keyword in ["pnl", "drawdown", "win rate", "profit factor", "sharpe"]
        ),
        "contains_strategy_rules": any(
            keyword in lowered for keyword in ["strategy", "rules", "checklist", "criteria"]
        ),
        "contains_symbol": bool(symbols),
        "contains_timeframe": bool(timeframes),
    }


def build_page_training_target(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_type": annotation["page_type"],
        "summary": annotation["summary"],
        "clean_text": annotation["combined_text"],
        "symbols": annotation["symbols"],
        "timeframes": annotation["timeframes"],
        "trading_concepts": annotation["trading_concepts"],
        "trading_domains": annotation["trading_domains"],
        "labels": annotation["labels"],
    }


def merge_page_market_fields(
    symbols: list[str],
    timeframes: list[str],
    asset_annotations: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    merged_symbols = unique_preserving_order(
        [*symbols, *(symbol for annotation in asset_annotations for symbol in annotation.get("symbols", []))]
    )
    merged_timeframes = unique_preserving_order(
        [*timeframes, *(timeframe for annotation in asset_annotations for timeframe in annotation.get("timeframes", []))]
    )
    return merged_symbols, merged_timeframes


def to_bbox(rect: Any) -> BBox:
    if isinstance(rect, (list, tuple)):
        x0, y0, x1, y1 = rect[:4]
    else:
        x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
    return (float(x0), float(y0), float(x1), float(y1))


def bbox_width(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0])


def bbox_height(bbox: BBox) -> float:
    return max(0.0, bbox[3] - bbox[1])


def bbox_area(bbox: BBox) -> float:
    return bbox_width(bbox) * bbox_height(bbox)


def bbox_sort_key(bbox: BBox) -> tuple[float, float]:
    return (bbox[1], bbox[0])


def bbox_intersection(bbox_a: BBox, bbox_b: BBox) -> BBox | None:
    x0 = max(bbox_a[0], bbox_b[0])
    y0 = max(bbox_a[1], bbox_b[1])
    x1 = min(bbox_a[2], bbox_b[2])
    y1 = min(bbox_a[3], bbox_b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def bbox_intersects(bbox_a: BBox, bbox_b: BBox) -> bool:
    return bbox_intersection(bbox_a, bbox_b) is not None


def bbox_expand(bbox: BBox, padding: float, page_bbox: BBox) -> BBox:
    return (
        max(page_bbox[0], bbox[0] - padding),
        max(page_bbox[1], bbox[1] - padding),
        min(page_bbox[2], bbox[2] + padding),
        min(page_bbox[3], bbox[3] + padding),
    )


def bbox_union(bbox_a: BBox, bbox_b: BBox) -> BBox:
    return (
        min(bbox_a[0], bbox_b[0]),
        min(bbox_a[1], bbox_b[1]),
        max(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    )


def bbox_overlap_ratio(bbox_a: BBox, bbox_b: BBox) -> float:
    overlap = bbox_intersection(bbox_a, bbox_b)
    if overlap is None:
        return 0.0
    return bbox_area(overlap) / max(min(bbox_area(bbox_a), bbox_area(bbox_b)), 1.0)


def bbox_axis_overlap_ratio(bbox_a: BBox, bbox_b: BBox, axis: str) -> float:
    if axis == "x":
        start = max(bbox_a[0], bbox_b[0])
        end = min(bbox_a[2], bbox_b[2])
        length = max(0.0, end - start)
        base = max(min(bbox_width(bbox_a), bbox_width(bbox_b)), 1.0)
        return length / base

    start = max(bbox_a[1], bbox_b[1])
    end = min(bbox_a[3], bbox_b[3])
    length = max(0.0, end - start)
    base = max(min(bbox_height(bbox_a), bbox_height(bbox_b)), 1.0)
    return length / base


def bbox_vertical_gap(bbox_a: BBox, bbox_b: BBox) -> float:
    if bbox_a[3] < bbox_b[1]:
        return bbox_b[1] - bbox_a[3]
    if bbox_b[3] < bbox_a[1]:
        return bbox_a[1] - bbox_b[3]
    return 0.0


def bbox_horizontal_gap(bbox_a: BBox, bbox_b: BBox) -> float:
    if bbox_a[2] < bbox_b[0]:
        return bbox_b[0] - bbox_a[2]
    if bbox_b[2] < bbox_a[0]:
        return bbox_a[0] - bbox_b[2]
    return 0.0


def bbox_contains(outer_bbox: BBox, inner_bbox: BBox, threshold: float = 0.95) -> bool:
    overlap = bbox_intersection(outer_bbox, inner_bbox)
    if overlap is None:
        return False
    return bbox_area(overlap) / max(bbox_area(inner_bbox), 1.0) >= threshold


def bbox_to_json(bbox: BBox) -> dict[str, float]:
    return {
        "x0": round(bbox[0], 2),
        "y0": round(bbox[1], 2),
        "x1": round(bbox[2], 2),
        "y1": round(bbox[3], 2),
    }


def bbox_ratio_to_json(bbox: BBox, page_bbox: BBox) -> dict[str, float]:
    page_width = max(bbox_width(page_bbox), 1.0)
    page_height = max(bbox_height(page_bbox), 1.0)
    return {
        "x0": round((bbox[0] - page_bbox[0]) / page_width, 4),
        "y0": round((bbox[1] - page_bbox[1]) / page_height, 4),
        "x1": round((bbox[2] - page_bbox[0]) / page_width, 4),
        "y1": round((bbox[3] - page_bbox[1]) / page_height, 4),
    }


def is_visual_candidate(bbox: BBox, page_bbox: BBox) -> bool:
    page_area = max(bbox_area(page_bbox), 1.0)
    width_ratio = bbox_width(bbox) / max(bbox_width(page_bbox), 1.0)
    height_ratio = bbox_height(bbox) / max(bbox_height(page_bbox), 1.0)
    area_ratio = bbox_area(bbox) / page_area
    return (
        width_ratio >= ASSET_MIN_WIDTH_RATIO
        and height_ratio >= ASSET_MIN_HEIGHT_RATIO
        and area_ratio >= ASSET_MIN_AREA_RATIO
    )


def normalize_context_line(line: str) -> str:
    line = repair_common_mojibake(line)
    line = line.replace("\x00", " ").replace("\u00a0", " ")
    line = re.sub(r"^[•*\-–—]+\s*", "", line.strip())
    line = re.sub(r"\s+", " ", line)
    return line.strip(" |")


def is_noise_line(line: str) -> bool:
    lowered = line.lower().strip()
    if not lowered:
        return True
    if re.fullmatch(r"(page\s*)?\d+(\s*(/|of)\s*\d+)?", lowered):
        return True
    if re.fullmatch(r"[-_=|. ]{3,}", lowered):
        return True
    if len(lowered) == 1 and not lowered.isalnum():
        return True
    return False


def polish_context_lines(text: str, max_lines: int | None = None) -> list[str]:
    if not text:
        return []

    prepared = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    raw_lines = prepared.splitlines() if "\n" in prepared else re.split(r"(?<=[.!?])\s+", prepared)
    cleaned_lines: list[str] = []
    for raw_line in raw_lines:
        line = normalize_context_line(raw_line)
        if is_noise_line(line):
            continue
        cleaned_lines.append(line)

    deduplicated = unique_lines(cleaned_lines)
    if max_lines is not None:
        return deduplicated[:max_lines]
    return deduplicated


def build_context_text(text_blocks: list[LayoutTextBlock]) -> str:
    lines: list[str] = []
    for text_block in text_blocks:
        lines.extend(polish_context_lines(text_block.text))
    return "\n".join(unique_lines(lines)[:10])


def infer_context_heading(text_blocks: list[LayoutTextBlock]) -> str:
    prioritized_blocks = sorted(text_blocks, key=lambda item: (-item.font_size, *bbox_sort_key(item.bbox)))
    for text_block in prioritized_blocks:
        lines = meaningful_text_lines(text_block.text)[:2]
        for line in lines:
            if 4 <= len(line) <= 110:
                return line

    fallback_lines = polish_context_lines("\n".join(text_block.text for text_block in text_blocks), max_lines=1)
    return fallback_lines[0] if fallback_lines else ""


def extract_page_layout(page: Any) -> tuple[list[LayoutTextBlock], list[LayoutImageBlock]]:
    payload = page.get_text("dict")
    text_blocks: list[LayoutTextBlock] = []
    image_blocks: list[LayoutImageBlock] = []

    for block in payload.get("blocks", []):
        bbox = block.get("bbox")
        if not bbox:
            continue

        if block.get("type") == 0:
            lines: list[str] = []
            max_font_size = 0.0
            for line in block.get("lines", []):
                span_parts: list[str] = []
                for span in line.get("spans", []):
                    span_text = str(span.get("text", ""))
                    if not span_text.strip():
                        continue
                    span_parts.append(span_text)
                    max_font_size = max(max_font_size, float(span.get("size") or 0.0))
                if span_parts:
                    lines.append("".join(span_parts))

            text = normalize_whitespace("\n".join(lines))
            if text:
                text_blocks.append(LayoutTextBlock(bbox=to_bbox(bbox), text=text, font_size=max_font_size))
        elif block.get("type") == 1:
            image_blocks.append(
                LayoutImageBlock(
                    bbox=to_bbox(bbox),
                    ext=str(block.get("ext") or ""),
                    image_bytes=block.get("image") or b"",
                    width=int(block.get("width") or 0),
                    height=int(block.get("height") or 0),
                    xres=int(block.get("xres") or 0),
                    yres=int(block.get("yres") or 0),
                )
            )

    text_blocks.sort(key=lambda item: bbox_sort_key(item.bbox))
    image_blocks.sort(key=lambda item: bbox_sort_key(item.bbox))
    return text_blocks, image_blocks


def merge_bboxes(bboxes: list[BBox], gap: float, page_bbox: BBox) -> list[BBox]:
    merged: list[BBox] = []
    for bbox in sorted(bboxes, key=bbox_sort_key):
        candidate = bbox
        for index, existing in enumerate(merged):
            expanded_existing = bbox_expand(existing, gap, page_bbox)
            expanded_candidate = bbox_expand(candidate, gap, page_bbox)
            if bbox_intersects(expanded_existing, expanded_candidate):
                merged[index] = bbox_union(existing, candidate)
                break
        else:
            merged.append(candidate)

    changed = True
    while changed:
        changed = False
        next_pass: list[BBox] = []
        for bbox in merged:
            for index, existing in enumerate(next_pass):
                expanded_existing = bbox_expand(existing, gap, page_bbox)
                expanded_candidate = bbox_expand(bbox, gap, page_bbox)
                if bbox_intersects(expanded_existing, expanded_candidate):
                    next_pass[index] = bbox_union(existing, bbox)
                    changed = True
                    break
            else:
                next_pass.append(bbox)
        merged = next_pass

    return sorted(merged, key=bbox_sort_key)


def extract_drawing_bboxes(page: Any, page_bbox: BBox) -> list[BBox]:
    get_drawings = getattr(page, "get_drawings", None)
    if get_drawings is None:
        return []

    raw_bboxes: list[BBox] = []
    page_area = max(bbox_area(page_bbox), 1.0)
    page_width = max(bbox_width(page_bbox), 1.0)
    for drawing in get_drawings():
        rect = drawing.get("rect")
        if not rect:
            continue
        bbox = to_bbox(rect)
        if bbox_area(bbox) <= 0:
            continue

        area_ratio = bbox_area(bbox) / page_area
        width_ratio = bbox_width(bbox) / page_width
        if area_ratio < max(ASSET_MIN_AREA_RATIO * 1.5, 0.03):
            continue
        if width_ratio < max(ASSET_MIN_WIDTH_RATIO, 0.25):
            continue
        raw_bboxes.append(bbox)

    return merge_bboxes(raw_bboxes, DRAWING_CLUSTER_GAP, page_bbox)


def infer_gap_visual_bboxes(text_blocks: list[LayoutTextBlock], page_bbox: BBox) -> list[BBox]:
    if not text_blocks:
        return []

    page_height = max(bbox_height(page_bbox), 1.0)
    horizontal_margin = min(ASSET_CONTEXT_MARGIN / 2, bbox_width(page_bbox) * 0.03)
    vertical_padding = min(ASSET_CONTEXT_MARGIN / 2, page_height * 0.03)

    occupied_intervals: list[tuple[float, float]] = []
    for text_block in text_blocks:
        start = max(page_bbox[1], text_block.bbox[1] - vertical_padding)
        end = min(page_bbox[3], text_block.bbox[3] + vertical_padding)
        occupied_intervals.append((start, end))

    occupied_intervals.sort()
    merged_intervals: list[tuple[float, float]] = []
    for start, end in occupied_intervals:
        if merged_intervals and start <= merged_intervals[-1][1]:
            previous_start, previous_end = merged_intervals[-1]
            merged_intervals[-1] = (previous_start, max(previous_end, end))
        else:
            merged_intervals.append((start, end))

    gap_bboxes: list[BBox] = []
    cursor = page_bbox[1]
    min_gap_height = page_height * TEXT_GAP_MIN_RATIO
    for start, end in merged_intervals:
        if start - cursor >= min_gap_height:
            gap_bboxes.append(
                (
                    page_bbox[0] + horizontal_margin,
                    cursor,
                    page_bbox[2] - horizontal_margin,
                    start,
                )
            )
        cursor = max(cursor, end)

    if page_bbox[3] - cursor >= min_gap_height:
        gap_bboxes.append(
            (
                page_bbox[0] + horizontal_margin,
                cursor,
                page_bbox[2] - horizontal_margin,
                page_bbox[3] - vertical_padding,
            )
        )

    return gap_bboxes


def merge_visual_asset_candidates(
    candidates: list[VisualAssetCandidate],
    page_bbox: BBox,
) -> list[VisualAssetCandidate]:
    merged: list[VisualAssetCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-bbox_area(item.bbox), *bbox_sort_key(item.bbox))):
        for index, existing in enumerate(merged):
            if (
                bbox_overlap_ratio(existing.bbox, candidate.bbox) >= 0.75
                or bbox_contains(existing.bbox, candidate.bbox, 0.9)
                or bbox_contains(candidate.bbox, existing.bbox, 0.9)
            ):
                merged_bbox = bbox_union(existing.bbox, candidate.bbox)
                merged_source_parts = sorted(set((existing.source + "+" + candidate.source).split("+")))
                merged[index] = VisualAssetCandidate(
                    bbox=bbox_expand(merged_bbox, 0, page_bbox),
                    source="+".join(merged_source_parts),
                    image_block=None,
                )
                break
        else:
            merged.append(candidate)

    return sorted(merged, key=lambda item: bbox_sort_key(item.bbox))


def discover_visual_asset_candidates(
    page: Any,
    text_blocks: list[LayoutTextBlock],
    image_blocks: list[LayoutImageBlock],
    page_bbox: BBox,
) -> list[VisualAssetCandidate]:
    candidates: list[VisualAssetCandidate] = []
    for image_block in image_blocks:
        if is_visual_candidate(image_block.bbox, page_bbox):
            candidates.append(
                VisualAssetCandidate(
                    bbox=image_block.bbox,
                    source="embedded_image",
                    image_block=image_block,
                )
            )

    for bbox in extract_drawing_bboxes(page, page_bbox):
        if is_visual_candidate(bbox, page_bbox):
            candidates.append(VisualAssetCandidate(bbox=bbox, source="vector_drawing"))

    candidates = merge_visual_asset_candidates(candidates, page_bbox)
    if candidates:
        return candidates

    fallback_bboxes = infer_gap_visual_bboxes(text_blocks, page_bbox)
    for bbox in fallback_bboxes:
        if is_visual_candidate(bbox, page_bbox):
            candidates.append(VisualAssetCandidate(bbox=bbox, source="text_gap"))
    return merge_visual_asset_candidates(candidates, page_bbox)


def score_text_block_for_asset(text_block: LayoutTextBlock, asset_bbox: BBox, page_bbox: BBox) -> float:
    context_window = bbox_expand(asset_bbox, ASSET_CONTEXT_MARGIN, page_bbox)
    max_vertical_gap = max(ASSET_CONTEXT_MARGIN * 2, bbox_height(page_bbox) * 0.18)
    x_overlap = bbox_axis_overlap_ratio(asset_bbox, text_block.bbox, axis="x")
    vertical_gap = bbox_vertical_gap(asset_bbox, text_block.bbox)
    horizontal_gap = bbox_horizontal_gap(asset_bbox, text_block.bbox)
    same_column = x_overlap >= 0.35
    close_to_asset = same_column and vertical_gap <= max_vertical_gap
    inside_window = bbox_intersects(context_window, text_block.bbox)
    if not inside_window and not close_to_asset:
        return float("-inf")

    score = x_overlap * 2.5
    score += 1.0 / (1.0 + (vertical_gap / 72.0))
    if text_block.bbox[3] <= asset_bbox[1]:
        score += 0.2
    if text_block.font_size >= 14 and text_block.bbox[3] <= asset_bbox[1]:
        score += 0.4
    score -= min(horizontal_gap / max(bbox_width(page_bbox), 1.0), 1.0)
    return score


def asset_text_block_preference(text_block: LayoutTextBlock, asset_bbox: BBox) -> int:
    block_center_y = (text_block.bbox[1] + text_block.bbox[3]) / 2
    asset_center_y = (asset_bbox[1] + asset_bbox[3]) / 2
    return 1 if block_center_y <= asset_center_y else 0


def select_asset_text_blocks(
    text_blocks: list[LayoutTextBlock],
    asset_bbox: BBox,
    page_bbox: BBox,
    sibling_asset_bboxes: list[BBox] | None = None,
) -> list[LayoutTextBlock]:
    scored_blocks: list[tuple[float, LayoutTextBlock]] = []
    sibling_asset_bboxes = sibling_asset_bboxes or []

    for text_block in text_blocks:
        current_score = score_text_block_for_asset(text_block, asset_bbox, page_bbox)
        if current_score == float("-inf"):
            continue

        competing_scores = [
            score_text_block_for_asset(text_block, sibling_bbox, page_bbox)
            for sibling_bbox in sibling_asset_bboxes
        ]
        best_competing_score = max(competing_scores, default=float("-inf"))
        if best_competing_score > current_score:
            continue
        if sibling_asset_bboxes and abs(best_competing_score - current_score) <= 1e-6:
            current_preference = asset_text_block_preference(text_block, asset_bbox)
            competing_preference = max(
                (
                    asset_text_block_preference(text_block, sibling_bbox)
                    for sibling_bbox, sibling_score in zip(sibling_asset_bboxes, competing_scores)
                    if abs(sibling_score - best_competing_score) <= 1e-6
                ),
                default=0,
            )
            if competing_preference > current_preference:
                continue

        scored_blocks.append((current_score, text_block))

    if not scored_blocks:
        fallback_blocks = [
            text_block
            for text_block in text_blocks
            if bbox_axis_overlap_ratio(asset_bbox, text_block.bbox, axis="x") >= 0.2
        ]
        fallback_blocks = sorted(
            fallback_blocks,
            key=lambda item: (bbox_vertical_gap(asset_bbox, item.bbox), bbox_horizontal_gap(asset_bbox, item.bbox)),
        )
        return sorted(fallback_blocks[:2], key=lambda item: bbox_sort_key(item.bbox))

    chosen: list[LayoutTextBlock] = []
    seen_texts: set[str] = set()
    for _score, text_block in sorted(scored_blocks, key=lambda item: item[0], reverse=True):
        fingerprint = text_block.text.lower()
        if fingerprint in seen_texts:
            continue
        chosen.append(text_block)
        seen_texts.add(fingerprint)
        if len(chosen) >= ASSET_MAX_CONTEXT_BLOCKS:
            break

    return sorted(chosen, key=lambda item: bbox_sort_key(item.bbox))


def extract_bias(text: str) -> str | None:
    lowered = ascii_fold(text).lower()
    has_bullish = bool(re.search(r"\b(bullish|bullisch)\b", lowered))
    has_bearish = bool(re.search(r"\b(bearish|bearisch)\b", lowered))
    if has_bullish and not has_bearish:
        return "bullish"
    if has_bearish and not has_bullish:
        return "bearish"
    return None


def extract_direction(text: str) -> str | None:
    lowered = ascii_fold(text).lower()
    has_long = bool(re.search(r"\b(long|buy|kauf)\b", lowered))
    has_short = bool(re.search(r"\b(short|sell|verkauf)\b", lowered))
    if has_long and not has_short:
        return "long"
    if has_short and not has_long:
        return "short"
    return None


def extract_setup_status(text: str) -> str | None:
    lowered = ascii_fold(text).lower()
    no_setup_patterns = [
        r"\bkein\s+setup\b",
        r"\bkeine[sr]?\s+setup\b",
        r"\bno\s+setup\b",
        r"\bwithout\s+setup\b",
        r"\bohne\s+setup\b",
        r"\bkein\s+signal\b",
    ]
    if any(re.search(pattern, lowered) for pattern in no_setup_patterns):
        return "no_setup"
    if re.search(r"\bsetup\b", lowered):
        return "setup_mentioned"
    return None


def display_name_for_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    return SYMBOL_DISPLAY_NAMES.get(symbol, symbol)


def extract_trade_levels(text: str) -> dict[str, str]:
    patterns = {
        "entry": [
            r"\bentry(?:\s*price)?\b[:\s-]*([0-9]+(?:[.,][0-9]+)*)",
        ],
        "stop_loss": [
            r"\bstop(?:\s*loss)?\b[:\s-]*([0-9]+(?:[.,][0-9]+)*)",
            r"\bsl\b[:\s-]*([0-9]+(?:[.,][0-9]+)*)",
        ],
        "take_profit": [
            r"\btake\s*profit\b[:\s-]*([0-9]+(?:[.,][0-9]+)*)",
            r"\btp\b[:\s-]*([0-9]+(?:[.,][0-9]+)*)",
            r"\btarget\b[:\s-]*([0-9]+(?:[.,][0-9]+)*)",
        ],
    }
    extracted: dict[str, str] = {}
    for key, key_patterns in patterns.items():
        for pattern in key_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                extracted[key] = match.group(1)
                break
    return extracted


def infer_asset_type(
    context_text: str,
    ocr_text: str,
    symbols: list[str],
    timeframes: list[str],
    trading_concepts: list[str],
    trading_domains: list[str],
) -> str:
    combined = f"{context_text}\n{ocr_text}".lower()
    if (
        symbols
        or timeframes
        or trading_concepts
        or "technical_analysis" in trading_domains
        or any(keyword in combined for keyword in ["chart", "candlestick", "trendline", "entry", "stop loss", "take profit"])
    ):
        return "chart"
    if "performance_review" in trading_domains or any(
        keyword in combined for keyword in ["equity curve", "drawdown", "win rate", "profit factor", "sharpe", "pnl"]
    ):
        return "performance_panel"
    if any(keyword in combined for keyword in ["strategy", "rules", "checklist", "playbook"]):
        return "strategy_figure"
    if context_text or ocr_text:
        return "figure"
    return "unknown"


def build_asset_summary(
    asset_type: str,
    page_type: str,
    primary_symbol: str | None,
    venue: str | None,
    symbols: list[str],
    timeframes: list[str],
    concepts: list[str],
    bias: str | None,
    setup_status: str | None,
    text: str,
) -> str:
    parts: list[str] = []
    if primary_symbol:
        primary_part = primary_symbol
        if timeframes:
            primary_part += " " + "/".join(timeframes[:2])
        primary_part += f" {asset_type.replace('_', ' ')}"
        parts.append(primary_part)
    elif timeframes:
        parts.append("/".join(timeframes[:2]) + f" {asset_type.replace('_', ' ')}")
    elif asset_type != "unknown":
        parts.append(asset_type.replace("_", " "))
    elif page_type != "unknown":
        parts.append(page_type.replace("_", " "))

    if bias:
        parts.append(bias + " bias")
    if setup_status == "no_setup":
        parts.append("no setup")
    if venue:
        parts.append(venue)
    if concepts:
        parts.append("concepts: " + ", ".join(concepts[:3]))
    focus_line = select_focus_line(text)
    excerpt = focus_line or sentence_excerpt(text, max_length=180)
    if excerpt:
        parts.append(excerpt)
    summary = " | ".join(parts)
    if len(summary) > SUMMARY_CHAR_LIMIT:
        summary = summary[: SUMMARY_CHAR_LIMIT - 3].rstrip() + "..."
    return summary


def build_asset_description(
    asset_type: str,
    page_type: str,
    primary_symbol: str | None,
    instrument_name: str | None,
    venue: str | None,
    symbols: list[str],
    timeframes: list[str],
    concepts: list[str],
    context_text: str,
    ocr_text: str,
    bias: str | None,
    direction: str | None,
    setup_status: str | None,
    trade_levels: dict[str, str],
) -> str:
    if asset_type == "chart":
        if primary_symbol and timeframes:
            opening = f"Trading chart for {primary_symbol} on {', '.join(timeframes[:2])}."
        elif primary_symbol:
            opening = f"Trading chart for {primary_symbol}."
        else:
            opening = "Trading chart."
    elif asset_type != "unknown":
        opening = f"{asset_type.replace('_', ' ').capitalize()}."
    elif page_type != "unknown":
        opening = f"{page_type.replace('_', ' ').capitalize()} visual."
    else:
        opening = "Trading-related visual."

    sentences = [opening]
    if instrument_name and primary_symbol and instrument_name != primary_symbol:
        sentences.append(f"Instrument label: {instrument_name}.")
    if venue:
        sentences.append(f"Venue: {venue}.")
    if bias:
        sentences.append(f"Market bias: {bias}.")
    if direction:
        sentences.append(f"Explicit direction: {direction}.")
    if setup_status == "no_setup":
        sentences.append("Context states that no setup was present.")
    elif setup_status == "setup_mentioned":
        sentences.append("A setup is mentioned in the surrounding context.")
    if concepts:
        sentences.append("Key concepts: " + ", ".join(concept.replace("_", " ") for concept in concepts[:4]) + ".")
    if trade_levels:
        level_summary = ", ".join(
            f"{field_name.replace('_', ' ')} {field_value}" for field_name, field_value in trade_levels.items()
        )
        sentences.append(f"Mentioned levels: {level_summary}.")

    excerpt_source = context_text or ocr_text
    excerpt = select_focus_line(excerpt_source) or sentence_excerpt(excerpt_source, max_length=200)
    if excerpt:
        sentences.append(excerpt)
    elif ocr_text:
        visible_text = select_focus_line(ocr_text) or sentence_excerpt(ocr_text, max_length=140)
        if visible_text:
            sentences.append(f"Visible text: {visible_text}")

    return " ".join(sentences[:5])


def build_asset_clean_text(
    asset_type: str,
    primary_symbol: str | None,
    instrument_name: str | None,
    venue: str | None,
    timeframes: list[str],
    bias: str | None,
    direction: str | None,
    setup_status: str | None,
    context_text: str,
    ocr_text: str,
    trade_levels: dict[str, str],
) -> str:
    lines: list[str] = [f"Asset type: {asset_type}"]
    if primary_symbol:
        lines.append(f"Instrument: {primary_symbol}")
    if instrument_name and instrument_name != primary_symbol:
        lines.append(f"Instrument label: {instrument_name}")
    if venue:
        lines.append(f"Venue: {venue}")
    if timeframes:
        lines.append("Timeframes: " + ", ".join(timeframes))
    if bias:
        lines.append(f"Bias: {bias}")
    if direction:
        lines.append(f"Direction: {direction}")
    if setup_status:
        lines.append("Setup status: " + setup_status.replace("_", " "))
    if trade_levels:
        level_summary = ", ".join(
            f"{field_name.replace('_', ' ')} {field_value}" for field_name, field_value in trade_levels.items()
        )
        lines.append("Levels: " + level_summary)

    focus_context = select_focus_line(context_text)
    if focus_context:
        lines.append("Context note: " + focus_context)

    context_fingerprint = focus_context.lower() if focus_context else None
    visible_lines = [
        line
        for line in select_visible_label_lines(ocr_text)
        if context_fingerprint is None or line.lower() != context_fingerprint
    ]
    if visible_lines:
        lines.append("Visible labels: " + "; ".join(visible_lines[:3]))

    return "\n".join(lines)


def build_asset_labels(
    asset_type: str,
    page_type: str,
    combined_text: str,
    context_text: str,
    ocr_text: str,
    symbols: list[str],
    timeframes: list[str],
    asset_bbox: BBox,
    page_bbox: BBox,
    paired_text_blocks: int,
) -> dict[str, Any]:
    lowered = combined_text.lower()
    return {
        "has_context_text": bool(context_text),
        "has_ocr_text": bool(ocr_text),
        "text_density": text_density_bucket(combined_text),
        "likely_chart": asset_type == "chart" or page_type == "chart",
        "contains_trade_levels": any(keyword in lowered for keyword in ["entry", "stop", "tp", "target"]),
        "contains_performance_metrics": any(
            keyword in lowered for keyword in ["pnl", "drawdown", "win rate", "profit factor", "sharpe"]
        ),
        "contains_strategy_rules": any(keyword in lowered for keyword in ["strategy", "rules", "checklist", "criteria"]),
        "contains_symbol": bool(symbols),
        "contains_timeframe": bool(timeframes),
        "is_large_visual": (bbox_area(asset_bbox) / max(bbox_area(page_bbox), 1.0)) >= 0.2,
        "paired_text_blocks": paired_text_blocks,
    }


def text_contains_value(text: str, value: str) -> bool:
    normalized_text = normalize_token(text)
    normalized_value = normalize_token(value)
    return bool(normalized_text and normalized_value and normalized_value in normalized_text)


def infer_scalar_text_source(
    value: str | None,
    context_text: str,
    ocr_text: str,
    fallback: str = "heuristic",
) -> str:
    if not value:
        return "missing"
    in_context = text_contains_value(context_text, value)
    in_ocr = text_contains_value(ocr_text, value)
    if in_context and in_ocr:
        return "context+ocr"
    if in_context:
        return "context"
    if in_ocr:
        return "ocr"
    return fallback


def infer_list_text_source(
    values: list[str],
    context_text: str,
    ocr_text: str,
    fallback: str = "heuristic",
) -> str:
    if not values:
        return "missing"
    in_context = any(text_contains_value(context_text, value) for value in values)
    in_ocr = any(text_contains_value(ocr_text, value) for value in values)
    if in_context and in_ocr:
        return "context+ocr"
    if in_context:
        return "context"
    if in_ocr:
        return "ocr"
    return fallback


def infer_dict_text_source(
    values: dict[str, str],
    context_text: str,
    ocr_text: str,
    fallback: str = "regex_extraction",
) -> str:
    if not values:
        return "missing"
    raw_values = list(values.values())
    in_context = any(text_contains_value(context_text, value) for value in raw_values)
    in_ocr = any(text_contains_value(ocr_text, value) for value in raw_values)
    if in_context and in_ocr:
        return "regex_from_context+ocr"
    if in_context:
        return "regex_from_context"
    if in_ocr:
        return "regex_from_ocr"
    return fallback


def source_uses_ocr(source_name: str) -> bool:
    return "ocr" in source_name


def promote_source_with_ocr_normalization(source_name: str, has_visible_signal: bool, ocr_text: str) -> str:
    if not has_visible_signal or not ocr_text or source_name == "missing":
        return source_name
    if source_uses_ocr(source_name):
        return source_name
    if source_name == "context":
        return "context+ocr_normalization"
    if source_name.startswith("heuristic"):
        return "ocr_normalization"
    return source_name


def filter_values_by_detected_visibility(values: list[str], detected_values: Iterable[str]) -> list[str]:
    visible_lookup = {value for value in detected_values if value}
    return [value for value in values if value in visible_lookup]


def filter_trade_levels_by_visibility(trade_levels: dict[str, str], ocr_text: str) -> dict[str, str]:
    if not trade_levels or not ocr_text:
        return {}
    return {
        field_name: field_value
        for field_name, field_value in trade_levels.items()
        if text_contains_value(ocr_text, field_value)
    }


def symbol_is_visible_in_ocr(symbol: str, ocr_text: str) -> bool:
    if not symbol or not ocr_text:
        return False
    if text_contains_value(ocr_text, symbol):
        return True
    if symbol in extract_symbols(ocr_text):
        return True

    display_name = display_name_for_symbol(symbol)
    return bool(display_name and text_contains_value(ocr_text, display_name))


def filter_visible_symbols(symbols: list[str], ocr_text: str) -> list[str]:
    return [symbol for symbol in symbols if symbol_is_visible_in_ocr(symbol, ocr_text)]


def timeframe_is_visible_in_ocr(timeframe: str, ocr_text: str) -> bool:
    if not timeframe or not ocr_text:
        return False

    normalized_text = ascii_fold(ocr_text).lower()
    aliases = [timeframe.lower()]
    aliases.extend(raw_timeframe for raw_timeframe, canonical in TIMEFRAME_MAP.items() if canonical == timeframe)

    seen_aliases: set[str] = set()
    for alias in aliases:
        if not alias or alias in seen_aliases:
            continue
        seen_aliases.add(alias)
        if re.search(rf"\b{re.escape(alias)}\b", normalized_text):
            return True

    if timeframe.startswith("M") and timeframe[1:].isdigit():
        minute_value = timeframe[1:]
        venue_pattern = "|".join(venue.lower() for venue in sorted(KNOWN_VENUES))
        if re.search(rf"\b{re.escape(minute_value)}\b(?=\s*[,/|-]?\s*(?:{venue_pattern})\b)", normalized_text):
            return True

    return False


def filter_visible_timeframes(timeframes: list[str], ocr_text: str) -> list[str]:
    return [timeframe for timeframe in timeframes if timeframe_is_visible_in_ocr(timeframe, ocr_text)]


def resolve_visible_crop_fields(annotation: dict[str, Any], field_sources: dict[str, str]) -> dict[str, Any]:
    ocr_text = annotation["ocr_text"]
    detected_venues = extract_venues(ocr_text)

    visible_instrument_name = (
        annotation["instrument_name"]
        if annotation["instrument_name"] and source_uses_ocr(field_sources["instrument_name"])
        else None
    )

    visible_primary_symbol = None
    if annotation["primary_symbol"] and source_uses_ocr(field_sources["primary_symbol"]):
        if symbol_is_visible_in_ocr(annotation["primary_symbol"], ocr_text) or visible_instrument_name:
            visible_primary_symbol = annotation["primary_symbol"]

    visible_symbol_candidates = set()
    if source_uses_ocr(field_sources["symbols"]):
        visible_symbol_candidates.update(filter_visible_symbols(annotation["symbols"], ocr_text))
        if visible_primary_symbol and visible_primary_symbol in annotation["symbols"]:
            visible_symbol_candidates.add(visible_primary_symbol)
    visible_symbols = [value for value in annotation["symbols"] if value in visible_symbol_candidates]

    visible_venue = (
        annotation["venue"]
        if annotation["venue"] and source_uses_ocr(field_sources["venue"]) and annotation["venue"] in detected_venues
        else None
    )
    visible_timeframes = (
        filter_visible_timeframes(annotation["timeframes"], ocr_text)
        if source_uses_ocr(field_sources["timeframes"])
        else []
    )
    visible_trade_levels = (
        filter_trade_levels_by_visibility(annotation["trade_levels"], ocr_text)
        if source_uses_ocr(field_sources["trade_levels"])
        else {}
    )

    return {
        "primary_symbol": visible_primary_symbol,
        "instrument_name": visible_instrument_name,
        "venue": visible_venue,
        "symbols": visible_symbols,
        "timeframes": visible_timeframes,
        "trade_levels": visible_trade_levels,
    }


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


def has_ocr_encoding_artifacts(text: str) -> bool:
    return any(marker in text for marker in ("\ufffd", "Ã", "Â", "â€", "ï¿½"))


def build_visual_element_tags(annotation: dict[str, Any]) -> list[str]:
    labels = annotation["labels"]
    elements: list[str] = []

    asset_type = annotation["asset_type"]
    if asset_type == "chart":
        elements.append("chart_panel")
    elif asset_type == "performance_panel":
        elements.append("performance_panel")
    elif asset_type == "strategy_figure":
        elements.append("strategy_panel")
    elif asset_type == "figure":
        elements.append("generic_figure")
    else:
        elements.append("unknown_visual")

    if labels.get("contains_symbol"):
        elements.append("symbol_label")
    if labels.get("contains_timeframe"):
        elements.append("timeframe_label")
    if annotation.get("venue"):
        elements.append("venue_label")
    if annotation.get("ocr_text"):
        elements.append("ocr_text_overlay")
    if annotation.get("trade_levels"):
        elements.append("price_level_text")
    if labels.get("likely_chart"):
        elements.append("price_axis_or_scale")
    if labels.get("contains_performance_metrics"):
        elements.append("performance_metric_text")
    if labels.get("contains_strategy_rules"):
        elements.append("strategy_rule_text")

    return unique_preserving_order(elements)


def build_crop_clean_text(
    asset_type: str,
    primary_symbol: str | None,
    instrument_name: str | None,
    venue: str | None,
    timeframes: list[str],
    ocr_text: str,
    trade_levels: dict[str, str],
) -> str:
    lines: list[str] = [f"Asset type: {asset_type}"]
    if primary_symbol:
        lines.append(f"Instrument: {primary_symbol}")
    if instrument_name and instrument_name != primary_symbol:
        lines.append(f"Instrument label: {instrument_name}")
    if venue:
        lines.append(f"Venue: {venue}")
    if timeframes:
        lines.append("Timeframes: " + ", ".join(timeframes))
    if trade_levels:
        level_summary = ", ".join(
            f"{field_name.replace('_', ' ')} {field_value}" for field_name, field_value in trade_levels.items()
        )
        lines.append("Visible levels: " + level_summary)

    visible_lines = select_visible_label_lines(ocr_text)
    if visible_lines:
        lines.append("Visible labels: " + "; ".join(visible_lines[:3]))

    return "\n".join(lines)


def build_visual_summary(
    asset_type: str,
    primary_symbol: str | None,
    instrument_name: str | None,
    venue: str | None,
    timeframes: list[str],
    ocr_text: str,
    trade_levels: dict[str, str],
    visual_elements: list[str],
) -> str:
    if asset_type == "chart":
        if primary_symbol and timeframes:
            opening = f"Chart crop showing {primary_symbol} on {', '.join(timeframes[:2])}."
        elif primary_symbol:
            opening = f"Chart crop showing {primary_symbol}."
        elif timeframes:
            opening = f"Chart crop with visible timeframe {', '.join(timeframes[:2])}."
        else:
            opening = "Trading chart crop."
    elif asset_type != "unknown":
        opening = f"{asset_type.replace('_', ' ').capitalize()} crop."
    else:
        opening = "Trading-related crop."

    sentences = [opening]
    if instrument_name and instrument_name != primary_symbol:
        sentences.append(f"Visible instrument label maps to {instrument_name}.")
    if venue:
        sentences.append(f"Visible venue label: {venue}.")
    if trade_levels:
        level_summary = ", ".join(
            f"{field_name.replace('_', ' ')} {field_value}" for field_name, field_value in trade_levels.items()
        )
        sentences.append(f"Visible price text includes {level_summary}.")

    visible_lines = select_visible_label_lines(ocr_text)
    if visible_lines:
        sentences.append("Visible text includes " + "; ".join(visible_lines[:2]) + ".")
    elif ocr_text:
        excerpt = sentence_excerpt(ocr_text, max_length=140)
        if excerpt:
            sentences.append(f"OCR excerpt: {excerpt}")

    if visual_elements:
        rendered_elements = ", ".join(element.replace("_", " ") for element in visual_elements[:6])
        sentences.append(f"Key visual elements: {rendered_elements}.")

    return " ".join(sentences[:4])


def build_review_reasons(annotation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    labels = annotation["labels"]

    if annotation["review_required"]:
        reasons.append("auto_review_flag")
    if labels.get("text_density") == "low":
        reasons.append("low_text_density")
    if annotation["asset_type"] == "unknown":
        reasons.append("unknown_asset_type")
    if annotation["page_type_confidence"] < 0.45:
        reasons.append("low_page_type_confidence")
    if "text_gap" in annotation["asset_source"]:
        reasons.append("text_gap_crop")
    if has_ocr_encoding_artifacts(annotation["ocr_text"]):
        reasons.append("ocr_encoding_artifacts")
    if not annotation["primary_symbol"]:
        reasons.append("missing_primary_symbol")
    if not annotation["timeframes"]:
        reasons.append("missing_timeframe")

    return unique_preserving_order(reasons)


def should_review_asset_annotation(
    combined_text: str,
    labels: dict[str, Any],
    asset_type: str,
    page_type_confidence: float,
    asset_source: str,
    ocr_text: str,
    primary_symbol: str | None,
    timeframes: list[str],
) -> bool:
    return (
        not combined_text
        or labels["text_density"] == "low"
        or asset_type == "unknown"
        or page_type_confidence < 0.45
        or "text_gap" in asset_source
        or has_ocr_encoding_artifacts(ocr_text)
        or not primary_symbol
        or not timeframes
    )


def should_review_page_annotation(
    combined_text: str,
    labels: dict[str, Any],
    page_type: str,
    page_type_confidence: float,
    ocr_text: str,
    symbols: list[str],
    timeframes: list[str],
    asset_count: int,
) -> bool:
    return (
        not combined_text
        or labels["text_density"] == "low"
        or page_type == "unknown"
        or page_type_confidence < 0.45
        or has_ocr_encoding_artifacts(ocr_text)
        or not timeframes
        or (asset_count > 0 and not symbols)
    )


def infer_annotation_quality(annotation: dict[str, Any], review_reasons: list[str]) -> str:
    score = 0
    if annotation["primary_symbol"]:
        score += 1
    if annotation["timeframes"]:
        score += 1
    if annotation["venue"]:
        score += 1
    if annotation["ocr_text"]:
        score += 1
    if annotation["context_text"]:
        score += 1
    if not has_ocr_encoding_artifacts(annotation["ocr_text"]):
        score += 1
    score -= min(len(review_reasons), 3)

    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def build_asset_training_target(annotation: dict[str, Any]) -> dict[str, Any]:
    context_text = annotation["context_text"]
    ocr_text = annotation["ocr_text"]
    primary_symbol_visible_in_ocr = symbol_is_visible_in_ocr(annotation["primary_symbol"], ocr_text)
    visible_symbol_matches = filter_visible_symbols(annotation["symbols"], ocr_text)
    visible_timeframe_matches = filter_visible_timeframes(annotation["timeframes"], ocr_text)
    detected_venues = extract_venues(ocr_text)
    venue_visible_in_ocr = bool(annotation["venue"] and annotation["venue"] in detected_venues)

    primary_symbol_source = infer_scalar_text_source(annotation["primary_symbol"], context_text, ocr_text)
    if primary_symbol_source == "heuristic" and annotation["primary_symbol"] and primary_symbol_visible_in_ocr:
        primary_symbol_source = "ocr_normalization"
    primary_symbol_source = promote_source_with_ocr_normalization(
        primary_symbol_source,
        has_visible_signal=primary_symbol_visible_in_ocr,
        ocr_text=ocr_text,
    )

    symbols_source = infer_list_text_source(
        annotation["symbols"],
        context_text,
        ocr_text,
        fallback="heuristic_normalization",
    )
    if (
        symbols_source == "heuristic_normalization"
        and annotation["symbols"]
        and visible_symbol_matches
    ):
        symbols_source = "ocr_normalization"
    symbols_source = promote_source_with_ocr_normalization(
        symbols_source,
        has_visible_signal=bool(visible_symbol_matches),
        ocr_text=ocr_text,
    )

    timeframes_source = infer_list_text_source(
        annotation["timeframes"],
        context_text,
        ocr_text,
        fallback="heuristic_normalization",
    )
    if (
        timeframes_source == "heuristic_normalization"
        and annotation["timeframes"]
        and visible_timeframe_matches
    ):
        timeframes_source = "ocr_normalization"
    timeframes_source = promote_source_with_ocr_normalization(
        timeframes_source,
        has_visible_signal=bool(visible_timeframe_matches),
        ocr_text=ocr_text,
    )

    venue_source = infer_scalar_text_source(annotation["venue"], context_text, ocr_text, fallback="heuristic_normalization")
    venue_source = promote_source_with_ocr_normalization(
        venue_source,
        has_visible_signal=venue_visible_in_ocr,
        ocr_text=ocr_text,
    )

    field_sources = {
        "asset_type": "heuristic_classifier",
        "page_type": "heuristic_classifier",
        "primary_symbol": primary_symbol_source,
        "instrument_name": (
            "symbol_lookup_from_ocr_symbol"
            if annotation["instrument_name"] and source_uses_ocr(primary_symbol_source)
            else "symbol_lookup_from_context_symbol"
            if annotation["instrument_name"] and annotation["primary_symbol"]
            else "missing"
        ),
        "venue": venue_source,
        "symbols": symbols_source,
        "timeframes": timeframes_source,
        "bias": infer_scalar_text_source(annotation["bias"], context_text, ocr_text, fallback="keyword_heuristic"),
        "direction": infer_scalar_text_source(
            annotation["direction"],
            context_text,
            ocr_text,
            fallback="keyword_heuristic",
        ),
        "setup_status": "keyword_heuristic" if annotation["setup_status"] else "missing",
        "trade_levels": infer_dict_text_source(annotation["trade_levels"], context_text, ocr_text),
        "trading_concepts": "keyword_heuristic" if annotation["trading_concepts"] else "missing",
        "trading_domains": "keyword_heuristic" if annotation["trading_domains"] else "missing",
    }

    visible_fields = resolve_visible_crop_fields(annotation, field_sources)
    visible_primary_symbol = visible_fields["primary_symbol"]
    visible_instrument_name = visible_fields["instrument_name"]
    visible_venue = visible_fields["venue"]
    visible_symbols = visible_fields["symbols"]
    visible_timeframes = visible_fields["timeframes"]
    visible_trade_levels = visible_fields["trade_levels"]
    visual_elements = build_visual_element_tags(annotation)
    review_reasons = build_review_reasons(annotation)

    return {
        "schema_version": "3.0",
        "task_type": "image_to_structured_annotation",
        "description": {
            "short_caption": annotation["caption"],
            "visual_summary": build_visual_summary(
                asset_type=annotation["asset_type"],
                primary_symbol=visible_primary_symbol,
                instrument_name=visible_instrument_name,
                venue=visible_venue,
                timeframes=visible_timeframes,
                ocr_text=ocr_text,
                trade_levels=visible_trade_levels,
                visual_elements=visual_elements,
            ),
            "context_augmented_summary": annotation["description"],
            "key_visual_elements": visual_elements,
            "limitations": review_reasons,
        },
        "observed": {
            "visible_in_crop": {
                "ocr_text": ocr_text,
                "clean_text": build_crop_clean_text(
                    asset_type=annotation["asset_type"],
                    primary_symbol=visible_primary_symbol,
                    instrument_name=visible_instrument_name,
                    venue=visible_venue,
                    timeframes=visible_timeframes,
                    ocr_text=ocr_text,
                    trade_levels=visible_trade_levels,
                ),
                "normalized_fields": {
                    "primary_symbol": visible_primary_symbol,
                    "instrument_name": visible_instrument_name,
                    "venue": visible_venue,
                    "symbols": visible_symbols,
                    "timeframes": visible_timeframes,
                },
                "trade_levels": visible_trade_levels,
                "visual_elements": visual_elements,
            },
            "paired_context": {
                "context_heading": annotation["context_heading"],
                "context_text": context_text,
            },
        },
        "derived": {
            "asset_type": annotation["asset_type"],
            "page_type": annotation["page_type"],
            "primary_symbol": annotation["primary_symbol"],
            "instrument_name": annotation["instrument_name"],
            "venue": annotation["venue"],
            "symbols": annotation["symbols"],
            "timeframes": annotation["timeframes"],
            "bias": annotation["bias"],
            "direction": annotation["direction"],
            "setup_status": annotation["setup_status"],
            "trade_levels": annotation["trade_levels"],
            "trading_concepts": annotation["trading_concepts"],
            "trading_domains": annotation["trading_domains"],
            "labels": annotation["labels"],
        },
        "provenance": {
            "source_document": {
                "source_pdf": annotation["source_pdf"],
                "page_number": annotation["page_number"],
                "asset_index": annotation["asset_index"],
                "asset_source": annotation["asset_source"],
            },
            "context_scope": "crop_plus_nearby_page_text" if context_text else "crop_only",
            "extraction_methods": annotation["extraction_methods"],
            "field_sources": field_sources,
            "quality": {
                "annotation_quality": infer_annotation_quality(annotation, review_reasons),
                "page_type_confidence": annotation["page_type_confidence"],
            },
            "review": {
                "required": annotation["review_required"],
                "reasons": review_reasons,
            },
            "ocr": {
                "enabled": annotation["ocr_metadata"].get("enabled"),
                "available": annotation["ocr_metadata"].get("available"),
                "status": annotation["ocr_metadata"].get("status"),
                "language": annotation["ocr_metadata"].get("language"),
            },
        },
    }


def asset_padding_for_source(asset_source: str) -> float:
    if "embedded_image" in asset_source:
        return 0.0
    return ASSET_RENDER_PADDING


def asset_render_scale(asset_bbox: BBox) -> float:
    long_edge_points = max(bbox_width(asset_bbox), bbox_height(asset_bbox), 1.0)
    target_scale = ASSET_TARGET_LONG_EDGE_PX / long_edge_points
    return round(max(PDF_RENDER_SCALE, min(target_scale, ASSET_MAX_RENDER_SCALE)), 2)


def reset_multimodal_asset_exports() -> None:
    if MULTIMODAL_PAIRS_DIR.exists():
        shutil.rmtree(MULTIMODAL_PAIRS_DIR)
    for root_dir in (MULTIMODAL_IMAGES_DIR, MULTIMODAL_ANNOTATIONS_DIR):
        if not root_dir.exists():
            continue
        for stale_dir in sorted(root_dir.rglob("assets"), reverse=True):
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_existing_llm_enrichments() -> dict[str, dict[str, Any]]:
    existing_enrichments: dict[str, dict[str, Any]] = {}
    if not MULTIMODAL_PAIRS_DIR.exists():
        return existing_enrichments

    for annotation_path in sorted(MULTIMODAL_PAIRS_DIR.glob("*.json")):
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(annotation, dict):
            continue
        if annotation.get("pair_type") != "visual_asset":
            continue
        if not isinstance(annotation.get("llm_enrichment"), dict):
            continue
        annotation_id = str(annotation.get("id", "")).strip()
        if not annotation_id:
            continue

        image_path = ROOT / str(annotation.get("image_path", ""))
        if not image_path.is_file():
            continue
        existing_enrichments[annotation_id] = {
            "annotation": annotation,
            "image_sha256": file_sha256(image_path),
        }
    return existing_enrichments


def find_existing_llm_enrichment(
    annotation: dict[str, Any],
    existing_enrichments: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    annotation_id = str(annotation.get("id", "")).strip()
    existing_by_id = existing_enrichments.get(annotation_id) if annotation_id else None

    image_path = ROOT / str(annotation.get("image_path", ""))
    if not image_path.is_file():
        if existing_by_id is not None:
            return None, "missing_current_image"
        return None, "none"

    current_image_sha256 = file_sha256(image_path)
    if existing_by_id is not None and current_image_sha256 == existing_by_id.get("image_sha256"):
        return existing_by_id, "preserved"

    hash_matches: list[dict[str, Any]] = []
    seen_match_ids: set[str] = set()
    for existing in existing_enrichments.values():
        if not isinstance(existing, dict):
            continue
        if current_image_sha256 != existing.get("image_sha256"):
            continue
        existing_annotation = existing.get("annotation")
        if not isinstance(existing_annotation, dict):
            continue
        existing_id = str(existing_annotation.get("id", "")).strip()
        match_key = existing_id or str(len(hash_matches))
        if match_key in seen_match_ids:
            continue
        seen_match_ids.add(match_key)
        hash_matches.append(existing)

    if len(hash_matches) == 1:
        return hash_matches[0], "preserved"
    if len(hash_matches) > 1:
        return None, "duplicate_image_hash"
    if existing_by_id is not None:
        return None, "image_changed"
    return None, "none"


def merge_existing_llm_target_json(
    target_json: dict[str, Any],
    existing_target_json: dict[str, Any],
) -> dict[str, Any]:
    merged = deepcopy(target_json)

    existing_description = existing_target_json.get("description")
    if isinstance(existing_description, dict):
        description = deepcopy(merged.get("description", {}))
        if not isinstance(description, dict):
            description = {}
        for field_name in LLM_TARGET_DESCRIPTION_FIELDS:
            if field_name in existing_description:
                description[field_name] = deepcopy(existing_description[field_name])
        merged["description"] = description

    existing_provenance = existing_target_json.get("provenance")
    if isinstance(existing_provenance, dict):
        provenance = deepcopy(merged.get("provenance", {}))
        if not isinstance(provenance, dict):
            provenance = {}

        current_methods = provenance.get("extraction_methods")
        extraction_methods = list(current_methods) if isinstance(current_methods, list) else []
        existing_methods = existing_provenance.get("extraction_methods")
        if not isinstance(existing_methods, list):
            existing_methods = []
        for method in existing_methods:
            if isinstance(method, str) and "llm" in method.lower() and method not in extraction_methods:
                extraction_methods.append(method)
        if extraction_methods:
            provenance["extraction_methods"] = extraction_methods

        for field_name in ("quality", "review"):
            existing_field = existing_provenance.get(field_name)
            if isinstance(existing_field, dict):
                provenance[field_name] = deepcopy(existing_field)

        merged["provenance"] = provenance

    return merged


def preserve_existing_llm_enrichment(
    annotation: dict[str, Any],
    existing_enrichments: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    existing, status = find_existing_llm_enrichment(annotation, existing_enrichments)
    if existing is None:
        return annotation, status

    existing_annotation = existing.get("annotation")
    if not isinstance(existing_annotation, dict):
        return annotation, "none"

    preserved = dict(annotation)
    for field_name in LLM_ENRICHMENT_PRESERVED_FIELDS:
        if field_name in existing_annotation:
            preserved[field_name] = existing_annotation[field_name]
    target_json = preserved.get("target_json")
    existing_target_json = existing_annotation.get("target_json")
    if isinstance(target_json, dict) and isinstance(existing_target_json, dict):
        preserved["target_json"] = merge_existing_llm_target_json(target_json, existing_target_json)
    return preserved, status


def export_visual_asset_image(
    page: Any,
    asset_candidate: VisualAssetCandidate,
    source_pdf: str,
    page_number: int,
    asset_index: int,
    fitz: Any,
) -> tuple[Path, dict[str, Any]]:
    embedded_image = asset_candidate.image_block
    if embedded_image is not None and asset_candidate.source == "embedded_image":
        image_extension = normalize_image_extension(embedded_image.ext)
        if image_extension and embedded_image.image_bytes:
            image_path, _json_path = build_asset_pair_paths(
                source_pdf,
                page_number,
                asset_index,
                image_extension=image_extension,
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(embedded_image.image_bytes)
            return image_path, {
                "mode": "embedded_original",
                "format": image_extension,
                "pixel_size": {
                    "width": embedded_image.width,
                    "height": embedded_image.height,
                },
                "source_resolution": {
                    "xres": embedded_image.xres,
                    "yres": embedded_image.yres,
                },
            }

    render_scale = asset_render_scale(asset_candidate.bbox)
    image_path, _json_path = build_asset_pair_paths(source_pdf, page_number, asset_index, image_extension="png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    asset_pixmap = page.get_pixmap(
        matrix=fitz.Matrix(render_scale, render_scale),
        clip=fitz.Rect(*asset_candidate.bbox),
        alpha=False,
    )
    asset_pixmap.save(str(image_path))
    return image_path, {
        "mode": "rendered_crop",
        "format": "png",
        "render_scale": render_scale,
    }


def render_pdf_multimodal_assets() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fitz = import_pymupdf()
    ocr_runtime = detect_ocr_runtime()
    if fitz is None:
        return [], [], {
            "available": False,
            "status": "missing_pymupdf",
            "message": "PyMuPDF is required to render PDF pages into image+json pairs.",
            "ocr_runtime": ocr_runtime,
        }

    page_annotations: list[dict[str, Any]] = []
    asset_annotations: list[dict[str, Any]] = []
    existing_enrichments = collect_existing_llm_enrichments()
    reset_multimodal_asset_exports()
    MULTIMODAL_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_count = 0
    page_count = 0
    asset_count = 0
    preserved_llm_enrichment_count = 0
    stale_llm_enrichment_count = 0
    asset_source_counts: dict[str, int] = {}

    for pdf_path in sorted(RAW_PDFS_DIR.rglob("*.pdf")):
        if not pdf_path.is_file():
            continue

        source_pdf = pdf_path.relative_to(RAW_PDFS_DIR).as_posix()
        pdf_slug = slugify(str(Path(source_pdf).with_suffix("")))
        page_image_dir = MULTIMODAL_IMAGES_DIR / pdf_slug / "pages"
        page_annotation_dir = MULTIMODAL_ANNOTATIONS_DIR / pdf_slug / "pages"
        page_image_dir.mkdir(parents=True, exist_ok=True)
        page_annotation_dir.mkdir(parents=True, exist_ok=True)

        pdf_document = fitz.open(pdf_path)
        pdf_count += 1
        try:
            for page_index, page in enumerate(pdf_document):
                page_number = page_index + 1
                page_bbox = to_bbox(page.rect)
                page_id = stable_id("pdf_page", source_pdf, str(page_number))
                page_image_path = page_image_dir / f"page_{page_number:04d}.png"
                page_annotation_path = page_annotation_dir / f"page_{page_number:04d}.json"

                text_blocks, image_blocks = extract_page_layout(page)
                embedded_text = normalize_whitespace("\n\n".join(text_block.text for text_block in text_blocks))
                asset_candidates = discover_visual_asset_candidates(page, text_blocks, image_blocks, page_bbox)
                expanded_asset_bboxes = [
                    bbox_expand(asset_candidate.bbox, asset_padding_for_source(asset_candidate.source), page_bbox)
                    for asset_candidate in asset_candidates
                ]

                asset_ids: list[str] = []
                page_asset_annotations: list[dict[str, Any]] = []
                for asset_index, asset_candidate in enumerate(asset_candidates, start=1):
                    asset_bbox = expanded_asset_bboxes[asset_index - 1]
                    asset_id = build_asset_pair_id(source_pdf, page_number, asset_index)
                    asset_image_path, image_export = export_visual_asset_image(
                        page=page,
                        asset_candidate=VisualAssetCandidate(
                            bbox=asset_bbox,
                            source=asset_candidate.source,
                            image_block=asset_candidate.image_block if asset_bbox == asset_candidate.bbox else None,
                        ),
                        source_pdf=source_pdf,
                        page_number=page_number,
                        asset_index=asset_index,
                        fitz=fitz,
                    )
                    _pair_image_path, asset_annotation_path = build_asset_pair_paths(
                        source_pdf,
                        page_number,
                        asset_index,
                        image_extension=asset_image_path.suffix,
                    )

                    raw_ocr_text, ocr_meta = run_ocr(asset_image_path)
                    polished_ocr_text = "\n".join(polish_context_lines(raw_ocr_text, max_lines=10))
                    sibling_asset_bboxes = [
                        sibling_bbox
                        for sibling_index, sibling_bbox in enumerate(expanded_asset_bboxes, start=1)
                        if sibling_index != asset_index
                    ]
                    nearby_text_blocks = select_asset_text_blocks(
                        text_blocks,
                        asset_bbox,
                        page_bbox,
                        sibling_asset_bboxes=sibling_asset_bboxes,
                    )
                    context_text = build_context_text(nearby_text_blocks)
                    combined_text, _unused_methods = merge_extracted_text(context_text, polished_ocr_text)
                    extraction_methods = []
                    if context_text:
                        extraction_methods.append("nearby_text")
                    if polished_ocr_text:
                        extraction_methods.append("ocr")

                    analysis_text = combined_text or embedded_text
                    market_fields = derive_asset_market_fields(
                        context_text=context_text,
                        ocr_text=polished_ocr_text,
                        fallback_text=analysis_text,
                    )
                    symbols = market_fields["symbols"]
                    primary_symbol = market_fields["primary_symbol"]
                    instrument_name = market_fields["instrument_name"]
                    venue = market_fields["venue"]
                    timeframes = market_fields["timeframes"]
                    trading_concepts = extract_keyword_labels(analysis_text, CONCEPT_KEYWORDS)
                    trading_domains = extract_keyword_labels(analysis_text, DOMAIN_KEYWORDS)
                    page_type, page_type_confidence = infer_page_type(analysis_text, symbols, timeframes)
                    asset_type = infer_asset_type(
                        context_text=context_text,
                        ocr_text=polished_ocr_text,
                        symbols=symbols,
                        timeframes=timeframes,
                        trading_concepts=trading_concepts,
                        trading_domains=trading_domains,
                    )
                    bias = extract_bias(analysis_text)
                    direction = extract_direction(analysis_text)
                    setup_status = extract_setup_status(analysis_text)
                    trade_levels = extract_trade_levels(analysis_text)
                    caption = build_asset_summary(
                        asset_type=asset_type,
                        page_type=page_type,
                        primary_symbol=primary_symbol,
                        venue=venue,
                        symbols=symbols,
                        timeframes=timeframes,
                        concepts=trading_concepts,
                        bias=bias,
                        setup_status=setup_status,
                        text=analysis_text,
                    )
                    description = build_asset_description(
                        asset_type=asset_type,
                        page_type=page_type,
                        primary_symbol=primary_symbol,
                        instrument_name=instrument_name,
                        venue=venue,
                        symbols=symbols,
                        timeframes=timeframes,
                        concepts=trading_concepts,
                        context_text=context_text,
                        ocr_text=polished_ocr_text,
                        bias=bias,
                        direction=direction,
                        setup_status=setup_status,
                        trade_levels=trade_levels,
                    )
                    clean_text = build_asset_clean_text(
                        asset_type=asset_type,
                        primary_symbol=primary_symbol,
                        instrument_name=instrument_name,
                        venue=venue,
                        timeframes=timeframes,
                        bias=bias,
                        direction=direction,
                        setup_status=setup_status,
                        context_text=context_text,
                        ocr_text=polished_ocr_text,
                        trade_levels=trade_levels,
                    )
                    labels = build_asset_labels(
                        asset_type=asset_type,
                        page_type=page_type,
                        combined_text=analysis_text,
                        context_text=context_text,
                        ocr_text=polished_ocr_text,
                        symbols=symbols,
                        timeframes=timeframes,
                        asset_bbox=asset_bbox,
                        page_bbox=page_bbox,
                        paired_text_blocks=len(nearby_text_blocks),
                    )
                    review_required = should_review_asset_annotation(
                        combined_text=combined_text,
                        labels=labels,
                        asset_type=asset_type,
                        page_type_confidence=page_type_confidence,
                        asset_source=asset_candidate.source,
                        ocr_text=polished_ocr_text,
                        primary_symbol=primary_symbol,
                        timeframes=timeframes,
                    )

                    asset_annotation = {
                        "id": asset_id,
                        "annotation_version": "3.0",
                        "pair_type": "visual_asset",
                        "image_path": root_relative(asset_image_path),
                        "json_path": root_relative(asset_annotation_path),
                        "source_pdf": source_pdf,
                        "page_number": page_number,
                        "asset_index": asset_index,
                        "asset_source": asset_candidate.source,
                        "asset_bbox": bbox_to_json(asset_bbox),
                        "asset_bbox_ratio": bbox_ratio_to_json(asset_bbox, page_bbox),
                        "asset_type": asset_type,
                        "page_type": page_type,
                        "page_type_confidence": page_type_confidence,
                        "caption": caption,
                        "summary": caption,
                        "description": description,
                        "clean_text": clean_text,
                        "context_heading": infer_context_heading(nearby_text_blocks),
                        "context_text": context_text,
                        "ocr_text": polished_ocr_text,
                        "combined_text": combined_text,
                        "extraction_methods": extraction_methods,
                        "ocr_metadata": ocr_meta,
                        "primary_symbol": primary_symbol,
                        "instrument_name": instrument_name,
                        "venue": venue,
                        "symbols": symbols,
                        "timeframes": timeframes,
                        "bias": bias,
                        "direction": direction,
                        "setup_status": setup_status,
                        "trade_levels": trade_levels,
                        "trading_concepts": trading_concepts,
                        "trading_domains": trading_domains,
                        "labels": labels,
                        "review_required": review_required,
                        "image_export": image_export,
                    }
                    asset_annotation, preserve_status = preserve_existing_llm_enrichment(
                        asset_annotation,
                        existing_enrichments,
                    )
                    asset_annotation["target_json"] = build_asset_training_target(asset_annotation)
                    if preserve_status == "preserved":
                        existing_enrichment, _status = find_existing_llm_enrichment(
                            asset_annotation,
                            existing_enrichments,
                        )
                        existing_target_json = (
                            existing_enrichment.get("annotation", {}).get("target_json")
                            if isinstance(existing_enrichment, dict)
                            else None
                        )
                        if isinstance(existing_target_json, dict):
                            asset_annotation["target_json"] = merge_existing_llm_target_json(
                                asset_annotation["target_json"],
                                existing_target_json,
                            )
                    if preserve_status == "preserved":
                        preserved_llm_enrichment_count += 1
                    elif preserve_status != "none":
                        stale_llm_enrichment_count += 1
                    write_json(asset_annotation_path, asset_annotation)
                    asset_annotations.append(asset_annotation)
                    page_asset_annotations.append(asset_annotation)
                    asset_ids.append(asset_id)
                    asset_count += 1
                    for source_name in asset_candidate.source.split("+"):
                        asset_source_counts[source_name] = asset_source_counts.get(source_name, 0) + 1

                page_pixmap = page.get_pixmap(matrix=fitz.Matrix(PDF_RENDER_SCALE, PDF_RENDER_SCALE), alpha=False)
                page_pixmap.save(str(page_image_path))

                ocr_text, ocr_meta = run_ocr(page_image_path)
                combined_text, extraction_methods = merge_extracted_text(embedded_text, ocr_text)

                symbols = extract_symbols(combined_text)
                timeframes = extract_timeframes(combined_text)
                symbols, timeframes = merge_page_market_fields(symbols, timeframes, page_asset_annotations)
                trading_concepts = extract_keyword_labels(combined_text, CONCEPT_KEYWORDS)
                trading_domains = extract_keyword_labels(combined_text, DOMAIN_KEYWORDS)
                page_type, page_type_confidence = infer_page_type(combined_text, symbols, timeframes)
                labels = build_page_labels(page_type, combined_text, embedded_text, ocr_text, symbols, timeframes)
                summary = build_auto_summary(page_type, symbols, timeframes, trading_concepts, combined_text)

                review_required = should_review_page_annotation(
                    combined_text=combined_text,
                    labels=labels,
                    page_type=page_type,
                    page_type_confidence=page_type_confidence,
                    ocr_text=ocr_text,
                    symbols=symbols,
                    timeframes=timeframes,
                    asset_count=len(asset_ids),
                )

                page_annotation = {
                    "id": page_id,
                    "annotation_version": "2.0",
                    "image_path": root_relative(page_image_path),
                    "json_path": root_relative(page_annotation_path),
                    "source_pdf": source_pdf,
                    "page_number": page_number,
                    "page_type": page_type,
                    "page_type_confidence": page_type_confidence,
                    "summary": summary,
                    "embedded_text": embedded_text,
                    "ocr_text": ocr_text,
                    "combined_text": combined_text,
                    "extraction_methods": extraction_methods,
                    "ocr_metadata": ocr_meta,
                    "symbols": symbols,
                    "timeframes": timeframes,
                    "trading_concepts": trading_concepts,
                    "trading_domains": trading_domains,
                    "labels": labels,
                    "review_required": review_required,
                    "asset_count": len(asset_ids),
                    "asset_ids": asset_ids,
                }
                page_annotation["target_json"] = build_page_training_target(page_annotation)
                write_json(page_annotation_path, page_annotation)
                page_annotations.append(page_annotation)
                page_count += 1
        finally:
            pdf_document.close()

    return page_annotations, asset_annotations, {
        "available": True,
        "status": "ok",
        "pdf_count": pdf_count,
        "page_count": page_count,
        "asset_count": asset_count,
        "preserved_llm_enrichment_count": preserved_llm_enrichment_count,
        "stale_llm_enrichment_count": stale_llm_enrichment_count,
        "asset_source_counts": asset_source_counts,
        "render_scale": PDF_RENDER_SCALE,
        "asset_target_long_edge_px": ASSET_TARGET_LONG_EDGE_PX,
        "asset_max_render_scale": ASSET_MAX_RENDER_SCALE,
        "pairs_dir": root_relative(MULTIMODAL_PAIRS_DIR),
        "ocr_enabled": OCR_ENABLED,
        "ocr_language": OCR_LANGUAGE,
        "ocr_runtime": ocr_runtime,
    }


def build_image_json_pair_index(asset_annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": annotation["id"],
            "image_path": annotation["image_path"],
            "json_path": annotation["json_path"],
            "image_export": annotation.get("image_export", {}),
            "source_pdf": annotation["source_pdf"],
            "page_number": annotation["page_number"],
            "asset_index": annotation["asset_index"],
            "asset_type": annotation["asset_type"],
            "page_type": annotation["page_type"],
            "asset_source": annotation["asset_source"],
            "review_required": annotation["review_required"],
        }
        for annotation in asset_annotations
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_NODES_DIR.mkdir(parents=True, exist_ok=True)
    MULTIMODAL_DIR.mkdir(parents=True, exist_ok=True)
    MULTIMODAL_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    MULTIMODAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MULTIMODAL_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    documents = collect_documents()
    documents_rows = build_documents_jsonl(documents)
    knowledge_rows = build_knowledge_jsonl(documents)
    trade_rows = build_trade_candidates(documents)
    page_annotations, asset_annotations, multimodal_status = render_pdf_multimodal_assets()
    image_json_pairs = build_image_json_pair_index(asset_annotations)

    documents_count = write_jsonl(PROCESSED_DIR / "documents.jsonl", documents_rows)
    knowledge_count = write_jsonl(PROCESSED_DIR / "knowledge_corpus.jsonl", knowledge_rows)
    trade_count = write_jsonl(PROCESSED_DIR / "trade_candidates.jsonl", trade_rows)
    page_annotation_count = write_jsonl(MULTIMODAL_DIR / "page_annotations.jsonl", page_annotations)
    asset_annotation_count = write_jsonl(MULTIMODAL_DIR / "asset_annotations.jsonl", asset_annotations)
    pair_index_count = write_jsonl(MULTIMODAL_DIR / "image_json_pairs.jsonl", image_json_pairs)

    manifest = {
        "documents": documents_count,
        "knowledge_chunks": knowledge_count,
        "trade_candidates": trade_count,
        "multimodal_page_annotations": page_annotation_count,
        "multimodal_asset_annotations": asset_annotation_count,
        "multimodal_image_json_pairs": pair_index_count,
        "raw_pdfs_dir": str(RAW_PDFS_DIR),
        "raw_nodes_dir": str(RAW_NODES_DIR),
        "multimodal_status": multimodal_status,
    }
    write_json(PROCESSED_DIR / "manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
