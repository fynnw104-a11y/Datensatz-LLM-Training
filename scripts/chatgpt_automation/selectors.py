from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass(frozen=True)
class SelectorEntry:
    selector_id: str
    label: str
    file_path: Path
    raw: dict[str, Any]

    @property
    def selectors(self) -> dict[str, Any]:
        return self.raw.get("selectors", {})

    @property
    def validation(self) -> dict[str, Any]:
        return self.raw.get("validation", {})

    def iter_candidates(self) -> Iterable[tuple[str, str]]:
        css = str(self.selectors.get("css", "")).strip()
        xpath = str(self.selectors.get("xpath", "")).strip()
        if css:
            yield ("css", css)
        if xpath:
            yield ("xpath", xpath)

        for alternative in self.selectors.get("alternatives", []):
            candidate = str(alternative).strip()
            if not candidate:
                continue
            selector_type = "xpath" if candidate.startswith("/") else "css"
            yield (selector_type, candidate)

    def matches_label(self, *terms: str) -> bool:
        normalized_label = _normalize(self.label)
        return all(_normalize(term) in normalized_label for term in terms if term.strip())


class SelectorCatalog:
    def __init__(self, entries: list[SelectorEntry]) -> None:
        self.entries = entries
        self.by_id = {entry.selector_id: entry for entry in entries}

    @classmethod
    def from_files(cls, selector_files: Iterable[Path]) -> "SelectorCatalog":
        entries: list[SelectorEntry] = []
        for selector_file in selector_files:
            if not selector_file.exists():
                continue
            payload = json.loads(selector_file.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                selector_id = str(item.get("id", "")).strip()
                label = str(item.get("label", "")).strip()
                if not selector_id:
                    continue
                entries.append(
                    SelectorEntry(
                        selector_id=selector_id,
                        label=label,
                        file_path=selector_file,
                        raw=item,
                    )
                )
        return cls(entries)

    def get(self, selector_id: str) -> SelectorEntry | None:
        return self.by_id.get(selector_id)

    def find_first(self, predicate: Callable[[SelectorEntry], bool]) -> SelectorEntry | None:
        for entry in self.entries:
            if predicate(entry):
                return entry
        return None

    def find_by_label_terms(self, *terms: str) -> SelectorEntry | None:
        return self.find_first(lambda entry: entry.matches_label(*terms))

    def find_by_attribute(self, attribute_name: str, attribute_value: str) -> SelectorEntry | None:
        expected = attribute_value.strip()
        return self.find_first(
            lambda entry: str(entry.raw.get("element", {}).get("attributes", {}).get(attribute_name, "")).strip()
            == expected
        )

    def require(self, selector_id: str) -> SelectorEntry:
        entry = self.get(selector_id)
        if entry is None:
            raise KeyError(f"Missing selector entry: {selector_id}")
        return entry
