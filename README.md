# Trading Dataset Prep for LLM Backtesting

Dieses Projektgeruest bereitet rohe Trading-Daten fuer zwei verschiedene Zwecke auf:

1. `knowledge_corpus.jsonl`
   Eine saubere Wissensbasis aus PDFs, Markdown, Text, JSON, JSONL und CSV.
2. `trade_candidates.jsonl`
   Automatisch normalisierte Kandidaten aus strukturierten Trading-Records.
3. `multimodal/`
   Einzelne PDF-Bilder/Charts als Bild+JSON-Paare fuer spaeteres Vision-/Multimodal-Training.

Wichtige Trennung:

- Historische Markt- und Backtest-Daten sollten fuer einen Agenten moeglichst zur Laufzeit verfuegbar sein.
- Eine LoRA ist eher fuer Entscheidungsstil, Regelanwendung, Antwortformat und Begruendung sinnvoll.
- Rohdokumente allein sind noch kein gutes SFT-Trainingsset. Deshalb erzeugt dieses Projekt zuerst ein belastbares Korpus und reviewbare Kandidaten.

## Ordnerstruktur

```text
data/
  raw/
    pdfs/
    nodes/
  curated/
  processed/
schemas/
scripts/
```

## Erwartete Eingangsdaten

Lege deine Dateien hier ab:

- `data/raw/pdfs`: PDF-Dateien mit Strategien, Notes, Reports, Playbooks
- `data/raw/nodes`: JSON, JSONL, CSV, TXT, MD, YAML/YML aus Nodes, Exports, Logs oder Notizen

Unterstuetzte Dateitypen:

- `.pdf`
- `.json`
- `.jsonl`
- `.csv`
- `.md`
- `.txt`
- `.yaml`
- `.yml`

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Rohdaten extrahieren

```powershell
python scripts/prepare_dataset.py
```

Erzeugte Dateien:

- `data/processed/documents.jsonl`
- `data/processed/knowledge_corpus.jsonl`
- `data/processed/trade_candidates.jsonl`
- `data/processed/multimodal/page_annotations.jsonl`
- `data/processed/multimodal/asset_annotations.jsonl`
- `data/processed/multimodal/image_json_pairs.jsonl`
- `data/processed/multimodal/images/...`
- `data/processed/multimodal/annotations/...`
- `data/processed/manifest.json`

Wenn `PyMuPDF` verfuegbar ist, analysiert der Script das PDF-Layout, extrahiert einzelne Bild-/Chart-Regionen und erzeugt dafuer eigene PNG+JSON-Paare.

Zusatznutzen:

- `page_annotations.jsonl` bleibt als Seiten-Metadaten und Traceability-Layer erhalten.
- `asset_annotations.jsonl` ist der eigentliche Trainingsexport auf Asset-Ebene.
- `image_json_pairs.jsonl` zeigt nur auf die extrahierten Bild-/Chart-Crops, nicht mehr auf ganze Seiten.
- Die Asset-JSONs enthalten normalisierte Felder wie `primary_symbol`, `instrument_name`, `venue`, `bias`, `setup_status` und ein trainingsfreundliches `clean_text`.

Wenn `pytesseract` und ein lokales Tesseract-Binary verfuegbar sind, wird zusaetzlich OCR auf die extrahierten Bilder angewendet.

Im aktuellen Worktree wird Tesseract automatisch an diesen Orten erkannt:

- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- `.\.tessdata\` als bevorzugter Projektordner fuer eigene Sprachmodelle

Optional relevante Umgebungsvariablen:

```powershell
$env:PDF_RENDER_SCALE="2.0"
$env:ENABLE_OCR="1"
$env:TESSERACT_LANG="deu+eng"
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:ASSET_MIN_AREA_RATIO="0.02"
$env:ASSET_CONTEXT_MARGIN="42"
```

Die Seiten-Annotationen folgen `schemas/pdf_page_annotation.schema.json`.

Die Asset-Annotationen folgen `schemas/pdf_visual_asset_annotation.schema.json`.

## 2. Trainingsbeispiele kuratieren

Die Datei `trade_candidates.jsonl` ist absichtlich eine Review-Queue, kein blindes Final-Trainingsset.

Der Zielzustand fuer LoRA/SFT ist ein manuell oder halbautomatisch geprueftes Set in:

- `data/curated/training_examples.jsonl`

Das erwartete Format ist in `schemas/training_example.schema.json` beschrieben.

## 3. Train/Eval-Split bauen

```powershell
python scripts/build_training_split.py
```

Erzeugte Dateien:

- `data/processed/train.jsonl`
- `data/processed/eval.jsonl`

## Empfohlene Datenstrategie

Fuer deinen Use Case ist diese Reihenfolge sauberer als direktes LoRA-Training auf allen PDFs:

1. Wissensquellen extrahieren und saubern.
2. Einzelne PDF-Bilder/Charts als Bild+JSON-Paare exportieren.
3. Strukturierte Trading-Records normalisieren.
4. Echte Decision-Beispiele kuratieren.
5. Ein separates Eval-Set bauen.
6. Erst dann ueber LoRA oder SFT nachdenken.

## Wichtige Modellgrenze

Eine reine Text-LoRA kann keine Bilder lernen. Wenn du die extrahierten PDF-Bilder wirklich in das Training einbeziehen willst, brauchst du spaeter ein Vision-Language-Modell oder ein Multimodal-Finetuning-Format.

Das hier erzeugte Bild+JSON-Format ist bewusst modellagnostisch vorbereitet, damit wir spaeter nicht frueh auf ein falsches Trainingsformat festgelegt sind.

## Was als Naechstes sinnvoll ist

Sobald du erste echte Rohdaten in `data/raw/` gelegt hast, koennen wir zusammen:

- die Feldnamen deiner Trading-Records auf ein gemeinsames Schema mappen
- schwache oder irrelevante Dokumente aussortieren
- aus deinen historischen Trades gute `user -> assistant` Beispiele bauen
- entscheiden, ob LoRA, reines SFT oder Retrieval + Tooling fuer den Agenten besser ist
