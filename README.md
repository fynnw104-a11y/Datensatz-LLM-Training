# Trading Dataset Prep for LLM Backtesting

## Schnellstart

Wenn du einfach nur willst, dass es ohne Nachdenken laeuft:

1. `pip install -r requirements.txt`
2. Lege deine PDFs nach `data/raw/pdfs/`
3. Starte unter Windows `START_HERE.bat`

Der Launcher zeigt ein einfaches Menue fuer:

- Projektcheck
- ChatGPT vorbereiten
- Dataset aus PDFs bauen
- Asset-Beschreibungen mit ChatGPT anreichern
- Train/Eval-Split bauen
- Alles automatisch

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
- `data/processed/multimodal/pairs/...`
- `data/processed/multimodal/images/...` fuer Seitenbilder
- `data/processed/multimodal/annotations/...` fuer Seiten-Metadaten
- `data/processed/manifest.json`

Wenn `PyMuPDF` verfuegbar ist, analysiert der Script das PDF-Layout, extrahiert einzelne Bild-/Chart-Regionen und erzeugt dafuer eigene Bild+JSON-Paare.

Zusatznutzen:

- `page_annotations.jsonl` bleibt als Seiten-Metadaten und Traceability-Layer erhalten.
- `data/processed/multimodal/pairs/` ist jetzt der autoritative Export fuer Asset-Bild+JSON-Paare mit gleichem Basenamen in genau einem Ordner.
- eingebettete PDF-Bilder werden dort wenn moeglich in ihrem Originalformat und ihrer Originalqualitaet gespeichert; andere Assets werden mit hoher Render-Aufloesung als PNG exportiert.
- `asset_annotations.jsonl` ist der aggregierte Index ueber dieselben Asset-Paare.
- `image_json_pairs.jsonl` zeigt auf die Dateien im zentralen `pairs`-Ordner.
- Die Asset-JSONs enthalten normalisierte Felder wie `primary_symbol`, `instrument_name`, `venue`, `bias`, `setup_status` und ein trainingsfreundliches `clean_text`.
- Das eigentliche `target_json` fuer Bild->JSON-Training trennt jetzt sauber zwischen `description`, `observed`, `derived` und `provenance`.
- `observed.visible_in_crop` enthaelt nur Crop-nahe Beobachtungen; `observed.paired_context` markiert zusaetzlichen Seitentext explizit als externen Kontext.
- Ueber `provenance.field_sources` und `provenance.context_scope` kannst du spaeter entscheiden, ob du strikt bildbasiert oder bild+kontextbasiert trainieren willst.

Wenn `pytesseract` und ein lokales Tesseract-Binary verfuegbar sind, wird zusaetzlich OCR auf die extrahierten Bilder angewendet.

Im aktuellen Worktree wird Tesseract automatisch an diesen Orten erkannt:

- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- `.\.tessdata\` als bevorzugter Projektordner fuer eigene Sprachmodelle

Optional relevante Umgebungsvariablen:

```powershell
$env:PDF_RENDER_SCALE="2.0"
$env:ASSET_TARGET_LONG_EDGE_PX="2200"
$env:ASSET_MAX_RENDER_SCALE="4.0"
$env:ENABLE_OCR="1"
$env:TESSERACT_LANG="deu+eng"
$env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:ASSET_MIN_AREA_RATIO="0.02"
$env:ASSET_CONTEXT_MARGIN="42"
```

Die Seiten-Annotationen folgen `schemas/pdf_page_annotation.schema.json`.

Die Asset-Annotationen folgen `schemas/pdf_visual_asset_annotation.schema.json`.

Das eigentliche Trainingsziel innerhalb von `target_json` folgt `schemas/visual_asset_training_target.schema.json`.

## 1b. Bildbeschreibungen per ChatGPT im Browser anreichern

Wenn du die heuristischen Beschreibungen in den Asset-JSONs mit einem LLM nachschaerfen willst, gibt es jetzt einen separaten Selenium-Stealth-Layer fuer `chatgpt.com`.

Wichtige Architektur:

- `scripts/chatgpt_automation/`
  Generischer Browser-Layer auf Basis von Selenium + `selenium-stealth` fuer Session-Restore, Selektoren, Prompt-Versand, Datei-Upload und Antworterfassung.
- `scripts/run_chatgpt_batch.py`
  Allgemeiner JSONL-Batch-Runner fuer Prompt+Attachment-Jobs.
- `scripts/enrich_multimodal_descriptions.py`
  Dataset-spezifischer Enricher fuer die bestehenden Asset-Annotationen unter `data/processed/multimodal/pairs/*.json`.

Der Browser-Layer nutzt ein konfiguriertes Browser-Profil oder eine konfigurierte Cookie-Datei, wenn vorhanden. Ohne explizite Pfade liegen die Standardpfade im ignorierten Projektordner `.runtime/chatgpt/...`. Wenn das nicht reicht, faellt er auf einen interaktiven Login in einem separaten normalen Browserfenster mit demselben Profil zurueck. Dieses Fenster nach dem Login wieder schliessen, damit die Automation das Profil uebernehmen kann. Zugangsdaten werden dabei nicht hart im Code aus den Selektor-Dateien uebernommen.

Beispiel: bestehende Asset-JSONs mit ChatGPT verbessern

```powershell
python scripts/enrich_multimodal_descriptions.py --limit 10 --language en
```

Wirkung:

- jedes Asset-Bild wird mit Kontexttext und OCR an ChatGPT geschickt
- der Multimodal-Enricher verarbeitet standardmaessig bis zu 20 Assets pro Chat und startet dann automatisch einen frischen Chat
- mit `--new-chat-per-asset` erzwingst du wieder einen komplett frischen Chat fuer jedes Asset
- die Antwort wird als strukturierte `llm_enrichment`-Sektion gespeichert
- `caption`, `summary`, `description`, `clean_text` und die `target_json.description`-Felder werden mit der LLM-Beschreibung aktualisiert
- `target_json.observed.visible_in_crop` bleibt crop-grounded und wird nicht mit Kontext- oder LLM-Zusatztext ueberschrieben
- rohe Batch-Ergebnisse landen unter `data/processed/chatgpt_runs/`

Nutzliche Optionen:

- `--dry-run`
  Fuehrt die Browserabfragen aus, schreibt aber noch keine Annotationen zurueck.
- `--limit 5`
  Gut fuer erste Tests.
- `--language de`
  Erzeugt deutsche Beschreibungen statt englischer.
- `--no-skip-existing-llm`
  Erzwingt eine Neubeschreibung bereits angereicherter Assets.
- `--max-assets-per-chat 5`
  Startet nach jeweils 5 Assets automatisch einen neuen Chat.
- `--max-assets-per-chat 0`
  Nutzt nach dem ersten frischen Chat denselben Verlauf fuer den kompletten Lauf weiter.
- `--new-chat-per-asset`
  Erzwingt wieder das alte Verhalten mit einem frischen Chat fuer jedes Asset.
- `--keep-browser-open`
  Laesst das Automations-Browserfenster nach dem Lauf offen, damit du die Session und die letzte ChatGPT-Antwort direkt pruefen kannst.

Beispiel: generischer ChatGPT-Batch fuer Bilder oder reine Textprompts

```powershell
python scripts/run_chatgpt_batch.py --input jobs.jsonl --output results.jsonl
```

Erwartetes `jobs.jsonl`-Format:

```json
{"id":"job-1","prompt":"Describe the attached chart as training data.","attachments":["C:/abs/path/image.png"],"metadata":{"kind":"image_caption"}}
{"id":"job-2","prompt":"Rewrite this text into a short LoRA training target.","attachments":[],"metadata":{"kind":"text_only"}}
```

Optional:

- `"new_chat": true`
  Startet fuer genau diesen Job einen neuen Chat statt den aktuellen Verlauf weiterzuverwenden.

Hinweise zum Browser-Setup:

- die Skripte lesen `ChatGPT/config.json`
- als Vorlage kannst du `ChatGPT/config.example.json` verwenden
- wenn du es maximal einfach willst, starte `START_HERE.bat` und waehle `ChatGPT vorbereiten`
- Browser-Profil und Cookies landen standardmaessig im ignorierten Ordner `.runtime/chatgpt/`
- falls `chatgpt.com` im Selenium-Fenster eine Bot- oder Security-Challenge zeigt, zuerst den separaten manuellen Profil-Login abschliessen; wenn die Challenge danach weiter erscheint, ist die Browser-Automation auf dieser Session aktuell blockiert
- `user_data_dir`, `cookies_file`, `driver_path` und `browser_executable` koennen dort oder per Umgebungsvariablen gesetzt werden
- unter Windows werden Chrome und Edge an typischen Standardpfaden gesucht
- falls kein passender WebDriver lokal verfuegbar ist, musst du den Browser-Driver auf deinem Rechner lauffaehig machen; `selenium` und `selenium-stealth` kommen ueber `pip install -r requirements.txt`

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

Fuer striktes Bild->JSON-Training solltest du nur Felder verwenden, deren Quelle in `provenance.field_sources` auf `ocr` basiert oder die unter `observed.visible_in_crop` liegen. Felder aus `observed.paired_context` sind bewusst getrennt, weil sie nicht zwingend direkt im Crop sichtbar sind.

## Was als Naechstes sinnvoll ist

Sobald du erste echte Rohdaten in `data/raw/` gelegt hast, koennen wir zusammen:

- die Feldnamen deiner Trading-Records auf ein gemeinsames Schema mappen
- schwache oder irrelevante Dokumente aussortieren
- aus deinen historischen Trades gute `user -> assistant` Beispiele bauen
- entscheiden, ob LoRA, reines SFT oder Retrieval + Tooling fuer den Agenten besser ist
