# OcrPlayground

Local OCR testing playground. Python 3.12+. Everything runs locally — no data sent to external servers.

## Architecture

- `ocr_engine.py` — Abstract `OcrEngine` base class and `OcrResult` dataclass. All OCR backends implement this interface.
- `engines/` — Concrete engine implementations:
  - `engines/rapid.py` — RapidOCR (default). Best accuracy on structured documents. Uses `return_word_box=True` and `unclip_ratio=1.2` to improve word separation.
  - `engines/tesseract.py` — Tesseract OCR. Better word segmentation but noisier output on forms with watermarks/backgrounds.
- `ocr.py` — CLI entry point for OCR. Uses the engine abstraction; swap backends by changing one import and one line.
- `extract.py` — Structured data extraction from OCR text using a local LLM (Qwen3:14B via Ollama). Currently extracts birth certificate fields (names, DOB, sex, parent names) into JSON.

## OCR Engine Notes

- RapidOCR is the preferred engine for documents like birth certificates. Key tuning parameters:
  - `Det.unclip_ratio` — controls text box expansion. Default 1.6 causes word concatenation. 1.2 is the best balance. Lower values (1.0, 0.8) fix more concatenation but start splitting real words.
  - `Global.return_word_box` — must be True for proper word-level output.
  - `Det.use_dilation` — disabling can help but tends to over-split words.
- Tesseract requires system package: `sudo apt install tesseract-ocr tesseract-ocr-eng`
- RapidOCR requires `onnxruntime` as a runtime dependency (not pulled in automatically by pip).

## LLM Extraction (Ollama)

- Uses Ollama to run Qwen3:14B locally for structured data extraction from OCR text.
- Ollama is installed but the systemd service is **disabled by default** to save resources. Start manually with `ollama serve &` when needed.
- Models are stored in `~/.ollama/models/`. Both `qwen3:8b` (5.2 GB) and `qwen3:14b` (9.3 GB) are downloaded.
- The 14B model fits in the RTX 4000 Ada's 12 GB VRAM and runs in a few seconds per document.
- Prompt engineering matters significantly for extraction quality. Key lessons:
  - `/no_think` disables Qwen3's reasoning mode (faster but less accurate).
  - Explicitly telling the LLM that "CA" is a state abbreviation prevents it from being used as a name.
  - Describing the label-then-value line pattern of the form helps the model parse the layout.

## Conventions

- New OCR backends go in `engines/` and must subclass `OcrEngine` from `ocr_engine.py`.
- Keep the `OcrEngine` interface minimal — `recognize(image_path: str) -> OcrResult`.
- No frameworks; this is a simple CLI tool. Keep it lightweight.
- Use `requirements.txt` for dependencies (no pyproject.toml or setup.py).

## Running

```bash
source .venv/bin/activate

# OCR an image
python ocr.py <image_or_directory>

# Extract structured data from OCR output (requires ollama serve running)
ollama serve &
python extract.py output/<file>.txt
```
