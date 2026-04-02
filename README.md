# OcrPlayground

Local OCR testing playground with a pluggable engine abstraction and LLM-powered structured data extraction. Everything runs locally — no data is sent to external servers.

## Prerequisites

- Python 3.12+
- [Tesseract](https://github.com/tesseract-ocr/tesseract) (optional, for Tesseract engine):
  ```bash
  sudo apt install tesseract-ocr tesseract-ocr-eng
  ```
- [Ollama](https://ollama.com) (optional, for structured data extraction):
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen3:14b
  ```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### OCR an image

```bash
python ocr.py image.png
python ocr.py scan1.jpg scan2.png
python ocr.py ./images/           # process all images in a directory
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, `.tif`

Extracted text is printed to the console and saved to the `output/` directory as `.txt` files.

### Extract structured data from OCR output

Requires Ollama running with Qwen3:14B:

```bash
ollama serve &
python extract.py output/birth_certificate.txt
```

Returns structured JSON with fields like child name, date of birth, sex, and parent names.

## Project Structure

```
ocr.py              # CLI entry point for OCR
ocr_engine.py       # Abstract OcrEngine interface and OcrResult dataclass
extract.py          # LLM-powered structured data extraction (Ollama + Qwen3)
engines/
  rapid.py          # RapidOCR implementation (default)
  tesseract.py      # Tesseract implementation
requirements.txt
```

## OCR Engines

The project uses an abstraction layer (`OcrEngine`) so OCR libraries can be swapped with a one-line change in `ocr.py`.

| Engine | Strengths | Weaknesses |
|---|---|---|
| **RapidOCR** (default) | High accuracy (97%+), good on structured documents | Words can concatenate on tight layouts |
| **Tesseract** | Better word segmentation | Noisier output on forms with watermarks |

### Adding a New OCR Engine

1. Create a new file in `engines/` (e.g., `engines/my_engine.py`).
2. Implement the `OcrEngine` interface:

```python
from ocr_engine import OcrEngine, OcrResult

class MyEngine(OcrEngine):
    def recognize(self, image_path: str) -> OcrResult:
        # your implementation here
        return OcrResult(text="...", confidence=0.95)
```

3. Swap the engine in `ocr.py`:

```python
from engines.my_engine import MyEngine
engine: OcrEngine = MyEngine()
```
