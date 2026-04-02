"""Simple local OCR. Pass image paths or PDFs as arguments."""

import sys
import tempfile
from pathlib import Path

from pdf2image import convert_from_path

from ocr_engine import OcrEngine
from engines.rapid import RapidOcrEngine

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


def main():
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_or_directory> [image2 ...]")
        print("  Supports: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
        sys.exit(1)

    # Collect all image paths from arguments (files or directories)
    image_paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    image_paths.append(f)
        elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(p)
        else:
            print(f"Skipping: {arg}")

    if not image_paths:
        print("No valid images found.")
        sys.exit(1)

    engine: OcrEngine = RapidOcrEngine()

    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for file_path in image_paths:
        print(f"\n{'='*60}")
        print(f"File: {file_path}")
        print(f"{'='*60}")

        if file_path.suffix.lower() in PDF_EXTENSIONS:
            # Convert PDF pages to images, OCR each page
            pages = convert_from_path(str(file_path), dpi=300)
            all_text = []
            all_conf = []
            for i, page in enumerate(pages):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    page.save(tmp.name, "PNG")
                    result = engine.recognize(tmp.name)
                if result.text:
                    if len(pages) > 1:
                        print(f"\n--- Page {i + 1} ---")
                    print(result.text)
                    all_text.append(result.text)
                    if result.confidence is not None:
                        all_conf.append(result.confidence)

            combined = "\n".join(all_text)
            if combined:
                avg_conf = sum(all_conf) / len(all_conf) if all_conf else None
                if avg_conf is not None:
                    print(f"(avg confidence: {avg_conf:.2f})")
                out_file = output_dir / f"{file_path.stem}.txt"
                out_file.write_text(combined, encoding="utf-8")
                print(f"\nSaved to: {out_file}")
            else:
                print("(no text detected)")
        else:
            result = engine.recognize(str(file_path))

            if result.text:
                print(result.text)
                if result.confidence is not None:
                    print(f"(avg confidence: {result.confidence:.2f})")

                out_file = output_dir / f"{file_path.stem}.txt"
                out_file.write_text(result.text, encoding="utf-8")
                print(f"\nSaved to: {out_file}")
            else:
                print("(no text detected)")


if __name__ == "__main__":
    main()
