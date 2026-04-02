"""Simple local OCR. Pass image paths as arguments."""

import sys
from pathlib import Path

from ocr_engine import OcrEngine
from engines.rapid import RapidOcrEngine

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


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

    for img_path in image_paths:
        print(f"\n{'='*60}")
        print(f"File: {img_path}")
        print(f"{'='*60}")

        result = engine.recognize(str(img_path))

        if result.text:
            print(result.text)
            if result.confidence is not None:
                print(f"(avg confidence: {result.confidence:.2f})")

            # Save result to output/
            out_file = output_dir / f"{img_path.stem}.txt"
            out_file.write_text(result.text, encoding="utf-8")
            print(f"\nSaved to: {out_file}")
        else:
            print("(no text detected)")


if __name__ == "__main__":
    main()
