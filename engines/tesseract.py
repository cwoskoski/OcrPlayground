"""Tesseract engine implementation."""

import pytesseract
from PIL import Image

from ocr_engine import OcrEngine, OcrResult


class TesseractEngine(OcrEngine):
    def __init__(self, lang: str = "eng"):
        self._lang = lang

    def recognize(self, image_path: str) -> OcrResult:
        img = Image.open(image_path)
        data = pytesseract.image_to_data(img, lang=self._lang, output_type=pytesseract.Output.DICT)
        confidences = [c for c in data["conf"] if c > 0]
        avg_conf = sum(confidences) / len(confidences) / 100 if confidences else None
        text = pytesseract.image_to_string(img, lang=self._lang).strip()
        return OcrResult(text=text, confidence=avg_conf)
