"""OCR engine abstraction layer for swapping OCR libraries."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class OcrResult:
    text: str
    confidence: float | None = None


class OcrEngine(ABC):
    """Base interface for OCR engines."""

    @abstractmethod
    def recognize(self, image_path: str) -> OcrResult:
        """Run OCR on an image file and return extracted text."""
        ...
