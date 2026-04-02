"""RapidOCR engine implementation."""

from rapidocr import RapidOCR

from ocr_engine import OcrEngine, OcrResult


class RapidOcrEngine(OcrEngine):
    def __init__(self):
        self._engine = RapidOCR(params={
            "Global.return_word_box": True,
            "Det.unclip_ratio": 1.2,
        })

    def recognize(self, image_path: str) -> OcrResult:
        result = self._engine(image_path)
        if not result or not result.word_results:
            return OcrResult(text="")
        lines = []
        all_confidences = []
        for line_words in result.word_results:
            words = []
            for text, conf, _box in line_words:
                words.append(text)
                all_confidences.append(conf)
            lines.append(" ".join(words))
        avg_conf = (
            sum(all_confidences) / len(all_confidences)
            if all_confidences
            else None
        )
        return OcrResult(
            text="\n".join(lines),
            confidence=avg_conf,
        )
