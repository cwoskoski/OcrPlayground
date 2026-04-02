"""Extract structured data from birth certificates using Amazon Bedrock.

Two modes:
  full   — Send the document directly to Claude (multimodal, skips OCR)
  hybrid — OCR locally with RapidOCR, then send text to Claude for extraction

Supports both images (.png, .jpg, etc.) and PDFs.
"""

import base64
import io
import json
import sys
import tempfile
from pathlib import Path

import boto3
from pdf2image import convert_from_path
from PIL import Image

from engines.rapid import RapidOcrEngine

AWS_PROFILE = "CVT_AWS_Dev"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

FORM_FIELDS_HINT = """\
US birth certificates follow a standard layout with numbered fields:
- 1A/1B/1C: Child's first name, middle name, last name
- 2: Sex (MALE/FEMALE)
- 3A/3B: Birth plurality (single, twin, etc.)
- 4A/4B: Date of birth, time of birth
- 5A/5B/5C/5D: Place of birth (hospital, address, city, county)
- 6A/6B/6C: Father/parent's first name, middle name, last name
- 7: Father's birthplace (state/country)
- 8: Father's date of birth
- 9A/9B/9C: Mother/parent's first name, middle name, last name (birth/maiden name)
- 10: Mother's birthplace (state/country)
- 11: Mother's date of birth
Field numbers and labels vary slightly by state but the structure is consistent."""

EXTRACTION_PROMPT = """\
Extract birth certificate information from the following OCR text.
The OCR has errors — words may run together and labels may be garbled.

{form_fields}

Extract:
- Child: first name, middle name, last name
- Date of birth: in MM/DD/YYYY format
- Sex: MALE or FEMALE
- Father: first name, middle name, last name
- Mother: first name, middle name, last name (birth/maiden name)

Two-letter abbreviations like CA, NY, TX are US state abbreviations, not names.
Return null for any field you cannot find.
Respond with ONLY valid JSON matching this schema, no other text:
{schema}

OCR TEXT:
{ocr_text}"""

IMAGE_PROMPT = """\
This is an image of a US birth certificate. Extract the following fields:
- Child: first name, middle name, last name
- Date of birth: in MM/DD/YYYY format
- Sex: MALE or FEMALE
- Father: first name, middle name, last name
- Mother: first name, middle name, last name (birth/maiden name)

{form_fields}

Return null for any field you cannot find.
Respond with ONLY valid JSON matching this schema, no other text:
{schema}"""

SCHEMA = json.dumps({
    "child": {"first_name": "", "middle_name": "", "last_name": ""},
    "date_of_birth": "",
    "sex": "",
    "father": {"first_name": "", "middle_name": "", "last_name": ""},
    "mother": {"first_name": "", "middle_name": "", "last_name": ""},
}, indent=2)


def get_bedrock_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client("bedrock-runtime")


def invoke_bedrock(client, messages: list, max_tokens: int = 1024) -> tuple[str, dict]:
    resp = client.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": messages,
        }),
    )
    body = json.loads(resp["body"].read())
    return body["content"][0]["text"], body.get("usage", {})


MAX_IMAGE_BYTES = 4_500_000  # Stay under Bedrock's 5 MB limit


def _is_pdf(file_path: str) -> bool:
    return file_path.lower().endswith(".pdf")


def _compress_image(img: Image.Image) -> tuple[bytes, str]:
    """Compress an image to JPEG under the Bedrock size limit."""
    for quality in (85, 70, 50, 30):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= MAX_IMAGE_BYTES:
            return data, "image/jpeg"
    # Last resort: resize to half
    img = img.reduce(2)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=50)
    return buf.getvalue(), "image/jpeg"


def _load_image_bytes(file_path: str) -> list[tuple[bytes, str]]:
    """Load image(s) from a file path, compressing if needed. Handles PDFs."""
    if _is_pdf(file_path):
        pil_images = convert_from_path(file_path, dpi=200)
    else:
        pil_images = [Image.open(file_path)]

    result = []
    for img in pil_images:
        # Try original format first
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        raw_bytes = buf.getvalue()

        if len(raw_bytes) <= MAX_IMAGE_BYTES:
            result.append((raw_bytes, "image/jpeg"))
        else:
            result.append(_compress_image(img))
    return result


def _image_content_blocks(file_path: str) -> list[dict]:
    """Build image content blocks for Bedrock messages, handling PDFs."""
    pages = _load_image_bytes(file_path)

    blocks = []
    for img_bytes, media_type in pages:
        b64_image = base64.b64encode(img_bytes).decode("utf-8")
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_image,
            },
        })
    return blocks


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def extract_full(client, file_path: str) -> tuple[dict, dict]:
    """Send document directly to Claude on Bedrock (multimodal)."""
    content = _image_content_blocks(file_path)
    content.append({
        "type": "text",
        "text": IMAGE_PROMPT.format(schema=SCHEMA, form_fields=FORM_FIELDS_HINT),
    })

    messages = [{"role": "user", "content": content}]
    raw, usage = invoke_bedrock(client, messages)
    return _parse_json(raw), usage


def extract_hybrid(client, file_path: str) -> dict:
    """OCR locally, then send text to Claude on Bedrock for extraction."""
    engine = RapidOcrEngine()

    if _is_pdf(file_path):
        pages = convert_from_path(file_path, dpi=200)
        all_text = []
        for page in pages:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                page.save(tmp.name, "PNG")
                result = engine.recognize(tmp.name)
                if result.text:
                    all_text.append(result.text)
        ocr_text = "\n".join(all_text)
        confidence = None
    else:
        ocr_result = engine.recognize(file_path)
        ocr_text = ocr_result.text
        confidence = ocr_result.confidence

    if not ocr_text:
        print("(no text detected by OCR)")
        return {}

    if confidence is not None:
        print(f"OCR confidence: {confidence:.2f}")
    print(f"OCR text length: {len(ocr_text)} chars\n")

    messages = [{
        "role": "user",
        "content": EXTRACTION_PROMPT.format(schema=SCHEMA, ocr_text=ocr_text, form_fields=FORM_FIELDS_HINT),
    }]

    raw, usage = invoke_bedrock(client, messages)
    return _parse_json(raw), usage


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_bedrock.py <mode> <file_path>")
        print("  Modes:")
        print("    full   — Send document directly to Bedrock (multimodal)")
        print("    hybrid — OCR locally, then extract via Bedrock")
        print("  Supports: images (.png, .jpg, etc.) and PDFs (.pdf)")
        sys.exit(1)

    mode = sys.argv[1]
    file_path = sys.argv[2]

    if mode not in ("full", "hybrid"):
        print(f"Unknown mode: {mode}. Use 'full' or 'hybrid'.")
        sys.exit(1)

    if not Path(file_path).is_file():
        print(f"File not found: {file_path}")
        sys.exit(1)

    print(f"Mode: {mode}")
    print(f"File: {file_path}")
    print(f"Model: {MODEL_ID}\n")

    client = get_bedrock_client()

    if mode == "full":
        result, usage = extract_full(client, file_path)
    else:
        result, usage = extract_hybrid(client, file_path)

    print(json.dumps(result, indent=2))

    if usage:
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        print(f"\n--- Token Usage ---")
        print(f"Input:  {input_tokens:,}")
        print(f"Output: {output_tokens:,}")
        print(f"Total:  {input_tokens + output_tokens:,}")


if __name__ == "__main__":
    main()
