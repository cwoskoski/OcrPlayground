"""Extract structured data from birth certificates using Amazon Bedrock.

Two modes:
  full   — Send the document directly to Claude (multimodal, skips OCR)
  hybrid — OCR locally with RapidOCR, then send text to Claude for extraction

Supports both images (.png, .jpg, etc.) and PDFs.
"""

import base64
import json
import sys
import tempfile
from pathlib import Path

import boto3
from pdf2image import convert_from_path

from engines.rapid import RapidOcrEngine

AWS_PROFILE = "CVT_AWS_Dev"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

EXTRACTION_PROMPT = """\
Extract birth certificate information from the following OCR text.
The OCR has errors — words may run together and labels may be garbled.

Extract:
- Child: first name, middle name, last name
- Date of birth: in MM/DD/YYYY format
- Sex: MALE or FEMALE
- Father: first name, middle name, last name
- Mother: first name, middle name, last name (birth/maiden name)

"CA" is a US state abbreviation, not a person's name.
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


def invoke_bedrock(client, messages: list, max_tokens: int = 1024) -> str:
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
    return body["content"][0]["text"]


def _is_pdf(file_path: str) -> bool:
    return file_path.lower().endswith(".pdf")


def _pdf_to_images(pdf_path: str) -> list[tuple[bytes, str]]:
    """Convert PDF pages to PNG images. Returns list of (bytes, media_type)."""
    pages = convert_from_path(pdf_path, dpi=300)
    result = []
    for page in pages:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            page.save(tmp.name, "PNG")
            result.append((Path(tmp.name).read_bytes(), "image/png"))
    return result


def _image_content_blocks(file_path: str) -> list[dict]:
    """Build image content blocks for Bedrock messages, handling PDFs."""
    if _is_pdf(file_path):
        pages = _pdf_to_images(file_path)
    else:
        media_type = "image/png" if file_path.lower().endswith(".png") else "image/jpeg"
        pages = [(Path(file_path).read_bytes(), media_type)]

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


def extract_full(client, file_path: str) -> dict:
    """Send document directly to Claude on Bedrock (multimodal)."""
    content = _image_content_blocks(file_path)
    content.append({
        "type": "text",
        "text": IMAGE_PROMPT.format(schema=SCHEMA),
    })

    messages = [{"role": "user", "content": content}]
    raw = invoke_bedrock(client, messages)
    return json.loads(raw)


def extract_hybrid(client, file_path: str) -> dict:
    """OCR locally, then send text to Claude on Bedrock for extraction."""
    engine = RapidOcrEngine()

    if _is_pdf(file_path):
        pages = convert_from_path(file_path, dpi=300)
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
        "content": EXTRACTION_PROMPT.format(schema=SCHEMA, ocr_text=ocr_text),
    }]

    raw = invoke_bedrock(client, messages)
    return json.loads(raw)


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
        result = extract_full(client, file_path)
    else:
        result = extract_hybrid(client, file_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
