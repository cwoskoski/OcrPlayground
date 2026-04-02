"""Extract structured data from birth certificates using Amazon Bedrock.

Two modes:
  full   — Send the image directly to Claude (multimodal, skips OCR)
  hybrid — OCR locally with RapidOCR, then send text to Claude for extraction
"""

import base64
import json
import sys
from pathlib import Path

import boto3

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


def extract_full(client, image_path: str) -> dict:
    """Send image directly to Claude on Bedrock (multimodal)."""
    img_bytes = Path(image_path).read_bytes()
    media_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": b64_image,
                },
            },
            {
                "type": "text",
                "text": IMAGE_PROMPT.format(schema=SCHEMA),
            },
        ],
    }]

    raw = invoke_bedrock(client, messages)
    return json.loads(raw)


def extract_hybrid(client, image_path: str) -> dict:
    """OCR locally, then send text to Claude on Bedrock for extraction."""
    engine = RapidOcrEngine()
    ocr_result = engine.recognize(image_path)

    if not ocr_result.text:
        print("(no text detected by OCR)")
        return {}

    print(f"OCR confidence: {ocr_result.confidence:.2f}")
    print(f"OCR text length: {len(ocr_result.text)} chars\n")

    messages = [{
        "role": "user",
        "content": EXTRACTION_PROMPT.format(schema=SCHEMA, ocr_text=ocr_result.text),
    }]

    raw = invoke_bedrock(client, messages)
    return json.loads(raw)


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_bedrock.py <mode> <image_path>")
        print("  Modes:")
        print("    full   — Send image directly to Bedrock (multimodal)")
        print("    hybrid — OCR locally, then extract via Bedrock")
        sys.exit(1)

    mode = sys.argv[1]
    image_path = sys.argv[2]

    if mode not in ("full", "hybrid"):
        print(f"Unknown mode: {mode}. Use 'full' or 'hybrid'.")
        sys.exit(1)

    if not Path(image_path).is_file():
        print(f"File not found: {image_path}")
        sys.exit(1)

    print(f"Mode: {mode}")
    print(f"Image: {image_path}")
    print(f"Model: {MODEL_ID}\n")

    client = get_bedrock_client()

    if mode == "full":
        result = extract_full(client, image_path)
    else:
        result = extract_hybrid(client, image_path)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
