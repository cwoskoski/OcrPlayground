"""Extract structured data from OCR text using a local LLM via Ollama."""

import json
import sys
from pathlib import Path

from ollama import chat
from pydantic import BaseModel


class Person(BaseModel):
    first_name: str | None = None
    middle_name: str | None = None
    last_name: str | None = None


class BirthCertificate(BaseModel):
    child: Person = Person()
    date_of_birth: str | None = None
    sex: str | None = None
    father: Person = Person()
    mother: Person = Person()


PROMPT = """\
/no_think
Extract data from this OCR'd US birth certificate. The OCR has errors — words run together and labels are garbled.

Here is the OCR text with line numbers:
{ocr_text}

Based on the text above, extract:
- Child: first name, middle name, last name (look after lines mentioning "CHILD" or "1A/1B/1C")
- Date of birth: in MM/DD/YYYY format
- Sex: MALE or FEMALE
- Father: first name, middle name, last name (look after lines mentioning "FATHER" or "6A/6B/6C")
- Mother: first name, middle name, last name (look after lines mentioning "MOTHER" or "9A/9B/9C")

Note: "BANAMEOFFATHERPARENT" means "6A NAME OF FATHER/PARENT". "MAMEOFMOTHERVPARENT" means "NAME OF MOTHER/PARENT".
Values like SCOTT, STRATTON, MELISSA, PASILLAS are people's names that appear after label lines.
"CA" is a US state abbreviation (California), NOT a person's name — never use it as a name field.
Dates look like MM/DD/YYYY. After each group of name labels (first/middle/last), the next 1-3 lines \
are the values in that same order. If only 2 names appear for a 3-field group, the middle name may be missing."""

MODEL = "qwen3:14b"


def extract(ocr_text: str) -> BirthCertificate:
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT.format(ocr_text=ocr_text)}],
        format=BirthCertificate.model_json_schema(),
        options={"temperature": 0},
    )
    result = BirthCertificate.model_validate_json(response.message.content)
    # Normalize empty strings to None
    for person in [result.child, result.father, result.mother]:
        for field in ["first_name", "middle_name", "last_name"]:
            if getattr(person, field) == "":
                setattr(person, field, None)
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract.py <ocr_output.txt or image>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    raw_text = path.read_text(encoding="utf-8")
    # Add line numbers to help the LLM reason about position
    lines = raw_text.splitlines()
    ocr_text = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
    print(f"Extracting from: {path}")
    print(f"Using model: {MODEL}\n")

    result = extract(ocr_text)
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
