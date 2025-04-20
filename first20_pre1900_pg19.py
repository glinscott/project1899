#!/usr/bin/env python3
"""Stream the first 20 pre-1900 Gutenberg texts via Hugging Face Datasets (PG-19)."""

import re
import json
from datasets import load_dataset

START_PATTERN = re.compile(
    r"\*\*\* START OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*", re.DOTALL
)
END_PATTERN = re.compile(
    r"\*\*\* END OF THIS PROJECT GUTENBERG EBOOK.*?\*\*\*", re.DOTALL
)

def strip_headers(text: str) -> str:
    """Remove Gutenberg header/footer boilerplate from text."""
    m = START_PATTERN.search(text)
    if m:
        text = text[m.end():]
    m = END_PATTERN.search(text)
    if m:
        text = text[:m.start()]
    return text.strip()


def main():
    # Stream the PG-19 dataset with custom code trust
    ds = load_dataset(
        "pg19", "all", split="train", streaming=True, trust_remote_code=True
    )
    out_path = "first20_pre1900.jsonl"
    count = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for ex in ds:
            print(f"Processing {ex.get('url', '')}")
            # parse publication year
            pub_date = ex.get("publication_date", "")
            try:
                year = int(pub_date)
            except ValueError:
                continue
            if year >= 1900:
                continue

            text = strip_headers(ex.get("text", ""))
            rec = {
                "id": ex.get("url", ""),
                "title": ex.get("short_book_title", ""),
                "publication_date": pub_date,
                "text": text,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
            if count >= 20:
                break
    print(f"Wrote {count} records to {out_path}")


if __name__ == "__main__":
    main()