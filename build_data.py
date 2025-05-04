from datasets import load_dataset, interleave_datasets

"""
load_dataset("TheBritishLibrary/blbooks", streaming=True)
    .filter(lambda x: x["date"]<=1899 and x["ocr_confidence"]>0.92),
load_dataset("dell-research-harvard/AmericanStories", "subset_years",
                year_list=[str(y) for y in range(1800,1900)])

# royal society
# TODO - no year - need regex filtering
https://huggingface.co/datasets/bigscience-data/roots_en_royal_society_corpus
"""
import re
import json
import random
import multiprocessing

_NUM_PROC = max(1, multiprocessing.cpu_count() - 1)

_MODERN_PATTERNS = [
    # Key modern scientific theories
    r"\b(general|special) relativity\b",
    r"\bquantum (?:mechanics|theory)\b",
    # Modern inventions and technologies
    r"\bairplane\b",
    r"\btelevision\b",
    r"\bcomputer\b",
    r"\binternet\b",
    r"\bjet engine\b",
    r"\blaser\b",
    r"\bsmartphone\b",
]
_MODERN_RE = re.compile("|".join(_MODERN_PATTERNS), flags=re.IGNORECASE)

_SNIPPET_LEN = 50
def extract_snippet(text, match, context_len=_SNIPPET_LEN):
    """Extract a snippet of text of approximately context_len characters around the regex match."""
    pre = context_len // 2
    post = context_len - pre
    start = max(0, match.start() - pre)
    end = min(len(text), match.end() + post)
    return str(match.start()) + ":" + text[start:end]

def is_obvious_modern(text):
    """Detect obvious modern references such as post-1900 years or specific modern terms."""
    return bool(_MODERN_RE.search(text))

def debug_is_obvious_modern(ds):
    # Capture a sample of filtered-out examples for manual review
    # capture removed examples for sampling in parallel
    ds_removed = ds.filter(lambda x: is_obvious_modern(x.get("text", "")), num_proc=_NUM_PROC)
    count_removed = ds_removed.num_rows
    print(f"Filtered out {count_removed} examples with obvious modern references.")
    sample_count = min(100, count_removed)
    if count_removed > 0:
        random.seed(42)
        indices = random.sample(range(count_removed), sample_count) if count_removed > sample_count else list(range(count_removed))
        sample_removed = ds_removed.select(indices)
        with open("filtered_out_samples.jsonl", "w", encoding="utf-8") as f:
            for ex in sample_removed:
                # extract only the snippet around the matched modern reference
                text = ex.get("text", "")
                m = _MODERN_RE.search(text)
                if m:
                    snippet = extract_snippet(text, m)
                    match_txt = m.group()
                else:
                    snippet = text[:_SNIPPET_LEN]
                    match_txt = None
                out = {
                    "publication_date": ex.get("publication_date"),
                    "short_book_title": ex.get("short_book_title"),
                    "match": match_txt,
                    "snippet": snippet,
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"Saved {sample_count} filtered samples to filtered_out_samples.jsonl")
    else:
        print("No filtered-out examples to sample.")

HEADER_RE = re.compile(r"(produced by|internet|scanner|executive director|gutenberg|computer)", re.IGNORECASE)

def clean_pg19(example):
    """
    Prunes any Gutenberg headers.  Current implementation gets us down
    to filtering only 439 books.  The remainder seem to scatter Gutenberg
    annotations randomly throughout the text.
    """
    text = example["text"]
    # Normalize newlines
    text = text.replace("\r\n", "\n")
    # Look for gutenberge headers
    header_len = 10000
    header = text[:header_len]
    matches = list(HEADER_RE.finditer(header))
    if len(matches) > 0:
        # Just truncate everything up to here plus some margin.
        header = header[matches[-1].start()+100:]
    text = header + text[header_len:]
    example["text"] = text.strip()
    return example

def load_gutenberg():
    ds = load_dataset("emozilla/pg19", num_proc=5, split="train")
    # metadata filter: keep only texts published before 1900
    ds = ds.filter(lambda x: x["publication_date"] < 1900, num_proc=_NUM_PROC)
    ds = ds.map(clean_pg19, num_proc=_NUM_PROC)
    count_meta = ds.num_rows

    # debug_is_obvious_modern(ds)

    # safety filter
    ds = ds.filter(lambda x: not is_obvious_modern(x.get("text", "")), num_proc=_NUM_PROC)
    count_keep = ds.num_rows
    num_filtered = count_meta - count_keep
    print(f"Metadata filter kept {count_meta} rows; regex filter removed {num_filtered} rows, keeping {count_keep} rows.")

    ds = ds.remove_columns(["short_book_title", "url"])
    return ds

def main():
    ds = load_gutenberg()
    cleaned = ds.shuffle(seed=42)
    cleaned.save_to_disk("pre1900_corpus.arrow")

if __name__ == "__main__":
    main()