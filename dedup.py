from __future__ import annotations

import argparse
from typing import Iterable, List

from datasets import Dataset, load_from_disk
from tqdm.auto import tqdm
from text_dedup.minhash import MinHashDeduper

# ---------------------------------------------------------------------------
# 1) Chunk helpers
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 128) -> Iterable[str]:
    """Yield ~``chunk_size``-word chunks with ``overlap`` to preserve context."""
    words = text.split()
    step = max(32, chunk_size - overlap)
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + chunk_size])


def explode_into_chunks(ds: Dataset, chunk_size: int) -> Dataset:
    rows: List[dict] = []
    for ex in tqdm(ds, desc="Chunking books"):
        book_id = ex.get("id") or ex.get("identifier") or ex["__index_level_0__"]
        for idx, chunk in enumerate(chunk_text(ex["text"], chunk_size)):
            rows.append({"book_id": book_id, "chunk_id": idx, "text": chunk})
    return Dataset.from_list(rows)

# ---------------------------------------------------------------------------
# 2) Self‑dedup via text‑dedup
# ---------------------------------------------------------------------------

def dedup_chunks(ds_chunks: Dataset, threshold: float = 0.9) -> Dataset:
    deduper = MinHashDeduper(
        shingle_size=5,
        threshold=threshold,
        column="text",
        num_perm=128,
        progress_bar=True,
    )
    return deduper.deduplicate_dataset(ds_chunks)

# ---------------------------------------------------------------------------
# 4) CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to saved HF dataset (save_to_disk)")
    ap.add_argument("--out", required=True, help="Output path (save_to_disk)")
    ap.add_argument("--chunk_size", type=int, default=1024, help="Words per chunk (default 1k)")
    ap.add_argument("--threshold", type=float, default=0.9, help="Jaccard threshold (default 0.9)")
    args = ap.parse_args()

    print("[dedup‑self] Loading dataset …")
    ds = load_from_disk(args.inp)

    ds = ds.map(
        lambda x:
    )

    ds_chunks = explode_into_chunks(ds_books, chunk_size=args.chunk_size)
    print(f"[dedup‑self] {len(ds_chunks):,} chunks generated from {len(ds_books):,} books.")

    ds_chunks_clean = dedup_chunks(ds_chunks, threshold=args.threshold)
    print(f"[dedup‑self] {len(ds_chunks_clean):,} unique chunks remain (removed {len(ds_chunks) - len(ds_chunks_clean):,}).")

    ds_chunks_clean.save_to_disk(args.out)

if __name__ == "__main__":
    main()
