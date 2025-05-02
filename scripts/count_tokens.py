#!/usr/bin/env python3
"""Count total word tokens in a HF Arrow dataset saved on disk."""
import multiprocessing
from datasets import load_from_disk


def main():
    """Load 'pre1900_corpus.arrow', count tokens (words), and print totals."""
    dataset_path = 'pre1900_corpus.arrow'
    # Use all but one CPU core for processing
    nproc = max(1, multiprocessing.cpu_count() - 1)
    ds = load_from_disk(dataset_path)
    # Compute token counts in parallel (split on whitespace)
    ds_tc = ds.map(
        lambda batch: {'token_count': [len(text.split()) for text in batch['text']]},
        batched=True,
        batch_size=1000,
        num_proc=nproc,
    )
    total_tokens = sum(ds_tc['token_count'])
    print(f"Dataset: {dataset_path}")
    print(f"Examples: {ds.num_rows}")
    print(f"Total tokens (words): {total_tokens}")
    print(f"~{total_tokens/1e6:.2f}M tokens")


if __name__ == '__main__':
    main()