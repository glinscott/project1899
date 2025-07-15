
## July 15, 2025

### Status

5.6B tokens:
```
% uv run scripts/count_tokens.py
Loading dataset from disk: 100%|████████████████████████████████████████████████████████| 80/80 [00:11<00:00,  6.82it/s]
Dataset: pre1900_corpus.arrow
Examples: 14041686
Total tokens (words): 5677715845
~5677.72M tokens
```

### TODOs

- Pull in https://huggingface.co/datasets/institutional/institutional-books-1.0
  - 260B tokens!  Need to filter to pre-1900 though.
- Run dedup.py
- 1B test training run on the existing 5.6B dataset?
- Prototype experiment environment
