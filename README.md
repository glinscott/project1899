# Core path

## Pretraining on pre-1900 texts

Pipeline:
```
[Raw texts]
[Metadata filter → drop post-1900]
[Regex pre-filter → drop obvious modern]
[LLM classifier → accept/reject ambiguous]
[Spot-check sample manually]
[Final corpus]
```

## LLM proposes theories/experiments to build knowledge

- LLM proposes experiments (actions)
- Environment gives feedback (reward)
- RL optimizer (GRPO or similar) updates LLM