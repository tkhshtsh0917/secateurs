# Embedding recovery example

English | [日本語](ja/recovery.md)

This is a sentence-embedding distillation example for the current prototype.
It does not implement UltraBERT's token-logit distillation or pretraining
objectives; see the [research comparison](research.md). Recovery is provided as
an example script, not a general training API.

Install the training extra and provide a UTF-8 file containing one training text
per line. Include the intended task prefixes in the texts.

```bash
uv sync --locked --extra training
uv run python examples/distill.py --student artifacts/student --texts data/train.txt --output artifacts/recovered
```

The example freezes the teacher through precomputed raw embeddings, then trains
the student with Sentence Transformers MSELoss. It uses CPU, batch size 4,
learning rate 2e-5 and one epoch by default. Pin the teacher revision explicitly
when changing the reference teacher. The example is a starting point, not a
validated optimal recovery schedule or a general training framework.

Keep training and calibration data separate from final evaluation data. A lower
training MSE does not establish restored retrieval rankings. Evaluate the recovered
student, the pruned student and the original under identical conditions.
The [experiment record](experiments.md) documents the small reference pilot and
observed regressions. The [roadmap](roadmap.md) tracks broader recovery support.
