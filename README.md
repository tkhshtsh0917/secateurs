# secateurs

English | [日本語](README.ja.md)

Early experiments in **UltraBERT-inspired pruning** for Transformer encoders.

This project is just getting started. It records an attempt to adapt the pruning
and recovery workflow described in [UltraBERT](https://arxiv.org/abs/2610.15028v1)
to existing embedding models, starting with ModernBERT and ruri-v3-310m. It is a
partial reproduction study with deliberate changes, not a reproduction of the
paper's training recipe or reported results.

The long-term goal is a convenient, reusable pruning library with clear recipes,
reliable saving/loading and reproducible evaluation. Today, this is a narrow
prototype: ModernBERT FFN-width pruning, an optional Sentence Transformers adapter,
and an embedding-recovery example. The API may change. Full quality and performance
validation is unfinished, and some evaluated tasks regress after recovery.

## Relationship to UltraBERT

The shared idea is to rank structures by forward activations, physically shrink
their weight matrices while keeping the layers, and train the student to recover
behavior from a teacher. The current implementation differs in important ways:

- It prunes only FFN intermediate neurons. Hidden width and attention are retained.
- Its score is the mean absolute gated activation over valid tokens, rather than
  UltraBERT's sequence-average and batch-L2 aggregation.
- Its recovery example matches sentence embeddings with MSE, rather than token
  prediction logits with KL divergence.
- It does not implement mixed masking ratios, separate LM heads, selective
  distillation or the paper's cascade of progressively smaller encoders.

See [the research comparison](docs/research.md) for the precise scope and sources,
[the experiment record](docs/experiments.md) for the pilot and preliminary results,
and [the roadmap](docs/roadmap.md) for work still required. Parameter reduction is
implemented; general quality preservation and inference speedups are not established.

## Install from source

Python 3.10–3.12 is supported. From a source checkout:

```bash
uv sync --locked --extra sentence-transformers
uv run pytest
```

For the embedding-recovery example, install `--extra training` as well.
The core package uses PyTorch and Transformers; Sentence Transformers is optional.
Model and dataset downloads are opt-in and subject to their own licenses.

## Prune a Sentence Transformer

```python
from sentence_transformers import SentenceTransformer
from secateurs import activation_scores, make_plan
from secateurs.sentence_transformers import prune_sentence_transformer

model = SentenceTransformer(
    "cl-nagoya/ruri-v3-310m",
    revision="18b60fb8c2b9df296fb4212bb7d23ef94e579cd3",
    device="cpu",
    model_kwargs={"attn_implementation": "sdpa"},
)
backbone = model[0].model

features = model.preprocess(["検索文書: 日本の首都は東京です。"])
batch = {key: features[key] for key in ("input_ids", "attention_mask")}

scores = activation_scores(backbone, [batch])
plan = make_plan(backbone, target_width=2304, scores=scores)
student = prune_sentence_transformer(model, plan)

student.save("artifacts/student")
plan.save("artifacts/student/pruning_plan.json")
```

The single sentence illustrates the API. Use representative training/calibration
texts for actual selection, and pin the model revision for reproducibility.
The source model is preserved; creating a copy requires additional memory.
`make_plan(..., scores=None, seed=0)` provides a seeded random baseline.
Plans can be saved and loaded with `FFNPlan.save` and `FFNPlan.load`.

## Supported scope

- ModernBertModel with a uniform target FFN intermediate width across layers.
- Explicit per-layer neuron indices or padding-aware gated-activation scores.
- Standard model configuration and saving/loading; no custom inference kernel.
- Residual width, vocabulary, attention, RoPE and pooling remain unchanged.
- Ordinary floating-point weights, before compilation, quantization, distributed
  wrapping, or optimizer construction.

Keeping the output dimension does not preserve embedding-space compatibility.
Re-encode documents with the student before evaluating retrieval. A reduction in
FFN width is not the same percentage reduction in total model parameters.
Recovery is currently an example script, not a general distillation API. Other
architectures and pruning recipes remain future work.

## Examples and validation

Each example has one main purpose and its own command-line entry point:

| Example | Purpose | Main input and output |
| --- | --- | --- |
| `examples/prepare_data.py` | Prepare pilot training texts | Pinned JSTS train split to a text corpus and provenance sidecar |
| `examples/ruri_pilot.py` | Calibrate and prune a ruri encoder | Calibration texts and a width to a saved student, plan and pilot report |
| `examples/distill.py` | Recover sentence embeddings with MSE | A teacher, student and training texts to a recovered student and training report |
| `examples/verify_saved.py` | Check a saved encoder | A model path to dimension, normalization and loading checks |
| `examples/benchmark.py` | Measure one CPU encoding workload | A model and sequence limit to timing and memory measurements |

The pruning pilot's reload check verifies its own output. Its optional timings
are diagnostics; use the separate benchmark command for a fixed CPU workload,
running each model in a fresh process. Use `--help` for arguments. All examples
require the Sentence Transformers extra; data preparation and recovery require
the training extra. The CPU benchmark uses the Unix `resource` module and runs on
Linux/macOS. Real-model examples require downloads or user-supplied data; default
tests use offline fixtures.

Tests use small offline models and cover dense-reference equivalence, source
immutability, gradients, invalid plans, and serialization.

See [the algorithm contract](docs/pruning.md), [compatibility](docs/compatibility.md),
[recovery](docs/recovery.md), [evaluation policy](docs/evaluation.md),
[contributing](CONTRIBUTING.md), [changelog](CHANGELOG.md), and [MIT license](LICENSE).
