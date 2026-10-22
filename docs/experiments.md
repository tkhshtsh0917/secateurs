# Experiment record

English | [日本語](ja/experiments.md)

Status: 2026-10-22. This is an early record of an adapted pruning experiment,
not a completed reproduction of UltraBERT. The
[research comparison](research.md) identifies the methodological differences.

## Reference pilot

The reference model is `cl-nagoya/ruri-v3-310m`, pinned to revision
`18b60fb8c2b9df296fb4212bb7d23ef94e579cd3`. The pilot keeps all 25 layers, the
768-dimensional residual stream and the existing sequence-length limit. It
reduces each FFN intermediate width from 3,072 to 2,304, selecting neurons
separately for each layer by mean absolute gated activation.

Calibration and recovery use 256 sentence entries from the first 128 pairs of
the JSTS training split, revision `5bac629e25799df4c9c80a6a5db983d6cba9e77d`.
This deliberately small corpus is a pilot, not a representative training mixture
or a limit imposed by the library. Calibration and recovery share this training
corpus; final evaluation must use separate data. The recovery teacher is the
original model. The student trains for one epoch with raw-embedding MSE, batch
size 4, learning rate 2e-5 and seed 42.

The resulting parameter count is 314,611,968 to 270,375,168, a reduction of about
14.06%. The 25% FFN-width reduction does not imply a 25% reduction in the entire
model, and neither number establishes a latency improvement.

## Run the source examples

From a source checkout, the following commands prepare the pilot data, select
neurons, save a student, train embedding recovery and verify the saved encoder:

```bash
uv sync --locked --extra training
uv run python examples/prepare_data.py --output data/train.txt
uv run python examples/ruri_pilot.py --texts data/train.txt --target-width 2304 --policy activation --seed 0 --output artifacts/student
uv run python examples/distill.py --student artifacts/student --texts data/train.txt --epochs 1 --seed 42 --output artifacts/recovered
uv run python examples/verify_saved.py --model artifacts/recovered --output reports/recovered-verification.json
```

Use a corpus filename other than `.json`; that suffix is reserved for the
provenance sidecar, and conflicting corpus paths are rejected before writing.

These examples download the pinned source model and training dataset and run on
CPU. They write provenance and measurements alongside the generated artifacts.
Use `--policy random` or `--policy weight` with a separate output directory for
the selection controls. Comparable recovery and evaluation are still required
before drawing conclusions about those policies.

This workflow reproduces the pilot steps, not the full downstream evaluation.
The verification script checks loading, dimensions and normalization on a few
texts; it does not run JMTEB or exercise a full 8,192-token input.

## Preliminary quality observations

The following completed pairs used the same source model and recovered student,
one run per model, on an NVIDIA A100-SXM4-40GB. Scores are nDCG@10 multiplied by
100; the delta is recovered minus original.

| Task | Original | Pruned and recovered | Delta, points |
| --- | ---: | ---: | ---: |
| JaCWIRRetrieval | 84.027 | 83.950 | -0.077 |
| MultiLongDocReranking | 93.495 | 90.961 | -2.534 |

These pairs used float32 with TF32 enabled, SDPA, a maximum sequence length of
8,192, a batch cap of 64 and length-dependent batching with an attention-cell
budget of 4,194,304. Long inputs can use a batch size of 1. The environment used
PyTorch 2.14.1+cu130, Transformers 4.57.6, Sentence Transformers 5.7.0,
datasets 5.0.1, MTEB 2.26.3 and JMTEB 2.1.0.

Both tasks used Japanese test data. JaCWIR used dataset revision
`abf6d5fb6759ad516f998bd887b75420d595672e`, 4,000 queries, 513,107 corpus
documents, corpus chunks of 1,024 and top-k 1,000. MultiLongDocReranking used
revision `ad09ce14c17bce6edae151b7f6ef12e15d91dbf3`, 200 queries and a candidate
corpus of 1,581 documents. Queries and documents used the prefixes `検索クエリ: `
and `検索文書: ` respectively.

These observations show a quality regression on long-document reranking. They do
not isolate the effect of pruning from the effect of recovery: matched
pruned-only results and complete controls are still needed. They also do not
establish a general speedup or a full-suite JMTEB score. Results collected with
different inference settings, including batch-16 experiments, are not combined
into an aggregate.

The complete evaluation driver and raw run archives are not included in this
source release. This table is a preliminary experiment log, not a benchmark that
can yet be independently reproduced from this checkout alone. Publishing a
reproducible evaluation entry point and shareable results is a pending
[roadmap](roadmap.md) item. The development lock covers the source examples;
it does not include the full JMTEB evaluation environment.
