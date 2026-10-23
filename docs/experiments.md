# Experiment record

English | [日本語](ja/experiments.md)

Status: 2026-10-24. This is an early record of an adapted pruning experiment,
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

| Task | Split | Original | Pruned and recovered | Delta, points |
| --- | --- | ---: | ---: | ---: |
| JaCWIRRetrieval | test | 84.027 | 83.950 | -0.077 |
| MIRACLRetrieval | dev | 72.310 | 64.454 | -7.856 |
| MrTidyRetrieval | test | 61.638 | 56.602 | -5.036 |
| MultiLongDocReranking | test | 93.495 | 90.961 | -2.534 |
| MultiLongDocRetrieval | dev | 44.327 | 42.542 | -1.785 |
| MultiLongDocRetrieval | test (supplemental) | 44.019 | 40.276 | -3.743 |

These pairs used float32 with TF32 enabled, SDPA, a maximum sequence length of
8,192, a batch cap of 64 and length-dependent batching with an attention-cell
budget of 4,194,304. Long inputs can use a batch size of 1. The environment used
PyTorch 2.14.1+cu130, Transformers 4.57.6, Sentence Transformers 5.7.0,
datasets 5.0.1, MTEB 2.26.3 and JMTEB 2.1.0.

All results use Japanese data. JaCWIR used dataset revision
`abf6d5fb6759ad516f998bd887b75420d595672e`, 4,000 queries, 513,107 corpus
documents, corpus chunks of 1,024 and top-k 1,000. MultiLongDocReranking used
revision `ad09ce14c17bce6edae151b7f6ef12e15d91dbf3`, 200 queries and a candidate
corpus of 1,581 documents. MultiLongDocRetrieval used revision
`837028901907a7d419b4ab906f28e011ce1cc824`, 200 queries and 10,000 corpus
documents per split, corpus chunks of 1,024 and top-k 1,000. JMTEB's summary
selects dev for this task; test is reported separately as supplemental evidence.
The two splits are not two independent tasks and are not averaged together.
Queries and documents used the prefixes `検索クエリ: ` and `検索文書: ` respectively.

MIRACLRetrieval used dev, 860 queries and 6,953,614 corpus documents, at
revision `9c09abc13478308c27598f350e31d8f06b9b5481`. MrTidyRetrieval used the
Japanese test subset, 720 queries and 7,000,027 corpus documents, at revision
`fc24a3ce8f09746410daee3d5cd823ff7a0675b7`. Both pairs used the inference settings
above, corpus chunks of 1,024 and top-k 1,000. Completed search checkpoints
confirm that every corpus document was processed; corpus fingerprints and query
order match across the original and recovered conditions. The evaluation source
hashes, library versions and recovered checkpoint hashes also match the earlier
runs.

These two-condition observations show quality regressions on MIRACL, Mr.TyDi,
long-document retrieval and reranking. The matched pruned-only comparisons below
separate checkpoint changes on five tasks. Pruned-only results for MIRACL,
Mr.TyDi and long-document retrieval are still missing. These runs do not
establish a general speedup or a full-suite JMTEB score. Results collected with
different inference settings, including batch-16 experiments, are not combined
into an aggregate.

## Three-condition pilot panel

A fixed panel evaluated the original, activation-pruned and recovered checkpoints
under the same A100 inference settings listed above. Each checkpoint was loaded
once and reused across JSTS, JSICK and JaGovFaqsRetrieval. All nine evaluations
completed; the evaluation controller took about 298 seconds, excluding setup,
control restoration and export. This is not an isolated encoding-latency measure.
Scores are the named metric multiplied by 100. Recovery delta is recovered minus
pruned, in points.

| Task | Split | Metric | Original | Pruned only | Recovered | Recovery delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| JSTS | validation | cosine Spearman | 83.5880 | 83.5490 | 85.3377 | +1.7887 |
| JSICK | test | cosine Spearman | 77.9658 | 78.7879 | 79.6457 | +0.8578 |
| JaGovFaqsRetrieval | test | nDCG@10 | 77.3160 | 76.8680 | 76.9920 | +0.1240 |

The dataset revisions were `5bac629e25799df4c9c80a6a5db983d6cba9e77d` (JSTS),
`729cfe4a16d3c2b61c6aa9f9f6c8a96bb5512868` (JSICK) and
`0726b2af53c907628067871e2e7c84e0d8e099c2` (JaGovFaqsRetrieval).
The retrieval run processed all 22,794 documents for 2,048 queries under each
condition; corpus identity and query order matched. Restored pruning tensors
matched the historical pruned checkpoint, while the recovered checkpoint was
held fixed. Tokenizer rules, effective encodings and model pipeline settings
were checked for compatibility. The historical recovery report did not record
a cryptographic parent-weight hash, so this comparison does not retroactively
prove the recovery checkpoint's training lineage.

In this panel, recovery raises both sentence-similarity scores. On JaGovFaqs it
recovers 0.1240 of the 0.4480-point pruning loss, leaving a 0.3240-point deficit
against the original. This is a task-dependent observation from one run per
condition. JSTS validation is separate from the training split but remains in
the same dataset used for calibration and recovery. The panel does not establish
statistical significance or explain the MIRACL, Mr.TyDi or long-document
retrieval regressions. Separate reranking diagnostics follow below. Random/weight
controls, width sweeps and repeated runs remain pending. Metrics are not averaged
across the three tasks.

## Three-condition reranking diagnostics

A separate diagnostic run evaluated JQaRA and MultiLongDocReranking with the same
three checkpoints, source hashes, library versions and inference settings as the
pilot panel. Each model was loaded once and reused across both tasks. All six
evaluations completed; the evaluation controller took about 1,407 seconds,
excluding setup, control restoration and export. Scores are the named metric
multiplied by 100; recovery delta is recovered minus pruned, in points.

| Task | Split | Metric | Original | Pruned only | Recovered | Recovery delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| JQaRAReranking | test | MAP@1000 | 63.469 | 58.586 | 58.498 | -0.088 |
| MultiLongDocReranking | test | nDCG@10 | 93.495 | 90.888 | 90.961 | +0.073 |

JQaRA used dataset revision `63254fa512e133e6300e35b6f9175c32c6cc8455`, subset
`default`. MultiLongDocReranking used the revision reported above, subset `ja`.
Its original and recovered scores agree with the earlier two-condition run at
the reported precision. Preparation records verified the same frozen pruning
plan, restored tensor signature, recovered checkpoint and compatible tokenizer.

In these fixed-checkpoint comparisons, pruning accounts for a 4.883-point drop
on JQaRA and a 2.607-point drop on long-document reranking. Recovery changes them
by -0.088 and +0.073 points, respectively, leaving deficits of 4.971 and 2.534
points against the original. The current small-corpus embedding-MSE recipe does
not restore reranking quality, despite the sentence-similarity gains in the
pilot panel. This finding applies to these checkpoints and tasks; it does not
establish the effectiveness of pruning or distillation methods in general.

These tasks were selected for diagnosis after observing regressions, so they
are not an independent holdout. Each condition has one evaluation; small recovery
deltas are not claims of statistical significance. The two notebooks complete
15 evaluations across five distinct tasks, not a three-condition full-suite
JMTEB evaluation. Pruned-only MIRACL, Mr.TyDi and long-document retrieval results,
selection-policy controls, repeated runs and reproducible public evaluation
remain outstanding. Do not use these test results to tune widths or recovery
hyperparameters and then present the same tests as untouched final evidence.

The complete evaluation driver and raw run archives are not included in this
source release. This table is a preliminary experiment log, not a benchmark that
can yet be independently reproduced from this checkout alone. Publishing a
reproducible evaluation entry point and shareable results is a pending
[roadmap](roadmap.md) item. The development lock covers the source examples;
it does not include the full JMTEB evaluation environment.
