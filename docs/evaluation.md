# Evaluation status and protocol

English | [日本語](ja/evaluation.md)

Full benchmark validation is in progress. This development snapshot makes no
aggregate JMTEB score, universal speed claim, or quality-preservation claim.
The numerical tests establish transformation and serialization properties, not
application quality. Benchmarks are not required to inspect or contribute to the
experimental implementation.

The [experiment record](experiments.md) describes the pilot and selected completed
pairs, including negative results. It also identifies the evaluation tooling and
evidence not yet available from this checkout. It is a progress record, not a
completed reproduction or a benchmark release. See the [roadmap](roadmap.md) for
the remaining reproducibility work.

For a reproducible evaluation, pin model, dataset and evaluator revisions. Record
calibration provenance, scoring policy, target widths, recovery settings and seeds.
Use separate training/calibration, development and final evaluation data.

Compare original, pruned and recovered models with identical tokenizer, prompts,
normalization, precision, attention backend and evaluation task definitions.
Report task coverage and per-task regressions; incomplete task subsets are not
full-suite aggregate scores. Useful controls include random and weight-based
selection and comparable recovery budgets.

Measure latency and throughput separately from quality. Record hardware, software,
batch size, actual sequence lengths, warmups, repetitions and memory. Keep startup,
loading, encoding and result-storage time distinguishable. Changing batch size can
change floating-point outputs, so identify results and resume checkpoints by their
full inference settings.

Do not redistribute datasets or derived model weights without checking their own
licenses. Generated artifacts and benchmark caches are excluded from source
publication.
