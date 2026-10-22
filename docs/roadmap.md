# Roadmap

English | [日本語](ja/roadmap.md)

The project is at the prototype stage. The goal is to turn an experiment in
UltraBERT-inspired pruning into a convenient library that others can use without
reconstructing the research workflow. The stages below are direction and exit
criteria, not release dates or a promise of quality or speed gains.

## 1. Establish the first transformation

Implemented: ModernBERT FFN intermediate-width pruning, serializable per-layer
plans, activation scoring, seeded random selection, an optional Sentence
Transformers adapter and standard model saving/loading.

Offline tests cover numerical equivalence to a dense reference, source-model
preservation, gradients, rejected inputs and reload. These establish the narrow
transformation contract, not application quality.

## 2. Complete and expose the experiments

In progress: downstream quality and workload-specific performance evaluation.
The [experiment record](experiments.md) includes preliminary pairs and regressions.

Before calling this stage complete, publish a reproducible evaluation entry point,
pin the environment and data revisions, report coverage, and compare original,
pruned-only and recovered models under the same conditions. Add random and
weight-based controls, width sweeps and repeated runs. Report memory and encoding
latency separately from model loading, data loading and checkpoint storage.

## 3. Make the supported workflow convenient

Provide a documented path from calibration data through plan creation, pruning,
optional recovery, verification and evaluation. Decide which recovery operations
belong in a reusable API, add suitable validation and report formats, and replace
model-specific assumptions before presenting the examples as general recipes.

The exit criterion is that a new user can run the supported workflow from the
documentation, save a standard model and assess its trade-offs without bespoke
research scripts. Stable APIs and broader support remain future work.

## 4. Extend recipes with separate evidence

Potential extensions include the paper's exact importance aggregation, hidden
channel and attention pruning, additional encoder architectures, and alternative
distillation objectives. Each needs its own documented contract and tests.

Token-logit distillation and UltraBERT's mixed-masking pretraining would be new
training work, not a renaming of the current embedding-MSE example. Any closer
reproduction must state exactly which paper components and evaluation protocols
it implements. A full UltraBERT pretraining reproduction is not a committed
deliverable of the initial library.
