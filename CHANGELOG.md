# Changelog

## Unreleased — 0.0.1.dev0

- Record completed MIRACL and Mr.TyDi retrieval pairs, including quality regressions.
- Illustrate the README pruning example with an animated workflow in both languages.
- Add type hints across the package, examples and tests, distribute a `py.typed`
  marker, and require callable annotations in Ruff checks.
- Credit the UltraBERT authors and provide the original paper's BibTeX in both
  READMEs; record long-document retrieval pairs and the MIRACL baseline by split.
- Reject colliding corpus/sidecar paths and unsupported encoder wrappers; keep
  saved-model verification active under optimized Python.
- Separate offline training CI job for the recovery example.

- Japanese README and translations of all seven documents, with language links.
- Smaller pruning, calibration, example and test responsibilities with
  Google-style docstrings and semantic spacing.
- Ignore rules for generated outputs, archives, caches and local credentials.
- Early UltraBERT-inspired pruning study, with documented differences from the
  paper, a pilot record and a roadmap toward a reusable library.
- Physical ModernBERT FFN-width pruning with serializable per-layer plans.
- Padding-aware activation scoring and deterministic seeded random selection.
- Source-model preservation, plan validation and standard save/reload support.
- Optional Sentence Transformers integration and embedding-recovery examples.
- Offline numerical tests and CI for Python 3.10–3.12.

This development snapshot is available for review and contribution while full
benchmark validation continues. API stability, quality preservation and workload
speedups are not established. Derived checkpoints are not included.
