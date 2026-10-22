# Contributing

secateurs is an early experiment in UltraBERT-inspired pruning. The intended
library is still taking shape. Read the [research scope](docs/research.md),
[experiment record](docs/experiments.md) and [roadmap](docs/roadmap.md) before
assuming that a paper component or model architecture is supported.

Install the locked environment and run the checks:

```bash
uv sync --locked --extra sentence-transformers
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Add `--extra training` to exercise recovery examples. Keep default tests offline
and small; network downloads and expensive evaluations belong in opt-in examples.

Structural transformations need dense-reference numerical equivalence, source
immutability, backward-pass, invalid-input and save/reload tests. Shape-only tests
are insufficient. Document the supported architecture, tensor layout, scoring
method, serialization and limitations. Attribute external methods and code.

Keep calibration/training, development selection and final evaluation separate.
Report hardware and software conditions for benchmarks, including negative results.
Do not infer quality or speed from parameter counts alone.

Submit minimal reproductions with synthetic inputs. Exclude credentials, personal
paths, confidential datasets, model weights and generated caches from changes.

Separate semantic steps with blank lines and give each function or class a clear
responsibility. Use Google-style docstrings: a concise summary, relevant Args,
Returns or Raises sections, and short examples where they clarify usage. Document
implementation constraints near the code they affect; keep research history in
the research documents. State deviations from cited methods explicitly and keep
implemented features, preliminary evidence and planned work distinct. Tests
should check individual properties with shared fixtures for setup. Examples should each teach one main workflow step, separate
argument parsing from execution, and avoid running work on import.

Annotate callable parameters and return values, including fixtures and example
helpers. Prefer concrete model, tensor and array types; use protocols for supported
interfaces and `object` for heterogeneous report values. Ruff checks annotation
coverage. Keep optional dependencies out of runtime imports used only for typing.
The package ships a `py.typed` marker for downstream type checkers.

English documents have Japanese counterparts in `README.ja.md` and `docs/ja/`.
Update both when changing commands, numerical results, supported scope or limits.
Keep language links working and preserve executable examples across translations.

The repository's `.gitignore` excludes generated models, data, reports, archives,
local credentials and caches so ordinary source changes can be staged with
`git add .`. Review the staged files before committing. Ignore rules do not
remove files that are already tracked or credentials embedded in source code.
