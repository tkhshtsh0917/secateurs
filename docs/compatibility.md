# Compatibility

English | [日本語](ja/compatibility.md)

The initial Python range is 3.10–3.12. CI defines tests, builds and isolated wheel
installation across these versions. A CI definition does not establish that a
particular hosted run has passed. A separate Python 3.12 CI job installs the
training extra to exercise recovery; without it, optional training tests can be
skipped. Both jobs disable Hugging Face model and dataset network access for tests.

The development lock exercises PyTorch 2.14.1, Transformers 4.57.6 and optional
Sentence Transformers 5.7.0. Broader package dependency ranges are installation
constraints, not a tested matrix. The Sentence Transformers adapter targets its
`Transformer.model` integration contract. Tokenizer reload for ruri requires
SentencePiece and protobuf, included in the optional extras.

The numerical test suite uses small CPU models with ordinary floating-point
weights and eager/SDPA attention. Additional backends, compiled models,
quantized weights and distributed wrappers require independent validation.
Backend support for inference does not establish support for calibration or
structural transformation on that backend.

Text files are read and written as UTF-8. The CPU benchmark example requires
Linux/macOS because it uses the Unix `resource` module; it is not a Windows
benchmark command. The library's transformation contract is independent of that
example's platform restriction.

A saved student uses ordinary Transformers/Sentence Transformers serialization.
Only ModernBertModel and uniform FFN intermediate widths are supported; other
architectures and hidden-size or attention pruning are outside this snapshot.

The adapter expects the first Sentence Transformer module to expose its
registered backbone as `Transformer.model`. It copies the encoder and retains
tokenizer, pooling and prompt settings. See the [research comparison](research.md)
for scope and the [roadmap](roadmap.md) for planned development.
