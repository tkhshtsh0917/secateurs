"""Load a saved encoder and check embeddings and normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from numpy.typing import NDArray


class VerificationArgs(argparse.Namespace):
    """Store the encoder path and verification-report destination."""

    model: str
    output: str


def parse_args() -> VerificationArgs:
    """Parse command-line options.

    Returns:
        VerificationArgs: Parsed encoder and report paths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)

    return parser.parse_args(namespace=VerificationArgs())


def check_embeddings(
    model: SentenceTransformer, texts: list[str]
) -> tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """Check finite embeddings, reported dimensions and normalization.

    Args:
        model: Loaded sentence encoder.
        texts: Input texts with any intended task prefixes.

    Returns:
        tuple: Raw embeddings, their norms and normalization error.

    Raises:
        ValueError: Embeddings are nonfinite, have incorrect dimensions,
            or disagree with explicit normalization.
    """
    raw = model.encode(
        texts,
        batch_size=1,
        normalize_embeddings=False,
        convert_to_numpy=True,
        convert_to_tensor=False,
    )
    normalized = model.encode(
        texts,
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        convert_to_tensor=False,
    )
    if not np.isfinite(raw).all() or not np.isfinite(normalized).all():
        raise ValueError("Encoder returned nonfinite embeddings")

    dimension = model.get_embedding_dimension()
    if dimension is None:
        raise ValueError("Model must report its sentence embedding dimension")
    if raw.shape != (len(texts), dimension) or normalized.shape != raw.shape:
        raise ValueError("Embedding shape differs from the reported dimension")

    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if not (norms > 0).all():
        raise ValueError("Cannot normalize zero-length embeddings")

    error = float(np.abs(raw / norms - normalized).max())
    if error >= 1e-5:
        raise ValueError(f"Embedding normalization disagreement: {error}")

    return raw, norms, error


def build_report(
    model: SentenceTransformer,
    path: Path,
    texts: list[str],
    raw: NDArray[np.floating],
    norms: NDArray[np.floating],
    error: float,
) -> dict[str, object]:
    """Summarize the encoder files and verified embeddings.

    Args:
        model: Loaded sentence encoder.
        path: Input file or model directory.
        texts: Input texts with any intended task prefixes.
        raw: Unnormalized sentence embeddings.
        norms: L2 norm of each raw embedding.
        error: Maximum error between explicit and model normalization.

    Returns:
        dict: JSON-compatible measurements and configuration.
    """
    return {
        "model": str(path),
        "device": "cpu",
        "backend": "sdpa",
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_file_bytes": sum(
            p.stat().st_size for p in path.rglob("*") if p.is_file()
        ),
        "safetensors_bytes": sum(p.stat().st_size for p in path.rglob("*.safetensors")),
        "token_lengths": model.preprocess(list(texts))["attention_mask"]
        .sum(-1)
        .tolist(),
        "embedding_shape": list(raw.shape),
        "raw_norms": norms[:, 0].tolist(),
        "normalization_max_error": error,
        "finite": True,
        "full_8192_tokens_tested": False,
    }


def main() -> None:
    """Load an encoder and write its embedding validation report."""
    args = parse_args()
    torch.set_num_threads(4)

    path = Path(args.model)
    model = SentenceTransformer(
        str(path), device="cpu", model_kwargs={"attn_implementation": "sdpa"}
    )

    texts = [
        "検索クエリ: 日本の首都は？",
        "検索文書: 東京は日本の首都です。",
        "検索文書: " + "これは日本語の検索文書です。" * 100,
    ]

    raw, norms, error = check_embeddings(model, texts)
    report = build_report(model, path, texts, raw, norms, error)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
