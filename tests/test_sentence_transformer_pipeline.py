"""Check Sentence Transformer pruning, serialization and saved-model verification."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from secateurs import make_plan
from secateurs.sentence_transformers import prune_sentence_transformer

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import ModernBertModel

st = pytest.importorskip("sentence_transformers")


@pytest.fixture
def texts() -> list[str]:
    """Provide three texts with different token counts."""
    return ["東京", "日本 首都", "東京 首都 日本"]


@pytest.fixture
def student(source: SentenceTransformer) -> SentenceTransformer:
    """Prune an independent copy of the source encoder."""
    backbone = cast("ModernBertModel", source[0].model)
    return prune_sentence_transformer(source, make_plan(backbone, 12))


def test_pruning_preserves_source_and_prompts(
    source: SentenceTransformer, texts: list[str]
) -> None:
    """Verify pruning preserves source and prompts."""
    before = source.encode(texts, prompt_name="query", normalize_embeddings=True)

    backbone = cast("ModernBertModel", source[0].model)
    student = prune_sentence_transformer(source, make_plan(backbone, 12))
    after = student.encode(texts, prompt_name="query", normalize_embeddings=True)

    assert cast("ModernBertModel", student[0].model).config.intermediate_size == 12
    assert backbone.config.intermediate_size == 24
    assert student.prompts == source.prompts
    assert np.isfinite(after).all() and not np.allclose(before, after)
    np.testing.assert_allclose(
        source.encode(texts, prompt_name="query", normalize_embeddings=True), before
    )


def test_save_reload_preserves_embeddings_and_prompts(
    student: SentenceTransformer, texts: list[str], tmp_path: Path
) -> None:
    """Verify save reload preserves embeddings and prompts."""
    before = student.encode(texts, prompt_name="query", normalize_embeddings=True)
    path = tmp_path / "student"
    student.save(str(path))

    reloaded = st.SentenceTransformer(
        str(path), device="cpu", model_kwargs={"attn_implementation": "eager"}
    )
    after = reloaded.encode(texts, prompt_name="query", normalize_embeddings=True)

    assert reloaded.prompts == student.prompts
    np.testing.assert_allclose(after, before, atol=1e-6)


def test_saved_model_cli_uses_reported_dimension(
    student: SentenceTransformer, tmp_path: Path
) -> None:
    """Verify saved model cli uses reported dimension."""
    path = tmp_path / "student"
    student.save(str(path))
    report = tmp_path / "verification.json"

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "examples/verify_saved.py"),
            "--model",
            str(path),
            "--output",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(report.read_text(encoding="utf-8"))

    assert result["embedding_shape"] == [3, 16]
    assert result["finite"]
