"""Exercise pruning examples without model or dataset downloads."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceEncoder
    from transformers import ModernBertModel

SentenceTransformer = pytest.importorskip("sentence_transformers").SentenceTransformer

ROOT = Path(__file__).parents[1]


def load_example(name: str) -> ModuleType:
    """Import an example without invoking its command-line entry point."""
    path = ROOT / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"secateurs_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@pytest.mark.parametrize("policy", ["activation", "weight", "random"])
def test_pruning_pilot_saves_reloadable_student(
    source: SentenceEncoder,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
) -> None:
    """Run each selection policy through the complete offline pruning example."""
    source_path = tmp_path / "source-model"
    source.save(str(source_path))
    module = load_example("ruri_pilot")

    def load_model(path: str | Path, **kwargs: Any) -> SentenceEncoder:
        """Resolve the reference model to the local offline fixture."""
        if str(path) == "cl-nagoya/ruri-v3-310m":
            path = source_path
        return SentenceTransformer(
            str(path), **{k: v for k, v in kwargs.items() if k != "revision"}
        )

    monkeypatch.setattr(module, "SentenceTransformer", load_model)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ruri_pilot.py",
            "--output",
            "student",
            "--target-width",
            "12",
            "--policy",
            policy,
        ],
    )

    module.main()
    loaded = SentenceTransformer("student", device="cpu")
    report = json.loads(Path("student/pruning_report.json").read_text())

    assert loaded[0].model.config.intermediate_size == 12
    assert report["policy"] == policy
    assert report["parameters_after"] < report["parameters_before"]
    assert report["reload_max_error"] == 0
    assert report["calibration_texts"] == 4
    assert Path("reports/student-pilot.json").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="CPU benchmark uses Unix resource")
def test_cpu_benchmark_outputs_finite_measurements(
    source: SentenceEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measure a small saved model through the CPU benchmark entry point."""
    path = tmp_path / "encoder"
    source.save(str(path))
    module = load_example("benchmark")
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark.py",
            "--model",
            str(path),
            "--length",
            "16",
            "--output",
            str(output),
        ],
    )

    module.main()
    report = json.loads(output.read_text())

    assert len(report["seconds"]) == 7
    assert np.isfinite(report["median_seconds"]) and report["median_seconds"] > 0
    assert report["sentences_per_second"] > 0
    assert report["actual_token_lengths"] == [16] * 8


def test_prepare_data_writes_corpus_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save 256 training entries with a matching dataset revision and file hash."""
    import hashlib
    from types import SimpleNamespace

    pytest.importorskip("datasets")
    module = load_example("prepare_data")
    rows = [{"sentence1": "日本", "sentence2": "東京"}] * 128
    dataset = SimpleNamespace(select=lambda indices: [rows[i] for i in indices])
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: dataset)
    output = tmp_path / "train.txt"
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--output", str(output)])

    module.main()
    sidecar = json.loads(output.with_suffix(".json").read_text())

    assert len(output.read_text().splitlines()) == 256
    assert sidecar["split"] == "train"
    assert sidecar["revision"] == module.REVISION
    assert sidecar["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_distillation_saves_recovered_model_and_teacher_revision(
    source: SentenceEncoder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train a small student offline and preserve recovery provenance."""
    pytest.importorskip("datasets")
    pytest.importorskip("accelerate")
    module = load_example("distill")
    teacher = tmp_path / "teacher"
    student = tmp_path / "student"
    output = tmp_path / "recovered"
    texts = tmp_path / "train.txt"
    texts.write_text("日本 首都\n東京\n", encoding="utf-8")
    source.save(str(teacher))

    from secateurs import make_plan
    from secateurs.sentence_transformers import prune_sentence_transformer

    backbone = cast("ModernBertModel", source[0].model)
    prune_sentence_transformer(source, make_plan(backbone, 12)).save(str(student))

    def load_model(path: str | Path, **kwargs: Any) -> SentenceEncoder:
        """Resolve the reference teacher to the local offline fixture."""
        if str(path) == "cl-nagoya/ruri-v3-310m":
            path = teacher
        return SentenceTransformer(
            str(path), **{k: v for k, v in kwargs.items() if k != "revision"}
        )

    monkeypatch.setattr(module, "SentenceTransformer", load_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "distill.py",
            "--student",
            str(student),
            "--texts",
            str(texts),
            "--output",
            str(output),
        ],
    )

    module.main()
    report = json.loads((output / "recovery_report.json").read_text())
    recovered = SentenceTransformer(str(output), device="cpu")

    assert report["teacher_revision"] == "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"
    assert report["training_texts"] == 2
    assert np.isfinite(report["metrics"]["train_loss"])
    assert recovered[0].model.config.intermediate_size == 12


@pytest.mark.parametrize("suffix", [".json", ".JSON"])
def test_prepare_data_rejects_colliding_sidecar_without_overwriting(
    tmp_path: Path, suffix: str
) -> None:
    """Reject a colliding sidecar path before changing the existing corpus."""
    pytest.importorskip("datasets")
    module = load_example("prepare_data")
    output = tmp_path / f"train{suffix}"
    output.write_text("keep original", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance sidecar"):
        module.write_corpus(output, ["日本", "東京"])

    assert output.read_text(encoding="utf-8") == "keep original"


@pytest.mark.parametrize(
    "case", ["nonfinite", "shape", "zero", "normalization", "dimension"]
)
def test_saved_model_verification_rejects_invalid_embeddings(case: str) -> None:
    """Reject invalid values, dimensions and inconsistent normalization."""
    from types import SimpleNamespace

    module = load_example("verify_saved")
    raw, normalized = np.array([[2.0, 0.0]]), np.array([[1.0, 0.0]])
    dimension: int | None = 2
    if case == "nonfinite":
        raw[0, 0] = np.nan
    elif case == "shape":
        normalized = np.ones((1, 3))
    elif case == "zero":
        raw = np.zeros((1, 2))
    elif case == "normalization":
        normalized = np.array([[0.0, 1.0]])
    else:
        dimension = None

    model = SimpleNamespace(
        encode=lambda texts, **kwargs: (
            normalized if kwargs["normalize_embeddings"] else raw
        ),
        get_embedding_dimension=lambda: dimension,
    )

    with pytest.raises(ValueError):
        module.check_embeddings(model, ["日本"])


def test_saved_model_verification_still_checks_under_python_optimization() -> None:
    """Reject nonfinite embeddings even when Python disables assertions."""
    import subprocess

    code = """
import importlib.util
from types import SimpleNamespace
import sys
import numpy as np

spec = importlib.util.spec_from_file_location("verify_example", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
model = SimpleNamespace(
    encode=lambda *args, **kwargs: np.array([[np.nan, 0.0]]),
    get_embedding_dimension=lambda: 2,
)
try:
    module.check_embeddings(model, ["日本"])
except ValueError:
    print("invalid embeddings rejected")
else:
    raise RuntimeError("invalid embeddings were accepted")
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code, str(ROOT / "examples/verify_saved.py")],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "invalid embeddings rejected" in result.stdout
