"""Train a student to match precomputed teacher sentence embeddings with MSE."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer import losses

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class RecoveryArgs(argparse.Namespace):
    """Store teacher, student, corpus and recovery-training options."""

    student: str
    texts: str
    teacher: str
    output: str
    epochs: float
    seed: int
    teacher_revision: str | None


def parse_args() -> RecoveryArgs:
    """Parse command-line options.

    Returns:
        RecoveryArgs: Parsed training and checkpoint options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument(
        "--texts", required=True, help="Training data only; include task prefixes"
    )
    parser.add_argument("--teacher", default="cl-nagoya/ruri-v3-310m")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-revision")

    return parser.parse_args(namespace=RecoveryArgs())


def load_training_texts(path: str | Path) -> list[str]:
    """Read nonempty training lines from a UTF-8 file.

    Args:
        path: UTF-8 training file with one sentence per line.

    Returns:
        list[str]: Nonempty training sentence entries.
    """
    texts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not texts:
        raise ValueError("No training texts")

    return texts


def teacher_embeddings(
    args: RecoveryArgs, texts: list[str]
) -> tuple[NDArray[np.float32], str | None]:
    """Encode raw teacher embeddings and return the pinned revision.

    Args:
        args: Parsed command-line options.
        texts: Input texts with any intended task prefixes.

    Returns:
        tuple: Raw teacher embeddings and resolved checkpoint revision.
    """
    teacher_revision = args.teacher_revision
    if teacher_revision is None and args.teacher == "cl-nagoya/ruri-v3-310m":
        teacher_revision = "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"

    teacher = SentenceTransformer(
        args.teacher,
        revision=teacher_revision,
        device="cpu",
        model_kwargs={"attn_implementation": "sdpa"},
    )

    labels = teacher.encode(texts, normalize_embeddings=False, show_progress_bar=True)
    del teacher

    return labels, teacher_revision


def train_student(args: RecoveryArgs, data: Dataset) -> dict[str, float]:
    """Train the student with embedding MSE and save its weights.

    Args:
        args: Parsed command-line options.
        data: Training dataset with sentence and teacher label columns.

    Returns:
        dict: Training metrics for the saved recovered student.
    """
    student = SentenceTransformer(
        args.student, device="cpu", model_kwargs={"attn_implementation": "sdpa"}
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        use_cpu=True,
        report_to=[],
        save_strategy="no",
        seed=args.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=student,
        args=training_args,
        train_dataset=data,
        loss=losses.MSELoss(student),
    )
    training_result = trainer.train()
    student.save(args.output)

    return training_result.metrics


def write_report(
    args: RecoveryArgs,
    texts: list[str],
    revision: str | None,
    metrics: Mapping[str, float],
) -> None:
    """Write the recovery configuration and training measurements.

    Args:
        args: Parsed command-line options.
        texts: Input texts with any intended task prefixes.
        revision: Resolved teacher checkpoint revision.
        metrics: Measured training statistics.
    """
    Path(args.output, "recovery_report.json").write_text(
        json.dumps(
            {
                "teacher": args.teacher,
                "student": args.student,
                "epochs": args.epochs,
                "training_texts": len(texts),
                "seed": args.seed,
                "teacher_revision": revision,
                "metrics": metrics,
                "loss": "MSELoss (raw embeddings)",
                "learning_rate": 2e-5,
                "batch_size": 4,
                "training_sha256": hashlib.sha256(
                    Path(args.texts).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Generate teacher labels, train the student and record recovery settings."""
    args = parse_args()
    torch.set_num_threads(4)

    texts = load_training_texts(args.texts)
    labels, revision = teacher_embeddings(args, texts)
    data = Dataset.from_dict({"sentence": texts, "label": labels.tolist()})

    metrics = train_student(args, data)
    write_report(args, texts, revision, metrics)


if __name__ == "__main__":
    main()
