"""Calibrate, prune and verify a saved ruri encoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import torch
from sentence_transformers import SentenceTransformer

from secateurs import FFNPlan, activation_scores, make_plan
from secateurs.sentence_transformers import prune_sentence_transformer

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from transformers import ModernBertModel
    from transformers.models.modernbert.modeling_modernbert import ModernBertMLP


class PilotArgs(argparse.Namespace):
    """Store calibration, selection and output options for the pruning pilot."""

    output: str
    texts: str | None
    target_width: int
    seed: int
    timing_repeats: int
    revision: str
    policy: Literal["activation", "random", "weight"]


def parse_args() -> PilotArgs:
    """Parse command-line options.

    Returns:
        PilotArgs: Parsed pruning and measurement options.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/ruri-ffn-2304")
    parser.add_argument("--texts")
    parser.add_argument("--target-width", type=int, default=2304)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timing-repeats", type=int, default=0)
    parser.add_argument(
        "--revision", default="18b60fb8c2b9df296fb4212bb7d23ef94e579cd3"
    )
    parser.add_argument(
        "--policy", choices=["activation", "random", "weight"], default="activation"
    )

    return parser.parse_args(namespace=PilotArgs())


def load_texts(path: str | Path | None) -> list[str]:
    """Read calibration texts or use the four built-in smoke inputs.

    Args:
        path: Optional UTF-8 calibration file; None selects smoke inputs.

    Returns:
        list[str]: Calibration sentence entries.
    """
    texts = [
        "検索クエリ: 日本の首都はどこですか？",
        "検索文書: 日本の首都は東京です。",
        "検索クエリ: 猫は何を食べますか？",
        "検索文書: 猫は肉食の動物です。",
    ]
    if path:
        texts = Path(path).read_text(encoding="utf-8").splitlines()
    if not texts or not any(text.strip() for text in texts):
        raise ValueError("Calibration requires nonempty training texts")

    return texts


def select_scores(
    backbone: ModernBertModel,
    batch: Mapping[str, torch.Tensor],
    policy: Literal["activation", "random", "weight"],
) -> torch.Tensor | None:
    """Compute activation or weight scores, or select the random baseline.

    Args:
        backbone: Source ModernBERT backbone.
        batch: Tokenized calibration inputs including attention_mask.
        policy: Activation, weight or random selection policy.

    Returns:
        torch.Tensor | None: Per-layer scores, or None for random selection.
    """
    width = backbone.config.intermediate_size
    scores: torch.Tensor | None = None
    if policy == "activation":
        scores = activation_scores(
            backbone,
            [
                {k: v[i : i + 8] for k, v in batch.items()}
                for i in range(0, len(batch["input_ids"]), 8)
            ],
        )
    elif policy == "weight":
        mlps = (cast("ModernBertMLP", layer.mlp) for layer in backbone.layers)
        scores = torch.stack(
            [
                mlp.Wi.weight[:width].norm(dim=1)
                * mlp.Wi.weight[width:].norm(dim=1)
                * mlp.Wo.weight.norm(dim=0)
                for mlp in mlps
            ]
        ).detach()

    return scores


def verify_reload(
    model: SentenceTransformer, output: Path, texts: list[str]
) -> tuple[SentenceTransformer, NDArray[np.float32], NDArray[np.float32]]:
    """Compare student embeddings before and after standard reload.

    Args:
        model: Loaded sentence encoder.
        output: Destination path.
        texts: Input texts with any intended task prefixes.

    Returns:
        tuple: Reloaded encoder and student embeddings before and after reload.

    Raises:
        ValueError: Student embeddings change after serialization.
    """
    student = model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, convert_to_tensor=False
    )
    loaded = SentenceTransformer(
        str(output), device="cpu", model_kwargs={"attn_implementation": "sdpa"}
    )
    reloaded = loaded.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, convert_to_tensor=False
    )
    if not torch.allclose(
        torch.from_numpy(student), torch.from_numpy(reloaded), atol=1e-5
    ):
        raise ValueError("Student embeddings changed after reload")

    return loaded, student, reloaded


def measure_encoding(
    model: SentenceTransformer, texts: list[str], repeats: int
) -> list[float]:
    """Collect the requested number of CPU encoding timings.

    Args:
        model: Loaded sentence encoder.
        texts: Input texts with any intended task prefixes.
        repeats: Number of measured encoding calls.

    Returns:
        list[float]: Elapsed seconds for each measured call.
    """
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.encode(texts, normalize_embeddings=True)
        times.append(time.perf_counter() - start)

    return times


def calibration_report(
    args: PilotArgs,
    texts: list[str],
    batch: Mapping[str, torch.Tensor],
    loaded: SentenceTransformer,
) -> dict[str, object]:
    """Record calibration provenance, token counts and scoring settings.

    Args:
        args: Parsed command-line options.
        texts: Input texts with any intended task prefixes.
        batch: Tokenized calibration inputs including attention_mask.
        loaded: Reloaded student encoder.

    Returns:
        dict: Input provenance, token counts and scoring settings.
    """
    return {
        "source": args.texts or "built-in API smoke sentences",
        "file_sha256": hashlib.sha256(Path(args.texts).read_bytes()).hexdigest()
        if args.texts
        else None,
        "parsed_texts_sha256": hashlib.sha256(
            json.dumps(texts, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "extra_prompt": None,
        "prompt_tokens": "included if present in the input text",
        "max_sequence_length": loaded.max_seq_length,
        "valid_tokens": int(batch["attention_mask"].sum()),
        "padded_sequence_length": batch["input_ids"].shape[1],
        "batch_size": 8 if args.policy == "activation" else None,
        "score": {
            "activation": "mean absolute gated activation per valid token",
            "weight": "product of paired Wi-row L2 norms and Wo-column L2 norm",
            "random": "seeded independent uniform scores",
        }[args.policy],
    }


def build_report(
    args: PilotArgs,
    plan: FFNPlan,
    count_before: int,
    loaded: SentenceTransformer,
    embeddings: tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],
    times: list[float],
    calibration: Mapping[str, object],
) -> dict[str, object]:
    """Summarize pruning size, reload agreement and pilot measurements.

    Args:
        args: Parsed command-line options.
        plan: Selected source neuron indices and dimensions.
        count_before: Source parameter count.
        loaded: Reloaded student encoder.
        embeddings: Original, student and reloaded embeddings.
        times: Encoding durations in seconds.
        calibration: Calibration provenance and score settings.

    Returns:
        dict: JSON-compatible measurements and configuration.
    """
    original, student, reloaded = embeddings

    return {
        "scope": "calibration/inference pilot; quality measured separately",
        "policy": args.policy,
        "seed": args.seed,
        "target_width": args.target_width,
        "calibration_texts": len(original),
        "calibration": calibration,
        "model_revision": plan.source_revision,
        "source_width": plan.source_width,
        "runtime": {
            name: version(name)
            for name in ("torch", "transformers", "sentence-transformers")
        },
        "normalized_embeddings": True,
        "backend": "sdpa",
        "parameters_before": count_before,
        "parameters_after": sum(p.numel() for p in loaded.parameters()),
        "reload_max_error": float(abs(student - reloaded).max()),
        "teacher_student_mean_cosine": float((original * student).sum(-1).mean()),
        "student_batch_seconds": times,
        "device": "cpu",
        "threads": 4,
    }


def save_report(output: Path, report: Mapping[str, object]) -> None:
    """Write the pilot report beside the student and in reports.

    Args:
        output: Destination path.
        report: JSON-compatible measurements and settings.
    """
    serialized_report = json.dumps(report, indent=2) + "\n"
    (output / "pruning_report.json").write_text(serialized_report, encoding="utf-8")
    Path("reports").mkdir(parents=True, exist_ok=True)
    (Path("reports") / (output.name + "-pilot.json")).write_text(
        serialized_report, encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def main() -> None:
    """Calibrate, prune and save a student with a reload verification report."""
    args = parse_args()
    torch.set_num_threads(4)
    texts = load_texts(args.texts)

    model = SentenceTransformer(
        "cl-nagoya/ruri-v3-310m",
        revision=args.revision,
        device="cpu",
        model_kwargs={"attn_implementation": "sdpa"},
    )
    backbone = cast("ModernBertModel", model[0].model)
    original = model.encode(texts, normalize_embeddings=True)
    batch = {
        k: v
        for k, v in model.preprocess(texts).items()
        if k in {"input_ids", "attention_mask"}
    }

    scores = select_scores(backbone, batch, args.policy)
    plan = make_plan(backbone, args.target_width, scores, seed=args.seed)
    count_before = sum(p.numel() for p in backbone.parameters())
    model = prune_sentence_transformer(model, plan)
    if sum(p.numel() for p in model.parameters()) >= count_before:
        raise ValueError("Target width must reduce the parameter count")
    del backbone
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    plan.save(output / "pruning_plan.json")

    loaded, student, reloaded = verify_reload(model, output, texts)
    times = measure_encoding(loaded, texts, args.timing_repeats)

    calibration = calibration_report(args, texts, batch, loaded)
    report = build_report(
        args,
        plan,
        count_before,
        loaded,
        (original, student, reloaded),
        times,
        calibration,
    )
    save_report(output, report)


if __name__ == "__main__":
    main()
