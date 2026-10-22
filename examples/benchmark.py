"""Measure CPU encoding throughput and peak memory for one model."""

import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def parse_args():
    """Parse command-line options.

    Returns:
        argparse.Namespace: Validated command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--revision")

    return parser.parse_args()


def load_encoder(args):
    """Load the pinned CPU encoder and set its sequence limit.

    Args:
        args: Parsed command-line options.

    Returns:
        SentenceTransformer: CPU encoder with the requested sequence limit.
    """
    revisions = {
        "cl-nagoya/ruri-v3-310m": "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3",
        "cl-nagoya/ruri-v3-130m": "e3114c6ee10dbab8b4b235fbc6dcf9dd4d5ac1a6",
    }
    model = SentenceTransformer(
        args.model,
        revision=args.revision or revisions.get(args.model),
        device="cpu",
        model_kwargs={"attn_implementation": "sdpa"},
    )
    model.max_seq_length = args.length

    return model


def measure_encoding(model, texts):
    """Warm up the encoder and collect seven timed batches.

    Args:
        model: Loaded sentence encoder.
        texts: Input texts with any intended task prefixes.

    Returns:
        list[float]: Elapsed seconds for each measured call.
    """
    for _ in range(2):
        model.encode(texts, batch_size=8, normalize_embeddings=True)
    times = []
    for _ in range(7):
        start = time.perf_counter()
        model.encode(texts, batch_size=8, normalize_embeddings=True)
        times.append(time.perf_counter() - start)

    return times


def build_report(model, args, texts, times):
    """Summarize throughput, token lengths and process memory.

    Args:
        model: Loaded sentence encoder.
        args: Parsed command-line options.
        texts: Input texts with any intended task prefixes.
        times: Encoding durations in seconds.

    Returns:
        dict: JSON-compatible measurements and configuration.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "model": args.model,
        "revision": getattr(model[0].model.config, "_commit_hash", None),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 4,
        "batch_size": 8,
        "max_length": args.length,
        "parameters": sum(p.numel() for p in model.parameters()),
        "median_seconds": statistics.median(times),
        "sentences_per_second": len(texts) / statistics.median(times),
        "actual_token_lengths": model.preprocess(texts)["attention_mask"]
        .sum(-1)
        .tolist(),
        "seconds": times,
        "peak_process_rss_bytes": rss if platform.system() == "Darwin" else rss * 1024,
    }


def main():
    """Measure a CPU workload and write its timing and memory report."""
    args = parse_args()
    torch.set_num_threads(4)

    model = load_encoder(args)
    texts = ["検索文書: " + "東京は日本の首都です。" * args.length] * 8

    times = measure_encoding(model, texts)
    report = build_report(model, args, texts, times)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
