"""Write 256 sentence entries from the pinned JSTS training split."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

REVISION = "5bac629e25799df4c9c80a6a5db983d6cba9e77d"


def parse_args():
    """Parse command-line options.

    Returns:
        argparse.Namespace: Validated command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/train.txt",
        help="Text corpus path; the .json suffix is reserved for provenance",
    )

    return parser.parse_args()


def load_training_texts():
    """Load the first 128 JSTS training pairs as 256 sentence entries.

    Returns:
        list[str]: Nonempty training sentence entries.
    """
    data = load_dataset("mteb/JSTS", revision=REVISION, split="train")
    texts = []
    for row in data.select(range(128)):
        texts.extend([row["sentence1"], row["sentence2"]])

    return texts


def write_corpus(output, texts):
    """Write the texts and their provenance sidecar.

    Args:
        output: Destination path.
        texts: Input texts with any intended task prefixes.

    Raises:
        ValueError: The corpus path uses the reserved .json suffix.
    """
    if output.suffix.lower() == ".json":
        raise ValueError(
            "The corpus path must not use .json; it is the provenance sidecar"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(texts) + "\n", encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": "mteb/JSTS",
                "revision": REVISION,
                "split": "train",
                "rows": [0, 128],
                "texts": len(texts),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main():
    """Prepare the fixed training corpus and its provenance sidecar."""
    args = parse_args()
    texts = load_training_texts()

    write_corpus(Path(args.output), texts)


if __name__ == "__main__":
    main()
