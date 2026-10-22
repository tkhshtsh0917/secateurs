"""Prune the ModernBERT backbone in a SentenceTransformer."""

import copy

from .modernbert import prune_ffn


def prune_sentence_transformer(model, plan):
    """Copy an encoder and prune its registered ModernBERT FFN.

    Shared backbone/config references are replaced together. Tokenizer,
    pooling and prompt settings are retained in the independent copy.

    Args:
        model: SentenceTransformer whose first module exposes model.
        plan: Compatible FFN pruning plan for that backbone.

    Returns:
        SentenceTransformer: Independent encoder containing the pruned backbone.

    Raises:
        ValueError: The first module or backbone is unsupported, or the plan is invalid.

    Examples:
        >>> student = prune_sentence_transformer(encoder, plan)
    """
    try:
        first_module = model[0]
    except (TypeError, IndexError, KeyError):
        raise ValueError(
            "Expected a Sentence Transformers Transformer.model module"
        ) from None

    if not hasattr(first_module, "model"):
        raise ValueError("Expected a Sentence Transformers Transformer.model module")

    backbone = first_module.model
    pruned = prune_ffn(backbone, plan)

    return copy.deepcopy(
        model, memo={id(backbone): pruned, id(backbone.config): pruned.config}
    )
