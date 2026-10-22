"""Prune the ModernBERT backbone in a SentenceTransformer."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from torch import nn

from .modernbert import FFNPlan, prune_ffn

if TYPE_CHECKING:
    from transformers import ModernBertModel


class _Encoder(Protocol):
    """Describe an encoder container without importing Sentence Transformers."""

    def __getitem__(self, index: int, /) -> nn.Module:
        """Return the registered module at the requested position."""
        ...


_EncoderT = TypeVar("_EncoderT", bound=_Encoder)


def prune_sentence_transformer(model: _EncoderT, plan: FFNPlan) -> _EncoderT:
    """Copy an encoder and prune its registered ModernBERT FFN.

    Shared backbone/config references are replaced together. Tokenizer,
    pooling and prompt settings are retained in the independent copy.

    Args:
        model: SentenceTransformer whose first module exposes model.
        plan: Compatible FFN pruning plan for that backbone.

    Returns:
        _EncoderT: Independent encoder of the input type with a pruned backbone.

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

    backbone = cast("ModernBertModel", first_module.model)
    pruned = prune_ffn(backbone, plan)

    return copy.deepcopy(
        model, memo={id(backbone): pruned, id(backbone.config): pruned.config}
    )
