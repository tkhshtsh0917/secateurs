"""Build a small, offline sentence encoder for integration tests."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast


def save_backbone(path):
    """Save a small ModernBERT backbone with eager attention."""
    config = ModernBertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=32,
        local_attention=8,
        pad_token_id=0,
        reference_compile=False,
    )
    config._attn_implementation = "eager"

    ModernBertModel(config).eval().save_pretrained(path)


def save_tokenizer(path):
    """Save a local word-level tokenizer with explicit input fields."""
    tokenizer = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[UNK]": 1, "東京": 2, "首都": 3, "日本": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()

    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        model_max_length=32,
        model_input_names=["input_ids", "attention_mask"],
    ).save_pretrained(path)


@pytest.fixture
def source(tmp_path):
    """Build an offline sentence encoder with tokenizer, pooling and a query prompt."""
    st = pytest.importorskip("sentence_transformers")
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer

    torch.manual_seed(42)
    path = tmp_path / "source"
    save_backbone(path)
    save_tokenizer(path)

    model = st.SentenceTransformer(
        modules=[
            Transformer(str(path), model_kwargs={"attn_implementation": "eager"}),
            Pooling(embedding_dimension=16, pooling_mode="mean", include_prompt=True),
        ],
        device="cpu",
    )
    model.prompts["query"] = "日本 "

    return model
