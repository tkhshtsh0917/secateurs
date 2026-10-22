import copy
import json
import subprocess
import sys

import pytest
import torch
from transformers import ModernBertConfig, ModernBertModel

from secateurs import FFNPlan, activation_scores, make_plan, prune_ffn


@pytest.fixture
def model():
    """Build a small biased ModernBERT backbone."""
    torch.manual_seed(42)
    config = ModernBertConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=64,
        local_attention=8,
        pad_token_id=0,
        reference_compile=False,
        mlp_bias=True,
        attention_dropout=0,
        mlp_dropout=0,
    )
    config._attn_implementation = "eager"

    return ModernBertModel(config).eval()


@pytest.fixture
def batch():
    """Provide padded inputs with different valid-token counts."""
    return {
        "input_ids": torch.tensor([[1, 2, 3, 0], [1, 4, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    }


@pytest.fixture
def plan():
    """Select non-contiguous source neurons in both layers."""
    return FFNPlan(16, 24, ((1, 4, 7, 15), (0, 3, 8, 20)))


@pytest.fixture
def student(model, plan):
    """Prune an independent copy of the source encoder."""
    return prune_ffn(model, plan)


def dense_reference(model, plan):
    """Zero removed output columns in a full-width source copy."""
    reference = copy.deepcopy(model)
    for layer, keep in zip(reference.layers, plan.indices, strict=False):
        removed = [i for i in range(plan.source_width) if i not in keep]
        with torch.no_grad():
            layer.mlp.Wo.weight[:, removed] = 0

    return reference


def test_zero_pruning(model, batch):
    """Verify zero pruning."""
    result = prune_ffn(model, make_plan(model, 24))

    torch.testing.assert_close(
        model(**batch).last_hidden_state, result(**batch).last_hidden_state
    )


def test_pruning_matches_dense_reference(model, plan, student, batch):
    """Verify pruning matches dense reference."""
    reference = dense_reference(model, plan)

    torch.testing.assert_close(
        reference(**batch).last_hidden_state, student(**batch).last_hidden_state
    )


def test_pruning_preserves_source_weights(model, plan):
    """Verify pruning preserves source weights."""
    before = copy.deepcopy(model.state_dict())

    prune_ffn(model, plan)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_model_save_reload(student, batch, tmp_path):
    """Verify model save reload."""
    student.save_pretrained(tmp_path)
    loaded = ModernBertModel.from_pretrained(tmp_path, attn_implementation="eager")

    torch.testing.assert_close(
        student(**batch).last_hidden_state, loaded(**batch).last_hidden_state
    )


def test_student_backward(student, batch):
    """Verify student backward."""
    student(**batch).last_hidden_state.square().mean().backward()

    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in student.parameters()
    )


def test_plan_save_reload(plan, tmp_path):
    """Verify plan save reload."""
    path = tmp_path / "plan.json"
    plan.save(path)

    assert FFNPlan.load(path) == plan


@pytest.mark.parametrize(
    "indices", [((0, 0), (0, 1)), ((24,), (1,)), ((), ()), ((0,),)]
)
def test_invalid_indices(model, indices):
    """Verify invalid indices."""
    with pytest.raises(ValueError):
        prune_ffn(model, FFNPlan(16, 24, indices))


def test_calibration_excludes_padding_across_batches(model, batch):
    """Verify calibration excludes padding across batches."""
    whole = activation_scores(model, [batch])
    split = activation_scores(
        model, [{k: v[i : i + 1] for k, v in batch.items()} for i in range(2)]
    )

    torch.testing.assert_close(whole, split)


@pytest.mark.parametrize("missing_mask", [False, True])
def test_calibration_failure_removes_hooks(model, batch, missing_mask):
    """Verify calibration failure removes hooks."""
    batches = [{"input_ids": batch["input_ids"]}] if missing_mask else []

    with pytest.raises(ValueError):
        activation_scores(model, batches)

    assert all(not layer.mlp.Wo._forward_pre_hooks for layer in model.layers)


def test_forward_failure_restores_calibration_state(model, batch, monkeypatch):
    """Verify forward failure restores calibration state."""
    model.train()
    states = [module.training for module in model.modules()]

    def fail(**kwargs):
        """Raise an artificial forward-pass error."""
        raise RuntimeError("forward interrupted")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="forward interrupted"):
        activation_scores(model, [batch])

    assert states == [module.training for module in model.modules()]
    assert all(not layer.mlp.Wo._forward_pre_hooks for layer in model.layers)


def test_sentence_transformer_registered_backbone(model):
    """Verify sentence transformer registered backbone."""
    from secateurs.sentence_transformers import prune_sentence_transformer

    class Wrapper(torch.nn.Module):
        """Expose the registered backbone as a model module."""

        def __init__(self, backbone):
            """Register the supplied backbone."""
            super().__init__()
            self.model = backbone

    original = torch.nn.ModuleList([Wrapper(model)])
    student = prune_sentence_transformer(original, make_plan(model, 12))

    assert student[0].model.config.intermediate_size == 12
    assert original[0].model.config.intermediate_size == 24
    assert sum(p.numel() for p in student.parameters()) < sum(
        p.numel() for p in original.parameters()
    )


def test_plan_rejects_nonfinite_scores(model):
    """Verify plan rejects nonfinite scores."""
    with pytest.raises(ValueError):
        make_plan(model, 12, torch.full((2, 24), float("nan")))


def test_plan_rejects_stale_dimensions(model):
    """Verify plan rejects stale dimensions."""
    with pytest.raises(ValueError):
        prune_ffn(model, FFNPlan(16, 23, ((0,), (0,))))


def test_score_ties_keep_source_order(model):
    """Verify score ties keep source order."""
    scores = torch.zeros(2, 24)

    assert make_plan(model, 3, scores).indices == ((0, 1, 2), (0, 1, 2))


def test_random_plan_is_reproducible(model):
    """Verify random plan is reproducible."""
    assert make_plan(model, 3, seed=7) == make_plan(model, 3, seed=7)


def encode_in_fresh_process(path, batch):
    """Load a saved backbone and encode inputs in a separate process."""
    code = """
import json
import sys

import torch
from transformers import ModernBertModel

model = ModernBertModel.from_pretrained(sys.argv[1], attn_implementation="eager").eval()
batch = {key: torch.tensor(value) for key, value in json.loads(sys.argv[2]).items()}

with torch.no_grad():
    output = model(**batch).last_hidden_state

print(json.dumps(output.tolist()))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(path),
            json.dumps({k: v.tolist() for k, v in batch.items()}),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    return torch.tensor(json.loads(result.stdout))


def test_fresh_process_reload(student, batch, tmp_path):
    """Verify fresh process reload."""
    student.save_pretrained(tmp_path)

    actual = encode_in_fresh_process(tmp_path, batch)

    torch.testing.assert_close(actual, student(**batch).last_hidden_state)


def test_revision_guard(model):
    """Verify revision guard."""
    model.config._commit_hash = "source-a"
    plan = make_plan(model, 12)
    model.config._commit_hash = "source-b"

    with pytest.raises(ValueError, match="revision"):
        prune_ffn(model, plan)


def test_calibration_preserves_mixed_training_flags(model, batch):
    """Verify calibration preserves mixed training flags."""
    model.train()
    model.layers[0].eval()
    states = [module.training for module in model.modules()]

    activation_scores(model, [batch])

    assert states == [module.training for module in model.modules()]


def test_pruning_preserves_training_and_gradient_flags(model, plan):
    """Verify pruning preserves training and gradient flags."""
    model.train()
    model.layers[0].eval()
    model.layers[0].mlp.Wi.requires_grad_(False)

    student = prune_ffn(model, plan)

    assert student.training
    assert not student.layers[0].training
    assert not student.layers[0].mlp.Wi.weight.requires_grad
    assert not student.layers[0].mlp.Wi.bias.requires_grad
    assert student.layers[1].mlp.Wi.weight.requires_grad


def test_unsupported_model_is_rejected():
    """Verify unsupported model is rejected."""
    with pytest.raises(ValueError, match="ModernBertModel"):
        make_plan(torch.nn.Linear(2, 2), 1)


@pytest.mark.parametrize(
    "encoder", [None, [], [torch.nn.Linear(2, 2)], torch.nn.Linear(2, 2)]
)
def test_adapter_rejects_unsupported_encoder_with_value_error(encoder):
    """Give the documented error for absent or unsupported first modules."""
    from secateurs.sentence_transformers import prune_sentence_transformer

    with pytest.raises(ValueError, match="Transformer.model"):
        prune_sentence_transformer(encoder, None)


def test_nonfinite_hook_input_restores_calibration_state(model, batch):
    """Remove hooks and restore modes when activation validation fails in a hook."""
    model.train()
    model.layers[0].eval()
    modes = [module.training for module in model.modules()]
    with torch.no_grad():
        model.layers[0].mlp.Wi.weight.fill_(float("nan"))

    with pytest.raises(ValueError, match="Non-finite calibration activations"):
        activation_scores(model, [batch])

    assert modes == [module.training for module in model.modules()]
    assert all(not layer.mlp.Wo._forward_pre_hooks for layer in model.layers)
