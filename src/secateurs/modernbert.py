"""Score FFN neurons and prune ModernBERT projection matrices."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle
    from transformers import ModernBertModel
    from transformers.models.modernbert.modeling_modernbert import ModernBertMLP


@dataclass(frozen=True)
class FFNPlan:
    """Store retained neuron indices relative to the source model.

    Attributes:
        hidden_size: Source residual width.
        source_width: Source FFN intermediate width.
        indices: Retained source indices for each layer.
        format_version: Serialized plan schema version.
        source_revision: Optional source checkpoint revision.
    """

    hidden_size: int
    source_width: int
    indices: tuple[tuple[int, ...], ...]
    format_version: int = 1
    source_revision: str | None = None

    def save(self, path: str | Path) -> None:
        """Write the pruning plan as UTF-8 JSON.

        Args:
            path: Destination file; its parent must exist.
        """
        Path(path).write_text(
            json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> FFNPlan:
        """Load source dimensions and retained indices from JSON.

        Args:
            path: Serialized plan file.

        Returns:
            FFNPlan: The loaded plan, with immutable index tuples.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["indices"] = tuple(tuple(row) for row in data["indices"])
        return cls(**data)


def _check(model: ModernBertModel) -> tuple[int, int]:
    """Validate the backbone's type, projections and floating-point weights.

    Args:
        model: Uncompiled ModernBertModel backbone.

    Returns:
        tuple: Source hidden width and FFN intermediate width.

    Raises:
        ValueError: The backbone or projection layout is unsupported.
    """
    if getattr(getattr(model, "config", None), "model_type", None) != "modernbert":
        raise ValueError("Only ModernBertModel is supported")
    if not hasattr(model, "layers"):
        raise ValueError("Pass the ModernBertModel backbone")
    if hasattr(model, "_orig_mod"):
        raise ValueError("Prune the original model before compilation")

    h, width = model.config.hidden_size, model.config.intermediate_size
    for layer in model.layers:
        mlp = cast("ModernBertMLP", layer.mlp)
        for projection in (mlp.Wi, mlp.Wo):
            if (
                type(projection) is not nn.Linear
                or not projection.weight.is_floating_point()
            ):
                raise ValueError(
                    "Only ordinary floating-point Linear projections are supported"
                )
        if mlp.Wi.weight.shape != (2 * width, h):
            raise ValueError("Unsupported packed input projection")
        if mlp.Wo.weight.shape != (h, width):
            raise ValueError("Unsupported output projection")

    return h, width


def make_plan(
    model: ModernBertModel,
    target_width: int,
    scores: torch.Tensor | Sequence[Sequence[float]] | None = None,
    seed: int = 0,
) -> FFNPlan:
    """Select FFN neurons by score, or by a seeded random baseline.

    Stable sorting breaks score ties in source order. Retained indices are
    stored in source order to preserve the paired gated-projection layout.

    Args:
        model: Source ModernBertModel backbone.
        target_width: Number of neurons to retain in every layer.
        scores: Optional finite scores shaped [layers, intermediate_size].
        seed: Random seed used only when scores are omitted.

    Returns:
        FFNPlan: Source metadata and per-layer retained indices.

    Raises:
        ValueError: The model, target width or scores are invalid.

    Examples:
        >>> plan = make_plan(model, target_width=12, scores=scores)
    """
    h, width = _check(model)
    if isinstance(target_width, bool) or not isinstance(target_width, int):
        raise ValueError("target_width must be an integer")
    if not 0 < target_width <= width:
        raise ValueError("target_width outside source width")

    if scores is None:
        generator = torch.Generator().manual_seed(seed)
        scores = torch.rand(len(model.layers), width, generator=generator)

    scores = torch.as_tensor(scores).detach().cpu()
    if scores.shape != (len(model.layers), width) or not torch.isfinite(scores).all():
        raise ValueError(
            "Expected finite scores with shape [layers, intermediate_size]"
        )

    selected = scores.argsort(dim=-1, descending=True, stable=True)[:, :target_width]
    return FFNPlan(
        h,
        width,
        tuple(tuple(sorted(row.tolist())) for row in selected),
        source_revision=getattr(model.config, "_commit_hash", None),
    )


def _validate_plan(
    model: ModernBertModel, plan: FFNPlan, hidden_size: int, source_width: int
) -> int:
    """Validate source metadata, uniform widths and retained indices.

    Args:
        model: Source backbone whose layer count and revision are checked.
        plan: Pruning plan to validate.
        hidden_size: Expected residual width.
        source_width: Expected FFN intermediate width.

    Returns:
        int: Validated target intermediate width.

    Raises:
        ValueError: Metadata, revision or neuron indices are incompatible.
    """
    expected = (1, hidden_size, source_width)
    if (plan.format_version, plan.hidden_size, plan.source_width) != expected:
        raise ValueError("Incompatible plan")
    if len(plan.indices) != len(model.layers):
        raise ValueError("Layer count mismatch")

    sizes = {len(row) for row in plan.indices}
    if len(sizes) != 1 or not sizes or next(iter(sizes)) == 0:
        raise ValueError("Nonempty uniform widths required")

    for row in plan.indices:
        if any(type(i) is not int or i < 0 or i >= source_width for i in row):
            raise ValueError("Invalid neuron index")
        if len(set(row)) != len(row):
            raise ValueError("Duplicate neuron indices")

    if plan.source_revision is not None and plan.source_revision != getattr(
        model.config, "_commit_hash", None
    ):
        raise ValueError("Model revision mismatch")

    return next(iter(sizes))


def _slice_linear(
    layer: nn.Linear, indices: torch.Tensor, axis: Literal[0, 1]
) -> nn.Linear:
    """Copy selected rows or columns into a smaller Linear layer.

    The copy retains device, dtype, matching bias, gradient flags and mode.

    Args:
        layer: Linear projection to copy.
        indices: Retained indices on the projection's device.
        axis: Zero for output rows, one for input columns.

    Returns:
        nn.Linear: Independent projection containing the selected weights.
    """
    out_features, in_features = layer.weight.shape
    if axis == 0:
        out_features = len(indices)
    else:
        in_features = len(indices)

    result = nn.Linear(
        in_features,
        out_features,
        bias=layer.bias is not None,
        device=layer.weight.device,
        dtype=layer.weight.dtype,
    )
    with torch.no_grad():
        result.weight.copy_(layer.weight.index_select(axis, indices))
        if layer.bias is not None:
            bias = layer.bias.index_select(0, indices) if axis == 0 else layer.bias
            result.bias.copy_(bias)

    result.weight.requires_grad_(layer.weight.requires_grad)
    if layer.bias is not None:
        result.bias.requires_grad_(layer.bias.requires_grad)
    result.train(layer.training)

    return result


def _prune_mlp(mlp: ModernBertMLP, retained: Sequence[int], source_width: int) -> None:
    """Shrink gated input rows and their matching output columns in place.

    Args:
        mlp: Copied MLP containing Wi and Wo projections.
        retained: Source neuron indices to retain.
        source_width: Width separating the two packed input branches.
    """
    indices = torch.tensor(retained, device=mlp.Wi.weight.device)
    packed = torch.cat((indices, indices + source_width))

    mlp.Wi = _slice_linear(mlp.Wi, packed, axis=0)
    mlp.Wo = _slice_linear(mlp.Wo, indices, axis=1)


def prune_ffn(model: ModernBertModel, plan: FFNPlan) -> ModernBertModel:
    """Copy a ModernBERT backbone and shrink its FFN matrices.

    The plan is validated before copying; the source model is never modified.

    Args:
        model: Source ModernBertModel with ordinary floating-point projections.
        plan: Compatible plan with a common retained width across layers.

    Returns:
        ModernBertModel: Independent model with its intermediate_size updated.

    Raises:
        ValueError: The model or plan violates the pruning contract.

    Examples:
        >>> student = prune_ffn(model, plan)
    """
    hidden_size, source_width = _check(model)
    target = _validate_plan(model, plan, hidden_size, source_width)

    result = copy.deepcopy(model)
    for layer, retained in zip(result.layers, plan.indices, strict=False):
        _prune_mlp(cast("ModernBertMLP", layer.mlp), retained, source_width)
    result.config.intermediate_size = target

    return result


class _ActivationAccumulator:
    """Accumulate valid-token activation magnitudes for each FFN neuron.

    Attributes:
        totals: Per-layer sums accumulated on CPU in float64.
        counts: Number of valid tokens observed in each layer.
        mask: Attention mask for the current batch.
    """

    def __init__(self, layers: int, width: int) -> None:
        """Allocate sums and counts for the source FFN layout.

        Args:
            layers: Number of encoder layers.
            width: Number of FFN neurons per layer.
        """
        self.totals = torch.zeros(layers, width, dtype=torch.float64)
        self.counts = torch.zeros(layers, dtype=torch.long)
        self.mask: torch.Tensor | None = None

    def hook_for(
        self, index: int
    ) -> Callable[[nn.Module, tuple[torch.Tensor, ...]], None]:
        """Create a hook for one layer's down-projection input.

        Args:
            index: Layer position in the accumulator.

        Returns:
            Callable: Forward pre-hook that records gated activations.
        """

        def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            """Record this layer's gated activations."""
            self.record(index, args[0])

        return hook

    def record(self, index: int, activation: torch.Tensor) -> None:
        """Validate and accumulate activations after excluding padding.

        Args:
            index: Layer position in the accumulator.
            activation: Tensor shaped [batch, sequence, intermediate_size].

        Raises:
            ValueError: Shapes mismatch or valid activations are nonfinite.
        """
        activation = activation.detach().float()
        if (
            activation.ndim != 3
            or self.mask is None
            or activation.shape[:2] != self.mask.shape
        ):
            raise ValueError(
                "Calibration requires padded 3D activations and attention_mask"
            )

        valid = activation[self.mask.bool()]
        if not torch.isfinite(valid).all():
            raise ValueError("Non-finite calibration activations")

        self.totals[index].add_(valid.abs().sum(0).double().cpu())
        self.counts[index] += len(valid)

    def mean(self) -> torch.Tensor:
        """Compute each neuron's mean magnitude across valid tokens.

        Returns:
            torch.Tensor: Float32 scores shaped [layers, intermediate_size].

        Raises:
            ValueError: Any layer has no valid observations.
        """
        if (self.counts == 0).any():
            raise ValueError("Calibration contains no valid tokens")

        return (self.totals / self.counts[:, None]).float()


@contextmanager
def _calibration_hooks(
    model: ModernBertModel, accumulator: _ActivationAccumulator
) -> Iterator[None]:
    """Install hooks in evaluation mode and always restore model state.

    The finally block removes hooks and restores each module's original
    training flag, including when a forward pass or hook fails.

    Args:
        model: Backbone whose down projections are observed.
        accumulator: Activation recorder used by the hooks.

    Yields:
        None: Control while calibration hooks are installed.
    """
    handles: list[RemovableHandle] = []
    training = {module: module.training for module in model.modules()}

    try:
        model.eval()
        for index, layer in enumerate(model.layers):
            hook = accumulator.hook_for(index)
            mlp = cast("ModernBertMLP", layer.mlp)
            handles.append(mlp.Wo.register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for module, state in training.items():
            module.training = state


@torch.no_grad()
def activation_scores(
    model: ModernBertModel, batches: Iterable[Mapping[str, torch.Tensor]]
) -> torch.Tensor:
    """Average gated activation magnitudes over non-padding tokens.

    Calibration disables gradients and dropout temporarily. Prompts present
    in input tokens are included; hooks and training flags are restored.

    Args:
        model: Source ModernBertModel with padded eager/SDPA activations.
        batches: Iterable of model inputs containing attention_mask.

    Returns:
        torch.Tensor: Float32 scores shaped [layers, intermediate_size].

    Raises:
        ValueError: The model, masks or observed activations are invalid.

    Examples:
        >>> scores = activation_scores(model, [batch])
    """
    _, width = _check(model)
    accumulator = _ActivationAccumulator(len(model.layers), width)

    with _calibration_hooks(model, accumulator):
        for batch in batches:
            accumulator.mask = batch.get("attention_mask")
            if accumulator.mask is None:
                raise ValueError("attention_mask required")

            model(**batch)

    return accumulator.mean()
