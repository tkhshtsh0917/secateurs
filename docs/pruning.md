# ModernBERT FFN pruning contract

English | [日本語](ja/pruning.md)

For hidden width H and intermediate width I, ModernBERT stores the packed input
projection as [2I, H] and the output projection as [H, I]. The two input branches
are combined by gated multiplication.

For retained neuron indices S, keep input rows S and S+I in branch order, and
output columns S. Biases, when present, are transformed consistently. Every layer
uses the same target intermediate width, while its selected indices may differ.
The standard `intermediate_size` configuration represents the resulting model.
Hidden width, vocabulary, attention and RoPE are not pruned by this recipe.

`activation_scores` measures the mean absolute gated activation entering the down
projection over valid tokens. Padding is excluded; prompts already present in
inputs are included. Calibration requires padded 3D activations and an attention
mask. It temporarily disables dropout and removes hooks on completion or failure.
This score is a proxy, not UltraBERT's exact aggregation. See the
[research comparison](research.md) for the methodological differences.

`make_plan` validates the target width and deterministically resolves ties. Without
scores, it uses seeded random selection. `prune_ffn` validates the plan before
copying and changing the model. Apply pruning before constructing optimizers or
adding compilation, quantization or distributed wrappers.

The package tests compare against a dense reference with removed gated neurons
zeroed, including non-contiguous indices. Saving and reloading must not require
objects left over from calibration.

Embedding dimension remains constant, but coordinates and rankings can change.
Use a consistent student model to encode both documents and queries.
