# Research scope and relationship to UltraBERT

English | [日本語](ja/research.md)

secateurs is an early reproduction study of UltraBERT-inspired pruning, adapted
to existing embedding models. The current prototype tests part of the pruning
and recovery workflow. It does not reproduce UltraBERT pretraining or establish
the paper's performance claims.

The intended outcome is a reusable pruning library. This is a development goal,
not a description of the current level of coverage or maturity. See the
[roadmap](roadmap.md) and [experiment record](experiments.md).

## Shared approach and current differences

Both approaches retain the encoder's depth, use forward activations to estimate
importance, reuse selected weights, and follow structural pruning with training
against a teacher. These ideas also occur in earlier pruning and distillation
work; they are not unique contributions of UltraBERT or secateurs.

| Aspect | UltraBERT v1 | Current secateurs prototype |
| --- | --- | --- |
| Starting point | A 1.3B encoder, compressed to 380M and then 150M | An existing ModernBERT embedding model; the reference pilot uses ruri-v3-310m |
| Structures removed | Hidden channels, attention heads or head dimensions, and FFN neurons | FFN intermediate neurons only; residual width, attention, vocabulary, RoPE and pooling are retained |
| FFN importance | Magnitudes of gated activations, averaged over the sequence and aggregated by an L2 norm over the batch | Absolute gated activations averaged over all valid tokens; padding is excluded and input prompt tokens are included |
| Compression stages | Pruning and distillation at both 1.3B to 380M and 380M to 150M | One pruning step followed by optional embedding recovery in the pilot |
| Teacher and target | The original 1.3B teacher supplies token prediction logits for both student sizes; the loss is KL divergence | The original Sentence Transformer supplies precomputed final sentence embeddings; the loss is MSE |
| Training objective | Separate LM heads for 5% and 50% masked sequences, mixed at a 4:1 sequence ratio; KL on the 5% sequences and standard MLM on the 50% sequences | No MLM objective, masking-ratio mixture, separate LM heads or masking-dependent distillation |
| Training scale | 2T tokens, including a final context extension to 32K | A small recovery pilot; the existing model's sequence-length limit is retained |

The UltraBERT column follows [Section 4 and Appendix A.2 of the v1 paper](https://arxiv.org/html/2610.15028v1).
The [official model card](https://huggingface.co/dmis-lab/UltraBERT-150M) also
describes the teacher, masking objectives and width changes.

The difference in aggregation can change which neurons are retained. A global
token average also gives longer sequences more weight than shorter ones.
Implementing a forward-activation score does not, by itself, reproduce the
paper's selection rule.

Embedding MSE is knowledge distillation, but it transfers a different signal from
the paper's token-logit objective. Increasing the recovery data alone would not
make the current method equivalent to UltraBERT.

## Claims and evidence

The supported description is **UltraBERT-inspired FFN pruning with an embedding
recovery example**. Claims of an UltraBERT implementation, faithful reproduction,
equivalent results, or general quality preservation would exceed current evidence.

The [pruning contract](pruning.md) specifies the implemented transformation, and
[recovery](recovery.md) describes the example's loss and defaults. Numerical tests
cover transformation correctness, source preservation, gradients and serialization.
They do not establish downstream quality or inference gains. See the
[evaluation policy](evaluation.md) for the remaining comparisons and reporting rules.

Adding hidden or attention pruning, an exact scoring recipe, or token-logit
distillation would require separate implementations and validation. The project
does not currently promise to reproduce every part of UltraBERT pretraining.
