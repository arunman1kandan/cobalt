# Activations & Softmax

## ReLU

ReLU(x) = max(0, x)

Key properties:
- Introduces sparsity in activations
- Avoids vanishing gradients for positive inputs
- Cheap to compute (piecewise linear)
- Dominant activation for modern deep networks (MLPs, CNNs, Transformers)

ReLU variants (future):
- Leaky ReLU
- GELU
- SiLU / Swish
- ELU
- Softplus

## Softmax

Softmax transforms logits into a probability distribution along a dimension.

Definition (with stability fix):
p_i = exp(x_i - max) / sum(exp(x_j - max))

Subtracting the maximum logit improves numerical stability during exponentiation.

Used for:
- Multi-class classification
- Cross entropy loss
- Attention mechanisms (Transformer softmax)
- Sequence models

## Phase 0 Implementation

Current behavior:
- Applies softmax along the last dimension
- Assumes 2D or higher-rank tensors flatten naturally along that dimension

## Future Extensions

Planned improvements:
- Batch-aware softmax
- Arbitrary axis softmax
- GPU kernel
- Parallel CPU kernel
- Mixed precision support (fp16/bf16)
- Fused softmax + cross entropy (common in training loops)
- Log-softmax for numerical stability in loss computations

## Notes

Softmax is numerically sensitive due to exponentiation. High-performance implementations use:
- max subtraction (current)
- log-sum-exp tricks for log-softmax
- fused kernels to avoid intermediate allocations
- hardware-specific execution paths (e.g., tensor cores)
