use crate::tensor::Tensor;

// ================================================================
// Softmax (Phase 0): applies along last dimension
// Numerical stability: subtract max
//
// Softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// ================================================================

pub fn softmax(a: &Tensor) -> Tensor {
    assert!(a.rank() >= 1, "softmax: tensor rank must be >= 1");

    let last_dim = a.shape[a.rank() - 1];
    let batch = a.numel() / last_dim;

    let mut out = vec![0.0; a.numel()];

    for b in 0..batch {
        let start = b * last_dim;
        let end = start + last_dim;

        let slice = &a.data[start..end];

        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0;
        for x in slice {
            sum += (x - max_val).exp();
        }

        for i in 0..last_dim {
            out[start + i] = (slice[i] - max_val).exp() / sum;
        }
    }

    Tensor::new(out, a.shape.clone())
}
