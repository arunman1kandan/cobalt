use crate::tensor::Tensor;

// ================================================================
// MatMul (Phase 0)
//
// 2D only: [A, B] x [B, C] -> [A, C]
// naive implementation (O(n^3))
//
// Future upgrades:
// - batched matmul (3D)
// - SIMD / BLAS acceleration
// - CUDA kernels
// - auto-diff
// ================================================================

pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.rank() == 2, "matmul: left tensor must be 2D");
    assert!(b.rank() == 2, "matmul: right tensor must be 2D");

    let m = a.shape[0];
    let k = a.shape[1];
    let k2 = b.shape[0];
    let n = b.shape[1];

    assert!(
        k == k2,
        "matmul: inner dimension mismatch {} vs {}",
        k, k2
    );

    let mut out = vec![0.0; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a.data[i * k + kk] * b.data[kk * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    Tensor::new(out, vec![m, n])
}
