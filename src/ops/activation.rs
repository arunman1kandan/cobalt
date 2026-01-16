use crate::tensor::Tensor;

// ================================================================
// Activation Ops
//
// Phase 0:
// - ReLU
//
// Phase 1:
// - GELU
// - Sigmoid
// - Tanh
// - LeakyReLU
// ================================================================

pub fn relu(a: &Tensor) -> Tensor {
    let data = a
        .data
        .iter()
        .map(|x| if *x > 0.0 { *x } else { 0.0 })
        .collect::<Vec<f32>>();

    Tensor::new(data, a.shape.clone())
}
