use crate::tensor::Tensor;

// ================================================================
// Elementwise Ops (Phase 0)
// add, mul
//
// Future phases may include:
// - broadcasting
// - fused kernels
// - device dispatch (CPU/CUDA)
// ================================================================

pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(
        a.shape == b.shape,
        "elementwise add: shape mismatch {:?} vs {:?}",
        a.shape,
        b.shape
    );

    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .collect::<Vec<f32>>();

    Tensor::new(data, a.shape.clone())
}

pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(
        a.shape == b.shape,
        "elementwise mul: shape mismatch {:?} vs {:?}",
        a.shape,
        b.shape
    );

    let data = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x * y)
        .collect::<Vec<f32>>();

    Tensor::new(data, a.shape.clone())
}
