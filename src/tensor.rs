// ================================================================
// Cobalt: Tensor (Phase 0.2 - ops extraction, CPU-only)
//
// Tensor now acts primarily as a data container.
//
// Responsibilities:
//   - Holds raw data (Vec<f32>)
//   - Holds shape metadata
//   - Performs shape validation
//   - Performs reshaping
//   - Provides ergonomic method wrappers for ops
//
// All math logic now lives inside `ops::*` modules.
//
// Future responsibilities:
//   - Gradient storage (autodiff)
//   - Graph node metadata
//   - Device placement (CPU/CUDA)
//   - Serialization
//   - Stride support + slicing
// ================================================================

use std::fmt;

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a tensor from raw data and shape.
    /// Checks that data length matches shape product.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert!(
            expected_len == data.len(),
            "Tensor::new mismatch: expected {} values for shape {:?}, got {}",
            expected_len,
            shape,
            data.len()
        ); 

        Self { data, shape }
    }

    /// Create a 1D tensor from Vec<f32>.
    pub fn from_vec(v: Vec<f32>) -> Self {
        Self {
            data: v.clone(),
            shape: vec![v.len()],
        }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Reshape (metadata only, no data movement)
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let expected: usize = new_shape.iter().product();
        assert!(
            expected == self.numel(),
            "reshape mismatch: expected {} elements, got {}",
            expected,
            self.numel()
        );
        self.shape = new_shape;
    }
}

// ----------------------------
// Display for debug
// ----------------------------
impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, data={:?})",
            self.shape, self.data
        )
    }
}

// ================================================================
// Ergonomic wrappers for ops (no math logic here)
// ================================================================
//
// These methods forward directly into the ops module.
// This keeps Tensor expressive without polluting it.
// ================================================================

impl Tensor {
    /// Elementwise add (Tensor + Tensor)
    pub fn add(&self, other: &Tensor) -> Tensor {
        crate::ops::elementwise::add(self, other)
    }

    /// Elementwise multiply (Tensor * Tensor)
    pub fn mul(&self, other: &Tensor) -> Tensor {
        crate::ops::elementwise::mul(self, other)
    }

    /// 2D matrix multiply
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        crate::ops::matmul::matmul(self, other)
    }

    /// ReLU activation
    pub fn relu(&self) -> Tensor {
        crate::ops::activation::relu(self)
    }

    /// Softmax (Phase 1: applied along last dim)
    pub fn softmax(&self) -> Tensor {
        crate::ops::softmax::softmax(self)
    }
}
