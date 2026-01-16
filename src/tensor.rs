// ===============================================
// Cobalt: Tensor (Phase 0.1, CPU, no autodiff)
// ===============================================
//
// Tensor is the core numerical container.
// Stores data contiguously in a Vec<f32>
// with dynamic shape.
//
// Future expansions:
// - strides
// - device backends (CPU/GPU)
// - gradient tracking (autodiff)
// - slicing and views
//
// This file is intentionally simple to establish
// architectural foundations without distractions.
//


use std::fmt; // For custom display formatting

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>, // Contiguous data storage
    pub shape: Vec<usize>, // Dynamic shape
}

impl Tensor {
    /// Create a tensor from the raw data and explict shape
    /// Ensures the shape product matches data length

    /// Constructor
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product(); // Calculate expected length from shape
        assert!(
            data.len() == expected_len,
            "Tensor::new shape mismatch: expected {} values, got {}",
            expected_len,
            data.len()
        );

        Self { data, shape }
    }

    /// Create a 1d Tensor from a slice
    pub fn from_vec(v: Vec<f32>) -> Self {
        let shape = vec![v.len()]; // 1d shape (Make sure to check if vec is empty?)
        Self { data: v, shape }
    }

    /// Method to get the number of elements in the tensor
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Method to check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Method to return the Rank (number of dimensions) of the tensor
    pub fn rank(&self) -> usize{
        self.shape.len() // Number of dimensions is the length of the shape vector
    }

    /// Method to return the total number of elements in the tensor as usize
     pub fn numel(&self) -> usize {
        self.data.len() // Total number of elements is the length of the data vector
    }

    /// Method to reshape the tensor (copy-free and metadata change)
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        let expected: usize = new_shape.iter().product(); // Calculate expected length from new shape
        assert!(
            expected == self.numel(),
            "Tensor::reshape shape mismatch: expected {} values, got {}",
            expected,
            self.numel()
        );
        self.shape = new_shape; // Update shape
    }

    /// Method to add two tensors 
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert!(
            self.shape == other.shape,
            "Tensor::add shape mismatch: expected {:?}, got {:?}",
            self.shape,
            other.shape
        );

        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect(); 

        // The above line iterates over both tensors' data, adds corresponding elements, and collects the results into a new Vec<f32>
        // The flow is as below and is done in a functional programming style:
        // 1. self.data.iter() creates an iterator over the elements of the first tensor's data
        // 2. other.data.iter() creates an iterator over the elements of the second tensor's data
        // 3. .zip(...) pairs elements from both iterators together into tuples
        // 4. .map(|(a, b)| a + b) applies the addition operation to each tuple of paired elements
        // 5. .collect() gathers all the results into a new Vec<f32>

        Tensor::new(data, self.shape.clone())
    }

    /// Method to multiply two tensors element-wise
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert!(
            self.shape == other.shape,
            "Tensor::mul shape mismatch: expected {:?}, got {:?}",
            self.shape,
            other.shape
        );

        let data = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).collect();

        Tensor::new(data, self.shape.clone())
    }

    /// Matrix multiplication (Phase 0: 2D only)
    ///
    /// - self: [A, B]
    /// - other: [B, C]
    /// - output: [A, C]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.rank() == 2, "matmul: left tensor must be 2D");
        assert!(other.rank() == 2, "matmul: right tensor must be 2D");

        let a = self.shape[0];
        let b = self.shape[1];
        let b2 = other.shape[0];
        let c = other.shape[1];

        assert!(
            b == b2,
            "matmul dim mismatch: {} (left) vs {} (right)",
            b, b2
        );

        let mut out = vec![0.0; a * c];

        for i in 0..a {
            for j in 0..c {
                let mut sum = 0.0;
                for k in 0..b {
                    sum += self.data[i * b + k] * other.data[k * c + j];
                }
                out[i * c + j] = sum;
            }
        }

        Tensor::new(out, vec![a, c])
    }

} 

/// Implement Display trait for pretty-printing the Tensor

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { 
        write!(f, "Tensor(shape={:?}, data={:?})", self.shape, self.data)
    }
}