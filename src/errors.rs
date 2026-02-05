use std::fmt;

/// Core error type for the Cobalt framework.
///
/// This enum captures all possible failures during tensor operations,
/// including shape mismatches, type errors, and device conflicts.
#[derive(Debug)]
pub enum FrameworkError {
    /// Shape mismatch during operations.
    ShapeMismatch { expected: String, got: String },
    /// Failure to broadcast shapes according to NumPy rules.
    BroadcastMismatch { a: Vec<usize>, b: Vec<usize> },
    /// Operation attempted on tensors with different dtypes (e.g., FP32 + INT32).
    DTypeMismatch,
    /// Operation attempted on tensors on different devices (e.g., CPU + GPU).
    DeviceMismatch,
    /// The requested operation is not supported for the current configuration.
    UnsupportedOp(&'static str),
    /// Unsupported data type for the operation.
    UnsupportedDType(String),
        /// Index out of bounds for slicing.
        IndexOutOfBounds { index: usize, length: usize },
        /// Invalid dimension index for reshaping or transposing.
        InvalidDimension { dim: usize, rank: usize },
        /// Duplicate axis in permutation.
        DuplicateAxis { axis: usize },
}

impl fmt::Display for FrameworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameworkError::ShapeMismatch { expected, got } =>
                write!(f, "Shape mismatch: expected {}, got {}", expected, got),
            FrameworkError::BroadcastMismatch { a, b } =>
                write!(f, "Broadcast mismatch: {:?} vs {:?}", a, b),
            FrameworkError::DTypeMismatch =>
                write!(f, "DType mismatch"),
            FrameworkError::DeviceMismatch =>
                write!(f, "Device mismatch"),
            FrameworkError::UnsupportedOp(op) =>
                write!(f, "Unsupported op: {}", op),
            FrameworkError::UnsupportedDType(msg) =>
                write!(f, "Unsupported DType: {}", msg),
                FrameworkError::IndexOutOfBounds { index, length } =>
                    write!(f, "Index {} out of bounds for length {}", index, length),
                FrameworkError::InvalidDimension { dim, rank } =>
                    write!(f, "Dimension {} invalid for rank {}", dim, rank),
                FrameworkError::DuplicateAxis { axis } =>
                    write!(f, "Duplicate axis {} in permutation", axis),
        }
    }
}
