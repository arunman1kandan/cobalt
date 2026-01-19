use std::fmt;

/// Core error type for the Cobalt framework.
///
/// This enum captures all possible failures during tensor operations,
/// including shape mismatches, type errors, and device conflicts.
#[derive(Debug)]
pub enum FrameworkError {
    /// Dimension mismatch during binary operations (e.g., trying to add [2, 3] and [4, 5]).
    ShapeMismatch { a: Vec<usize>, b: Vec<usize> },
    /// Failure to broadcast shapes according to NumPy rules.
    BroadcastMismatch { a: Vec<usize>, b: Vec<usize> },
    /// Operation attempted on tensors with different dtypes (e.g., FP32 + INT32).
    DTypeMismatch,
    /// Operation attempted on tensors on different devices (e.g., CPU + GPU).
    DeviceMismatch,
    /// The requested operation is not supported for the current configuration.
    UnsupportedOp(&'static str),
}

impl fmt::Display for FrameworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FrameworkError::ShapeMismatch { a, b } =>
                write!(f, "Shape mismatch: {:?} vs {:?}", a, b),
            FrameworkError::BroadcastMismatch { a, b } =>
                write!(f, "Broadcast mismatch: {:?} vs {:?}", a, b),
            FrameworkError::DTypeMismatch =>
                write!(f, "DType mismatch"),
            FrameworkError::DeviceMismatch =>
                write!(f, "Device mismatch"),
            FrameworkError::UnsupportedOp(op) =>
                write!(f, "Unsupported op: {}", op),
        }
    }
}
