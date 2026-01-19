/// Supported datatypes for Tensors.
/// Cobalt uses a simplified type system similar to PyTorch/NumPy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    /// 8-bit floating point (E4M3), useful for LLM quantization
    FP8E4M3,
    /// 8-bit floating point (E5M2), useful for LLM quantization
    FP8E5M2,
    /// 16-bit floating point (Half precision)
    FP16,
    /// Brain Floating Point (16-bit)
    BF16,
    /// Standard 32-bit floating point (Single availability)
    FP32,
    /// 64-bit floating point (Double precision)
    FP64,
    /// 4-bit integer (packed)
    INT4,
    /// 8-bit signed integer
    INT8,
    /// 32-bit signed integer
    INT32,
    /// 64-bit signed integer
    INT64,
    /// 8-bit unsigned integer (often used for images)
    UINT8,
    /// Boolean (stored as 1 byte)
    BOOL,
}

impl DType {
    /// Returns the size in bytes for a single element of this dtype.
    /// Useful for calculating buffer sizes.
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::FP8E4M3 => 1,
            DType::FP8E5M2 => 1,
            DType::FP16 => 2,
            DType::BF16 => 2,
            DType::FP32 => 4,
            DType::FP64 => 8,
            DType::INT4 => 1, // 2 values per byte, packed
            DType::INT8 => 1,
            DType::INT32 => 4,
            DType::INT64 => 8,
            DType::UINT8 => 1,
            DType::BOOL => 1,
        }
    }
}
