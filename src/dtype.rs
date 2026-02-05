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
    /// Standard 32-bit floating point (Single precision)
    FP32,
    /// 64-bit floating point (Double precision)
    FP64,
    /// 4-bit integer (packed)
    INT4,
    /// 8-bit signed integer
    INT8,
    /// 16-bit signed integer
    INT16,
    /// 32-bit signed integer
    INT32,
    /// 64-bit signed integer
    INT64,
    /// 8-bit unsigned integer (often used for images)
    UINT8,
    /// 16-bit unsigned integer
    UINT16,
    /// 32-bit unsigned integer
    UINT32,
    /// 64-bit unsigned integer
    UINT64,
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
            DType::INT16 => 2,
            DType::INT32 => 4,
            DType::INT64 => 8,
            DType::UINT8 => 1,
            DType::UINT16 => 2,
            DType::UINT32 => 4,
            DType::UINT64 => 8,
            DType::BOOL => 1,
        }
    }
    
    /// Returns true if this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::FP32 | DType::FP64 | DType::FP16 | DType::BF16 | 
                       DType::FP8E4M3 | DType::FP8E5M2)
    }
    
    /// Returns true if this is an integer type.
    pub fn is_int(&self) -> bool {
        matches!(self, DType::INT4 | DType::INT8 | DType::INT16 | DType::INT32 | DType::INT64 |
                       DType::UINT8 | DType::UINT16 | DType::UINT32 | DType::UINT64)
    }
}
