/// CPU-specific backend implementation.
///
/// Handles generic scalar dispatch and optimized SIMD kernels.
pub mod isa;

pub mod add_scalar;
pub mod add_avx2;
pub mod add_avx512;
pub mod add;
