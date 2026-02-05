/// CPU ISA detection utilities for x86_64 architecture
/// Still experimental and may be expanded in the future

/// Checks if the CPU supports AVX2 (Advanced Vector Extensions 2).
/// Includes 256-bit SIMD registers.
#[inline]
pub fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2")
}

/// Checks if the CPU supports AVX-512 Foundation instructions.
/// Includes 512-bit SIMD registers and many new operations.
#[inline]
pub fn has_avx512() -> bool {
    // generic AVX-512 feature (not BF16/VNNI yet)
    is_x86_feature_detected!("avx512f")
}