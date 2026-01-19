/// Represents the compute device where the Tensor data resides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// Host CPU (Central Processing Unit).
    CPU,
    /// NVIDIA GPU (Compute Unified Device Architecture).
    /// Note: CUDA support is currently planned for Phase 3.
    CUDA,
}
