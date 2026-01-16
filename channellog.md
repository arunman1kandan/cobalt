commit: Add initial Tensor implementation with basic CPU math ops

Summary:
    Introduces the foundational Tensor abstraction for the framework.
    Implements core CPU-side operations including elementwise addition,
    multiplication, and a naive 2D matrix multiplication.

Details:
    - Added `Tensor` struct with contiguous Vec<f32> storage
    - Added shape management and validation logic
    - Added basic shape utilities (`rank`, `numel`, `reshape`)
    - Added Display implementation for debug-friendly output
    - Implemented basic elementwise ops: add, mul
    - Implemented naive 2D matmul with shape checks
    - Added initial docs for tensor and shape semantics:
        * notes/010-tensors.md
        * notes/020-shapes-and-math.md
    - Verified via manual forward execution in main()

Notes:
    This marks the end of Phase 0 for Tensor, establishing the numerical
    backbone prior to introducing autograd, computational graph nodes,
    and backend abstractions.

Next Steps:
    - Move math ops out of Tensor into an ops module
    - Introduce autograd metadata (parents + backward)
    - Prepare backend trait for future CUDA support
    - Expand op surface (relu, softmax, cross entropy, etc)

