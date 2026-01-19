# Debug Log

This log tracks issues encountered during development, their root causes, and resolutions. It serves as a learning resource for future reference.

## [2026-01-19] Datatype Implementation

### 1. Duplicate Import Error
**Issue:** `cargo run` failed with `error[E0252]: the name FrameworkError is defined multiple times`.
**Context:** In `src/backend/cpu/add_scalar.rs`.
**Cause:** I accidentally added `use crate::errors::FrameworkError;` twice when refactoring the file.
**Resolution:** Removed the duplicate `use` statement.

### 2. Stack Overflow in AVX Kernels
**Issue:** `cargo run` exited with `stack overflow` (exit code `0xc00000fd`) when running tests, specifically during the `add` operation.
**Context:** In `src/backend/cpu/add_avx2.rs` and `add_avx512.rs`.
**Diagnosis:**
The AVX kernels were calling `crate::ops::elementwise::add(a, b)` to handle the scalar broadcast resolution (Phase 1).
However, `crate::ops::elementwise::add` delegates to `crate::backend::cpu::add::add`.
And `crate::backend::cpu::add::add` calls the AVX kernels (like `add_avx2`) if the hardware supports it.
This created an infinite loop: `add` -> `add_avx2` -> `add` -> `add_avx2` ...
**Resolution:**
Modified the AVX kernels to call `crate::backend::cpu::add_scalar::add_scalar_dispatch(a, b)` directly for the fallback/broadcast phase, avoiding the re-entry into the top-level dispatch function.

### 3. Unimplemented Macros in Main
**Issue:** Panic with "not implemented: migrating to backend SIMD" during verification.
**Cause:** `main.rs` was calling `mul`, `matmul`, `relu`, and `softmax`.
**Resolution:** Commented out these calls in `main.rs` until they are implemented for the new datatype system.
