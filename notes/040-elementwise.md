# 040: Elementwise Operations

## 1. Motivation
The vast majority of FLOPs (Floating Point Operations) in a neural network are arguably in the matrix multiplications, but the vast majority of **kernels** (unique instructions) are Elementwise Ops. These include activation functions (`ReLU`), normalization steps (`Add`, `Div`), and loss calculations.
We need these to be:
1.  **Fast**: Consuming memory bandwidth efficiently.
2.  **Correct**: Handling NaNs and Infinities propertly.
3.  **Composable**: Supporting arbitrary chains of math ($y = \sigma(x) * x + b$).

## 2. Context / Precedence
*   **BLAS Level 1**: Basic Linear Algebra Subprograms defined vector-vector operations (`axpy`: $\alpha x + y$).
*   **NumPy UFuncs**: Universal Functions that operate element-by-element and support broadcasting.

## 3. Intuition
**The Assembly Line**:
Imagine a factory belt carrying raw numbers.
You have a row of robots. Each robot has a specific tool (a "Adder", a "Multiplier").
*   **Scalar Execution**: One robot works on one item.
*   **SIMD Execution**: One giant robot arm picks up 8 items at once and stamps them all instantly.

Correctness is guaranteed because the $i$-th result depends *only* on the $i$-th input. There are no dependencies between neighbors.

## 4. Formal Definition
Let $A, B$ be tensors of shape $S$ with elements $a_\mathbf{i}, b_\mathbf{i}$ indexed by $\mathbf{i}$.
A binary elementwise operation $\odot$ yields tensor $C$ where:
$$ c_\mathbf{i} = a_\mathbf{i} \odot b_\mathbf{i} $$
The time complexity is $O(N)$ where $N$ is the total number of elements. The space complexity is $O(N)$ (for the output).

## 5. Mathematical Deep Dive
**Vectorization Potential**:
Since the operation is independent, it is "Embarrassingly Parallel".
$$ \text{map}(f, [x_0, \dots, x_n]) = [f(x_0), \dots, f(x_n)] $$

This allows splitting the array into $K$ chunks and processing them on $K$ cores without any synchronization locks.

## 6. Computation / Implementation Details
*   **SIMD (Single Instruction, Multiple Data)**:
    *   `AVX2`: 256-bit registers. Holds 8 `f32` or 4 `f64`.
    *   `AVX-512`: 512-bit registers. Holds 16 `f32`.
*   **Kernel Unrolling**: We manually unroll loops to reduce branch prediction overhead.
*   **Reference Implementation (Rust)**:
    ```rust
    // Raw Loop (Slow)
    for i in 0..len { out[i] = a[i] + b[i]; }
    
    // Auto-Vectorized (Fast)
    out.iter_mut().zip(a.iter()).zip(b.iter()).for_each(|((o, x), y)| *o = x + y);
    ```

## 7. Minimal Code

### SIMD Intrinsic Example (Pseudocode)
```rust
unsafe fn add_avx(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let mut i = 0;
    while i <= n - 8 {
        // Load 8 floats into registers
        let ra = _mm256_loadu_ps(a.as_ptr().add(i));
        let rb = _mm256_loadu_ps(b.as_ptr().add(i));
        // Add them
        let rc = _mm256_add_ps(ra, rb);
        // Store back
        _mm256_storeu_ps(out.as_mut_ptr().add(i), rc);
        i += 8;
    }
    // Handle remaining (scalar loop)
    while i < n {
        out[i] = a[i] + b[i];
        i += 1;
    }
}
```

## 8. Practical Behavior
*   **Memory Bound**: These operations are usually limited by how fast you can read RAM, not how fast the CPU calculates.
*   **In-Place vs Out-of-Place**: `a += b` (In-Place) is faster because it reuses the cache line of `a`. `c = a + b` allocates new memory, which is expensive.

## 9. Tuning / Debugging Tips
*   **Check Alignment**: SIMD loads (`load_ps`) crash if memory isn't 32-byte aligned. Use unaligned loads (`loadu_ps`) for safety unless you control the allocator.
*   **Fused Ops**: If doing $A + B \times C$, doing it in two passes reads memory twice. Fusing it into one loop ($A + (B \times C)$) is 2x faster.

## 10. Historical Notes
The separation of scalar coprocessors (8087) and later integration of MMX/SSE instructions marked the beginning of consumer-grade high-performance computing. Deep Learning effectively resurrected the need for massive vector throughput.

## 11. Variants / Related Forms
*   **Unary Ops**: $y = f(x)$ (Log, Exp, Sqrt).
*   **Ternary Ops**: $z = ax + y$ (FMA - Fused Multiply Add).
*   **Reduction**: Technically not elementwise, but often fused with it (e.g., Sum of Squares).

## 12. Examples / Exercises
**Exercise**: Implementing ReLU.
$$ \text{ReLU}(x) = \max(0, x) $$
*Scalar*: `if x > 0.0 { x } else { 0.0 }`.
*SIMD*: `_mm256_max_ps(vec, _mm256_setzero_ps())`.
*Note*: The SIMD version avoids a CPU "Branch Misprediction" which is a huge performance killer.

## 13. Failure Cases / Limitations
*   **NaN Propagation**: $NaN + 5 = NaN$. One bad number spoils the whole tensor.
*   **Underflow/Overflow**: $e^{100}$ overflows `f32`. $\log(0)$ is $-\infty$.

## 14. Applications
*   **Residual Connections**: `x = x + block(x)` (ResNet).
*   **Gating Mechanisms**: `x * sigmoid(gate)` (GLU/LSTM).
*   **Normalization**: `(x - mean) / std`.

## 15. Connections to Other Concepts
*   **Functional Programming**: Specifically `Map`.
*   **Memory Hierarchies**: The Arithmetic Intensity (Ops per Byte) is low (1:1), putting stress on L1/L2 bandwidth.

## 16. Frontier / Research Angle (Optional)
**Stochastic Rounding**: For low precision (BF16/FP8), simple rounding introduces bias. Stochastic rounding (probabilistic) preserves statistical properties of gradients during elementwise updates.

## 17. Glossary of Terms
*   **SIMD**: Single Instruction Multiple Data.
*   **FMA**: Fused Multiply-Add.
*   **Throughput**: How many elements processed per second.
*   **Latency**: How long one operation takes.

## 18. References / Further Reading
*   [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
*   [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
