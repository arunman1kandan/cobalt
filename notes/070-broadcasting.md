# 070: Broadcasting

## 1. Motivation
In Deep Learning, we frequently need to operations between tensors of different sizes.
*   Subtracting the **Mean** (scalar) from an **Image** (matrix).
*   Adding a **Bias** (vector) to a **Batch** of features (matrix).
Explicitly copying the data to match shapes (e.g., repeating the vector 32 times) is wasteful in both Memory and Memory Bandwidth. **Broadcasting** provides a zero-cost abstraction to handle this.

## 2. Context / Precedence
*   **APL / J**: The origin of rank-polymorphic operations.
*   **NumPy (2006)**: Standardized the "General Broadcasting Rules" used today by PyTorch, TensorFlow, MXNet, and JAX.
*   **MATLAB**: Historically required `bsxfun` (Binary Singleton Expansion Function), which was clunky.

## 3. Intuition
**The Virtual Clone**:
Imagine you want to paint a grid of 100 boxes Red.
*   **Copying**: Buying 100 buckets of Red paint.
*   **Broadcasting**: Buying 1 bucket and telling the universe "This bucket is everywhere."

The computer doesn't actualy duplicate the number. It just "virtualizes" the access pattern.

## 4. Formal Definition
Given two shapes $S_A$ and $S_B$.
1.  **Right Alignment**: Pad the shorter shape with 1s on the left until rank matches.
2.  **Compatibility**: For each dimension $i$, dimensions are compatible if:
    $$ s_{A,i} == s_{B,i} \lor s_{A,i} == 1 \lor s_{B,i} == 1 $$
3.  **Result Shape**:
    $$ s_{C,i} = \max(s_{A,i}, s_{B,i}) $$

## 5. Mathematical Deep Dive
**Virtual Striding**:
If a dimension has size 1, we set its **Stride** to 0.
Recall index formula: $k = \sum i_j \cdot w_j$.
If $w_j = 0$, then incrementing the index $i_j$ does not move the memory pointer. We read the same value over and over.

$$ \text{Stride}_{\text{broadcast}} = \text{Stride}_{\text{original}} \times \mathbb{I}(\text{Size} > 1) $$

## 6. Computation / Implementation Details
*   **Pre-Computation**: During the Shape Check phase, we calculate the stride vector for both inputs.
*   **Kernel Loop**: The inner loop doesn't know about broadcasting. It just uses `ptr + i * stride`. If stride is 0, it works naturally.
*   **Coalescing**: Broadcasting breaks memory coalescing if the inner-most dimension is broadcasted (stride 0). This is a known performance killer on GPUs ("Bank Conflicts").

## 7. Minimal Code

### Rust Logic
```rust
fn broadcast_stride(shape: &[usize], original_strides: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let mut new_strides = vec![0; target_shape.len()];
    let offset = target_shape.len() - shape.len();
    
    for i in 0..shape.len() {
        if shape[i] == 1 {
            new_strides[i + offset] = 0; // The Magic: Stride is 0
        } else {
            new_strides[i + offset] = original_strides[i];
        }
    }
    new_strides
}
```

## 8. Practical Behavior
*   **Implicit vs Explicit**: Python does it implicitly (`a + b`). Some languages (Julia) require explicit dots (`a .+ b`).
*   **Memory Savings**: Broadcasting `[1000, 1000]` + `[1000]` saves 4MB of RAM allocation compared to expanding.

## 9. Tuning / Debugging Tips
*   **The Trailing Dim Trap**: `[32, 10]` + `[32]` fails because `[32]` aligns to the right: `[32, 10]` vs `[1, 32]`. Mismatch (10 vs 32). You must reshape `[32]` to `[32, 1]`.
*   **In-Place Ops**: You cannot broadcast in-place into a smaller tensor. `small += huge` is illegal.

## 10. Historical Notes
The elegance of NumPy's broadcasting rules is largely credited as a major reason Python displaced MATLAB in data science. It allowed writing mathematical expressions that looked like scalar math.

## 11. Variants / Related Forms
*   **Outer Product**: An extreme form of broadcasting. `[N, 1] * [1, M] -> [N, M]`.
*   **Meshgrid**: Function that explicitly generates coordinate matrices (useful for plotting).

## 12. Examples / Exercises
**Exercise**: Determine Shape.
A: `[8, 1, 6, 1]`
B: `[   7, 1, 5]`
1. Pad B: `[1, 7, 1, 5]`
2. Match:
   - 1 vs 5 -> 5
   - 6 vs 1 -> 6
   - 1 vs 7 -> 7
   - 8 vs 1 -> 8
Result: `[8, 7, 6, 5]`.

## 13. Failure Cases / Limitations
*   **Ambiguity**: `[32]` + `[32]` works. `[32]` + `[32, 32]` works. Sometimes you want an error but get a broadcast.
*   **Performance (GPU)**: Broadcasting a scalar to a huge matrix is memory-bandwidth bound (reading the matrix). The arithmetic density is very low.

## 14. Applications
*   **Bias Addition**: Adding `b` term in Linear/Conv layers.
*   **Normalization**: Subtracting mean `[C, 1, 1]` from Image `[C, H, W]`.
*   **Masking**: Multiplying `[Seq, Seq]` attention scores by `[1, Seq]` mask.

## 15. Connections to Other Concepts
*   **Lazy Evaluation**: Broadcasting is a form of lazy data expansion.
*   **Views**: Broadcasting creates a View, not a Copy.

## 16. Frontier / Research Angle (Optional)
**Lazy Tensors**: Systems like TensorFlow XLA fuse the broadcast into the generating kernel (e.g., generating random noise + scalar) so that the data is never even written to memory in the first place.

## 17. Glossary of Terms
*   **Singleton Dimension**: A dimension with size 1.
*   **Expansion**: The conceptual act of repeating data.

## 18. References / Further Reading
*   [NumPy Broadcasting Docs](https://numpy.org/doc/stable/user/basics.broadcasting.html)
*   [Ezyang's Broadcasting Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
