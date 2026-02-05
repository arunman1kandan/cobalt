# Cobalt Learning Path Index

Welcome to Cobalt! This index provides a structured learning path through all documentation, organized from fundamentals to advanced topics.

## 🚀 Quick Start

**New to Cobalt?** Start here:
1. [intro.md](intro.md) - Overview and motivation (5 min read)
2. [110-phase-0-1-guide.md](110-phase-0-1-guide.md) - Complete implementation reference (30 min read)

**Want hands-on?** Run:
```bash
cargo run --release
```

---

## 📚 Recommended Learning Path

### Phase 0: Core Concepts (Beginner)

Start with these fundamentals in order:

1. **[intro.md](intro.md)** (5 min)
   - What is Cobalt?
   - Why build a tensor library?
   - High-level architecture

2. **[010-tensors.md](010-tensors.md)** (10 min)
   - What are tensors?
   - Multi-dimensional arrays
   - Shape, rank, and indexing
   - Memory layout fundamentals

3. **[015-datatypes.md](015-datatypes.md)** (8 min)
   - Type system basics
   - Float vs integer types
   - Precision trade-offs
   - When to use each dtype

4. **[020-shapes-and-math.md](020-shapes-and-math.md)** (15 min)
   - Shape arithmetic
   - Dimension manipulation
   - Strides and memory addressing
   - Contiguous vs non-contiguous layouts

5. **[030-ops-architecture.md](030-ops-architecture.md)** (12 min)
   - Operation design patterns
   - Type erasure architecture
   - Backend dispatch mechanism
   - Extensibility principles

### Phase 0: Operations (Intermediate)

Continue with operation implementations:

6. **[040-elementwise.md](040-elementwise.md)** (15 min)
   - Elementwise operations (add, mul)
   - Multi-dtype support
   - SIMD optimizations (AVX2/AVX512)
   - Performance benchmarks

7. **[070-broadcasting.md](070-broadcasting.md)** (20 min)
   - NumPy broadcasting rules
   - Shape compatibility
   - Dimension expansion
   - Common broadcasting patterns
   - **✅ COMPLETE - Phase 0**

8. **[050-matmul.md](050-matmul.md)** (18 min)
   - Matrix multiplication basics
   - O(n³) naive algorithm
   - Memory access patterns
   - Future: tiling and optimization

9. **[060-activations-softmax.md](060-activations-softmax.md)** (12 min)
   - ReLU activation (with integer support)
   - Numerically stable softmax
   - Gradient flow considerations
   - Use cases in neural networks

### Phase 1: Advanced Memory (Advanced)

Deep dive into memory management:

10. **[065-shared-memory-arc.md](065-shared-memory-arc.md)** (25 min)
    - Arc<Vec<u8>> architecture
    - Reference counting mechanics
    - Shared ownership patterns
    - Memory lifecycle
    - Trade-offs vs Box/Rc
    - **✅ COMPLETE - Phase 1**

11. **[080-views-and-slicing.md](080-views-and-slicing.md)** (15 min)
    - Zero-copy views overview
    - Why views matter
    - Common use cases
    - View vs copy trade-offs

12. **[085-view-implementation.md](085-view-implementation.md)** (35 min)
    - TensorView struct internals
    - Strided indexing algorithm
    - All view operations detailed:
      - slice(), transpose(), permute()
      - reshape(), flatten(), squeeze()
    - Performance characteristics
    - **✅ COMPLETE - Phase 1**

13. **[095-contiguity-materialization.md](095-contiguity-materialization.md)** (40 min)
    - Contiguity definition
    - Detection algorithm
    - Materialization process
    - SIMD requirements
    - Cache efficiency
    - Optimization strategies
    - **✅ COMPLETE - Phase 1**

### System Architecture (Advanced)

Understand the complete system:

14. **[105-tensor-architecture.md](105-tensor-architecture.md)** (45 min)
    - Complete system overview
    - Type erasure to operations to backends
    - Integration between components
    - Data flow through the stack
    - Design patterns and principles
    - **✅ COMPLETE - Phase 1**

15. **[110-phase-0-1-guide.md](110-phase-0-1-guide.md)** (60 min)
    - Complete implementation reference
    - File organization
    - All algorithms with code
    - Testing strategy
    - Extension guide
    - Common patterns
    - Debugging tips
    - **✅ YOUR MAIN REFERENCE**

### Future Work (Not Yet Implemented)

16. **[090-python-bindings.md](090-python-bindings.md)** (Planning stage)
    - PyO3 integration (future)
    - NumPy interop (future)
    - Python API design (future)

---

## 🎯 Learning Tracks

### Track 1: "I want to USE Cobalt"
Perfect for users who want to build with Cobalt:

1. [intro.md](intro.md)
2. [110-phase-0-1-guide.md](110-phase-0-1-guide.md) ⭐ **Start here**
3. [010-tensors.md](010-tensors.md)
4. [070-broadcasting.md](070-broadcasting.md)
5. [080-views-and-slicing.md](080-views-and-slicing.md)
6. Run `cargo run --release` and explore `src/main.rs`

**Time**: ~90 minutes

### Track 2: "I want to UNDERSTAND Cobalt internals"
For developers who want to understand how it works:

1. [intro.md](intro.md)
2. [010-tensors.md](010-tensors.md)
3. [015-datatypes.md](015-datatypes.md)
4. [020-shapes-and-math.md](020-shapes-and-math.md)
5. [030-ops-architecture.md](030-ops-architecture.md)
6. [065-shared-memory-arc.md](065-shared-memory-arc.md)
7. [085-view-implementation.md](085-view-implementation.md)
8. [095-contiguity-materialization.md](095-contiguity-materialization.md)
9. [105-tensor-architecture.md](105-tensor-architecture.md)

**Time**: ~3 hours

### Track 3: "I want to EXTEND Cobalt"
For contributors adding new features:

1. [110-phase-0-1-guide.md](110-phase-0-1-guide.md) - Section 9: "Extending the System" ⭐
2. [030-ops-architecture.md](030-ops-architecture.md) - Design patterns
3. [105-tensor-architecture.md](105-tensor-architecture.md) - System integration
4. Read specific operation files (040, 050, 060) for examples
5. Check `src/ops/` and `src/backend/cpu/` for code patterns

**Time**: ~2 hours

### Track 4: "I'm researching tensor systems"
For academics or researchers:

1. [010-tensors.md](010-tensors.md) - Fundamentals
2. [020-shapes-and-math.md](020-shapes-and-math.md) - Memory layout
3. [065-shared-memory-arc.md](065-shared-memory-arc.md) - Memory management
4. [095-contiguity-materialization.md](095-contiguity-materialization.md) - Performance theory
5. [105-tensor-architecture.md](105-tensor-architecture.md) - System design
6. [040-elementwise.md](040-elementwise.md) - SIMD optimizations

**Time**: ~4 hours

---

## 📖 By Topic

### Memory & Performance
- [020-shapes-and-math.md](020-shapes-and-math.md) - Strides and layout
- [065-shared-memory-arc.md](065-shared-memory-arc.md) - Arc reference counting
- [095-contiguity-materialization.md](095-contiguity-materialization.md) - Cache efficiency
- [040-elementwise.md](040-elementwise.md) - SIMD optimizations

### Type System
- [015-datatypes.md](015-datatypes.md) - 16 data types
- [030-ops-architecture.md](030-ops-architecture.md) - Type erasure
- [105-tensor-architecture.md](105-tensor-architecture.md) - Type dispatch

### Operations
- [040-elementwise.md](040-elementwise.md) - Add, Mul
- [070-broadcasting.md](070-broadcasting.md) - Shape expansion
- [050-matmul.md](050-matmul.md) - Matrix multiplication
- [060-activations-softmax.md](060-activations-softmax.md) - Activations

### Views & Slicing
- [080-views-and-slicing.md](080-views-and-slicing.md) - Overview
- [085-view-implementation.md](085-view-implementation.md) - Implementation
- [095-contiguity-materialization.md](095-contiguity-materialization.md) - Materialization

### System Design
- [030-ops-architecture.md](030-ops-architecture.md) - Operation patterns
- [105-tensor-architecture.md](105-tensor-architecture.md) - Complete architecture
- [110-phase-0-1-guide.md](110-phase-0-1-guide.md) - Implementation guide

---

## 🔍 Quick Reference

### I need to...

**Create a tensor**
→ [110-phase-0-1-guide.md § 4.1](110-phase-0-1-guide.md#41-tensor-creation)

**Understand broadcasting**
→ [070-broadcasting.md](070-broadcasting.md)

**Work with views**
→ [085-view-implementation.md](085-view-implementation.md)

**Fix performance issues**
→ [095-contiguity-materialization.md § 10](095-contiguity-materialization.md#10-performance-optimization)

**Add a new operation**
→ [110-phase-0-1-guide.md § 9.1](110-phase-0-1-guide.md#91-adding-a-new-operation)

**Debug view problems**
→ [110-phase-0-1-guide.md § 10.2](110-phase-0-1-guide.md#102-check-view-metadata)

**Understand Arc memory**
→ [065-shared-memory-arc.md](065-shared-memory-arc.md)

**See test examples**
→ [110-phase-0-1-guide.md § 6](110-phase-0-1-guide.md#6-testing-strategy)

---

## 📊 Documentation Statistics

- **Total Files**: 15 markdown documents
- **Total Content**: ~10,000+ lines
- **Coverage**: Phases 0-1 (100% complete)
- **Code Examples**: 200+ snippets
- **Diagrams**: Multiple ASCII/visual aids
- **Implementation**: ~3,500 lines of Rust

---

## ✅ Completion Status

### Phase 0 (Core Operations) - ✅ COMPLETE
- [x] Tensors & shapes
- [x] 16 data types
- [x] Elementwise ops (add, mul)
- [x] Broadcasting
- [x] Matrix multiplication
- [x] Activations (ReLU, Softmax)
- [x] SIMD optimizations

### Phase 1 (Views & Slicing) - ✅ COMPLETE
- [x] Arc-based shared memory
- [x] TensorView implementation
- [x] Zero-copy operations
- [x] Contiguity detection
- [x] Materialization
- [x] Full test coverage

### Phase 1.5 (Planned)
- [ ] Reduction operations (sum, mean, max, min)
- [ ] Batched MatMul (3D+)
- [ ] Optimized MatMul (tiling)
- [ ] More activations (GELU, Sigmoid, Tanh)

### Phase 2 (Future)
- [ ] Autograd
- [ ] Python bindings
- [ ] GPU backend

---

## 🎓 Study Tips

1. **Start with intro.md** - Get oriented first
2. **Hands-on learning** - Run `cargo run --release` early
3. **Code + docs together** - Read markdown while viewing `src/`
4. **Follow numbered order** - Files are numbered for a reason
5. **Don't skip 110** - The complete guide ties everything together
6. **Test as you learn** - Run `cargo test` to see concepts in action
7. **Use multiple tracks** - Combine tracks based on your goals

---

## 🚧 Work in Progress

None! Phases 0-1 are complete and documented.

**Next**: Phase 1.5 (reduction ops, batched matmul, optimizations)

---

## 📝 Contributing

Want to add documentation or improve existing notes?

1. Check [110-phase-0-1-guide.md § 9](110-phase-0-1-guide.md#9-extending-the-system)
2. Follow existing note structure (Motivation, Implementation, Examples, Summary)
3. Include code examples and ASCII diagrams
4. Update this index with your new file

---

## 🔗 External Resources

- **Rust**: https://doc.rust-lang.org/book/
- **NumPy Broadcasting**: https://numpy.org/doc/stable/user/basics.broadcasting.html
- **SIMD**: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- **Arc/Rc**: https://doc.rust-lang.org/std/sync/struct.Arc.html

---

**Last Updated**: February 5, 2026  
**Version**: Phase 0-1 Complete  
**Maintainer**: Cobalt Development Team

**Ready to learn?** Start with [intro.md](intro.md) or jump to [110-phase-0-1-guide.md](110-phase-0-1-guide.md)! 🚀
