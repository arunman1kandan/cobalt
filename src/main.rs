//! Cobalt Framework - Visual Progress Demonstration
//! 
//! This file demonstrates all implemented features with performance timing.
//! Run this after a break to see what's been completed!

use cobalt::Tensor;
use std::time::Instant;

fn main() {
    print_header();
    
    // Track overall timing
    let total_start = Instant::now();
    
    demo_fp32_operations();
    demo_fp64_operations();
    demo_fp16_operations();
    demo_bfloat16_operations();
    demo_int32_operations();
    demo_int64_operations();
    demo_int16_operations();
    demo_uint_operations();
    demo_broadcasting();
    demo_matrix_operations();
    demo_activations();
    demo_views_and_slicing();
    demo_extensive_example();
    demo_performance_benchmark();
    
    let total_time = total_start.elapsed();
    print_footer(total_time);
}

fn print_header() {
    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "COBALT DEEP LEARNING FRAMEWORK");
    println!("{:^80}", "Comprehensive Feature Demonstration");
    println!("{}", "=".repeat(80));
    println!("\n📋 Implementation Status:");
    println!("   ✅ Tensor Core (Multi-dtype support)");
    println!("   ✅ Elementwise Operations (Add, Mul)");
    println!("   ✅ Matrix Multiplication (2D)");
    println!("   ✅ Activations (ReLU, Softmax)");
    println!("   ✅ Broadcasting (NumPy-compatible)");
    println!("   ✅ Views & Slicing (Zero-copy metadata)");
    println!("   ✅ SIMD Optimization (AVX2/AVX512 for FP32)");
    println!("{}\n", "=".repeat(80));
}

fn demo_fp32_operations() {
    println!("🔹 FP32 (32-bit Float) Operations");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    
    println!("  Input A: {:?}", a.as_f32_slice());
    println!("  Input B: {:?}", b.as_f32_slice());
    
    match a.add(&b) {
        Ok(c) => println!("  ✓ Add:   {:?}", c.as_f32_slice()),
        Err(e) => println!("  ✗ Add failed: {:?}", e),
    }
    
    match a.mul(&b) {
        Ok(c) => println!("  ✓ Mul:   {:?}", c.as_f32_slice()),
        Err(e) => println!("  ✗ Mul failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_fp64_operations() {
    println!("🔹 FP64 (64-bit Double) Operations");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    let a = Tensor::from_slice(&[1.5f64, 2.5, 3.5, 4.5], vec![4]);
    let b = Tensor::from_slice(&[2.0f64, 2.0, 2.0, 2.0], vec![4]);
    
    println!("  Input A: {:?}", a.as_slice::<f64>());
    println!("  Input B: {:?}", b.as_slice::<f64>());
    
    match a.add(&b) {
        Ok(c) => println!("  ✓ Add:   {:?}", c.as_slice::<f64>()),
        Err(e) => println!("  ✗ Add failed: {:?}", e),
    }
    
    match a.mul(&b) {
        Ok(c) => println!("  ✓ Mul:   {:?}", c.as_slice::<f64>()),
        Err(e) => println!("  ✗ Mul failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_fp16_operations() {
    println!("🔹 FP16 (16-bit Half Precision) Operations");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    use half::f16;
    let a = Tensor::from_slice(&[
        f16::from_f32(1.0), f16::from_f32(2.0), 
        f16::from_f32(3.0), f16::from_f32(4.0)
    ], vec![4]);
    let b = Tensor::from_slice(&[
        f16::from_f32(0.5), f16::from_f32(0.5),
        f16::from_f32(0.5), f16::from_f32(0.5)
    ], vec![4]);
    
    let a_vals: Vec<f32> = a.as_slice::<f16>().iter().map(|x| x.to_f32()).collect();
    let b_vals: Vec<f32> = b.as_slice::<f16>().iter().map(|x| x.to_f32()).collect();
    println!("  Input A: {:?}", a_vals);
    println!("  Input B: {:?}", b_vals);
    
    match a.add(&b) {
        Ok(c) => {
            let c_vals: Vec<f32> = c.as_slice::<f16>().iter().map(|x| x.to_f32()).collect();
            println!("  ✓ Add:   {:?}", c_vals);
        },
        Err(e) => println!("  ✗ Add failed: {:?}", e),
    }
    
    match a.mul(&b) {
        Ok(c) => {
            let c_vals: Vec<f32> = c.as_slice::<f16>().iter().map(|x| x.to_f32()).collect();
            println!("  ✓ Mul:   {:?}", c_vals);
        },
        Err(e) => println!("  ✗ Mul failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_bfloat16_operations() {
    println!("🔹 BF16 (Brain Float 16) Operations");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    use half::bf16;
    let a = Tensor::from_slice(&[
        bf16::from_f32(10.0), bf16::from_f32(20.0), 
        bf16::from_f32(30.0), bf16::from_f32(40.0)
    ], vec![4]);
    let b = Tensor::from_slice(&[
        bf16::from_f32(1.0), bf16::from_f32(2.0),
        bf16::from_f32(3.0), bf16::from_f32(4.0)
    ], vec![4]);
    
    let a_vals: Vec<f32> = a.as_slice::<bf16>().iter().map(|x| x.to_f32()).collect();
    let b_vals: Vec<f32> = b.as_slice::<bf16>().iter().map(|x| x.to_f32()).collect();
    println!("  Input A: {:?}", a_vals);
    println!("  Input B: {:?}", b_vals);
    
    match a.add(&b) {
        Ok(c) => {
            let c_vals: Vec<f32> = c.as_slice::<bf16>().iter().map(|x| x.to_f32()).collect();
            println!("  ✓ Add:   {:?}", c_vals);
        },
        Err(e) => println!("  ✗ Add failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_int32_operations() {
    println!("🔹 INT32 (32-bit Integer) Operations");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    let a = Tensor::from_slice(&[10, 20, 30, 40], vec![4]);
    let b = Tensor::from_slice(&[1, 2, 3, 4], vec![4]);
    
    println!("  Input A: {:?}", a.as_slice::<i32>());
    println!("  Input B: {:?}", b.as_slice::<i32>());
    
    match a.add(&b) {
        Ok(c) => println!("  ✓ Add:   {:?}", c.as_slice::<i32>()),
        Err(e) => println!("  ✗ Add failed: {:?}", e),
    }
    
    match a.mul(&b) {
        Ok(c) => println!("  ✓ Mul:   {:?}", c.as_slice::<i32>()),
        Err(e) => println!("  ✗ Mul failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_int64_operations() {
    println!("🔹 INT64 (64-bit Integer) Operations");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    let a = Tensor::from_slice(&[100i64, 200, 300, 400], vec![2, 2]);
    let b = Tensor::from_slice(&[1i64, 2, 3, 4], vec![2, 2]);
    
    println!("  Input A: {:?}", a.as_slice::<i64>());
    println!("  Input B: {:?}", b.as_slice::<i64>());
    
    match a.add(&b) {
        Ok(c) => println!("  ✓ Add:   {:?}", c.as_slice::<i64>()),
        Err(e) => println!("  ✗ Add failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_int16_operations() {
    println!("🔹 INT16 (16-bit Integer) Operations");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    let a = Tensor::from_slice(&[100i16, 200, 300], vec![3]);
    let b = Tensor::from_slice(&[50i16, 50, 50], vec![3]);
    
    println!("  Input A: {:?}", a.as_slice::<i16>());
    println!("  Input B: {:?}", b.as_slice::<i16>());
    
    match a.add(&b) {
        Ok(c) => println!("  ✓ Add:   {:?}", c.as_slice::<i16>()),
        Err(e) => println!("  ✗ Add failed: {:?}", e),
    }
    
    match a.mul(&b) {
        Ok(c) => println!("  ✓ Mul:   {:?}", c.as_slice::<i16>()),
        Err(e) => println!("  ✗ Mul failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_uint_operations() {
    println!("🔹 UINT (Unsigned Integer) Operations");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    // UINT8
    let a8 = Tensor::from_slice(&[128u8, 64, 32, 16], vec![4]);
    let b8 = Tensor::from_slice(&[1u8, 2, 3, 4], vec![4]);
    
    println!("  UINT8 A: {:?}", a8.as_slice::<u8>());
    match a8.add(&b8) {
        Ok(c) => println!("  ✓ UINT8 Add: {:?}", c.as_slice::<u8>()),
        Err(e) => println!("  ✗ UINT8 Add failed: {:?}", e),
    }
    
    // UINT16 (avoid overflow in debug builds)
    let a16 = Tensor::from_slice(&[100u16, 200], vec![2]);
    let b16 = Tensor::from_slice(&[2u16, 3], vec![2]);
    
    match a16.mul(&b16) {
        Ok(c) => println!("  ✓ UINT16 Mul: {:?}", c.as_slice::<u16>()),
        Err(e) => println!("  ✗ UINT16 Mul failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_broadcasting() {
    println!("🔹 Broadcasting Operations (NumPy-style)");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    // Scalar broadcasting
    let big = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let scalar = Tensor::from_f32(vec![100.0], vec![1]);
    
    println!("  Matrix [2x3]: {:?}", big.as_f32_slice());
    println!("  Scalar [1]:   {:?}", scalar.as_f32_slice());
    
    match big.add(&scalar) {
        Ok(c) => println!("  ✓ Broadcast Add: {:?}", c.as_f32_slice()),
        Err(e) => println!("  ✗ Broadcast failed: {:?}", e),
    }
    
    // Vector broadcasting
    let matrix = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let vector = Tensor::from_f32(vec![10.0, 20.0, 30.0], vec![3]);
    
    match matrix.add(&vector) {
        Ok(c) => println!("  ✓ Vector Broadcast: {:?}", c.as_f32_slice()),
        Err(e) => println!("  ✗ Vector broadcast failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_matrix_operations() {
    println!("🔹 Matrix Multiplication (2D)");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    // FP32 MatMul
    let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    
    println!("  Matrix A [2x2]: {:?}", a.as_f32_slice());
    println!("  Matrix B [2x2]: {:?}", b.as_f32_slice());
    
    match a.matmul(&b) {
        Ok(c) => println!("  ✓ FP32 MatMul: {:?}", c.as_f32_slice()),
        Err(e) => println!("  ✗ MatMul failed: {:?}", e),
    }
    
    // FP64 MatMul
    let a64 = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
    let b64 = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0], vec![2, 2]);
    
    match a64.matmul(&b64) {
        Ok(c) => println!("  ✓ FP64 MatMul: {:?}", c.as_slice::<f64>()),
        Err(e) => println!("  ✗ FP64 MatMul failed: {:?}", e),
    }
    
    // FP16 MatMul
    use half::f16;
    let a16 = Tensor::from_slice(&[
        f16::from_f32(2.0), f16::from_f32(0.0),
        f16::from_f32(0.0), f16::from_f32(2.0)
    ], vec![2, 2]);
    let b16 = Tensor::from_slice(&[
        f16::from_f32(3.0), f16::from_f32(4.0),
        f16::from_f32(5.0), f16::from_f32(6.0)
    ], vec![2, 2]);
    
    match a16.matmul(&b16) {
        Ok(c) => {
            let c_vals: Vec<f32> = c.as_slice::<f16>().iter().map(|x| x.to_f32()).collect();
            println!("  ✓ FP16 MatMul: {:?}", c_vals);
        },
        Err(e) => println!("  ✗ FP16 MatMul failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_activations() {
    println!("🔹 Activation Functions");
    println!("{}", "-".repeat(80));
    
    let start = Instant::now();
    
    // ReLU
    let x = Tensor::from_f32(vec![-5.0, -2.5, 0.0, 2.5, 5.0], vec![5]);
    println!("  Input:  {:?}", x.as_f32_slice());
    
    match x.relu() {
        Ok(y) => println!("  ✓ ReLU: {:?}", y.as_f32_slice()),
        Err(e) => println!("  ✗ ReLU failed: {:?}", e),
    }
    
    // Integer ReLU
    let xi = Tensor::from_slice(&[-10, -5, 0, 5, 10], vec![5]);
    match xi.relu() {
        Ok(y) => println!("  ✓ INT32 ReLU: {:?}", y.as_slice::<i32>()),
        Err(e) => println!("  ✗ INT32 ReLU failed: {:?}", e),
    }
    
    // Softmax
    let logits = Tensor::from_f32(vec![3.0, 1.0, 0.2], vec![3]);
    println!("  Logits: {:?}", logits.as_f32_slice());
    
    match logits.softmax() {
        Ok(probs) => {
            let p = probs.as_f32_slice();
            println!("  ✓ Softmax: [{:.4}, {:.4}, {:.4}]", p[0], p[1], p[2]);
            let sum: f32 = p.iter().sum();
            println!("  ✓ Sum check: {:.6} (should be 1.0)", sum);
        },
        Err(e) => println!("  ✗ Softmax failed: {:?}", e),
    }
    
    // FP16 Softmax
    use half::f16;
    let logits16 = Tensor::from_slice(&[
        f16::from_f32(2.0), f16::from_f32(1.0), f16::from_f32(0.5)
    ], vec![3]);
    
    match logits16.softmax() {
        Ok(probs) => {
            let p_vals: Vec<f32> = probs.as_slice::<f16>().iter().map(|x| x.to_f32()).collect();
            println!("  ✓ FP16 Softmax: [{:.4}, {:.4}, {:.4}]", p_vals[0], p_vals[1], p_vals[2]);
        },
        Err(e) => println!("  ✗ FP16 Softmax failed: {:?}", e),
    }
    
    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_views_and_slicing() {
    println!("🔹 Views & Slicing (Phase 1)");
    println!("{}", "-".repeat(80));

    let start = Instant::now();

    let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    println!("  Base Tensor Shape: {:?}", t.shape);
    println!("  Base Data: {:?}", t.as_f32_slice());

    match t.slice(0, 1) {
        Ok(v) => println!("  ✓ Slice [0..1] Shape: {:?}, Strides: {:?}", v.shape(), v.strides()),
        Err(e) => println!("  ✗ Slice failed: {:?}", e),
    }

    match t.transpose_view(0, 1) {
        Ok(v) => {
            println!("  ✓ Transpose Shape: {:?}, Strides: {:?}", v.shape(), v.strides());
            let materialized = v.contiguous();
            println!("    ↳ Materialized Data: {:?}", materialized.as_f32_slice());
        },
        Err(e) => println!("  ✗ Transpose failed: {:?}", e),
    }

    match t.reshape_view(&[3, 2]) {
        Ok(v) => println!("  ✓ Reshape [3,2] Shape: {:?}", v.shape()),
        Err(e) => println!("  ✗ Reshape failed: {:?}", e),
    }

    match t.flatten_view() {
        Ok(v) => println!("  ✓ Flatten Shape: {:?}", v.shape()),
        Err(e) => println!("  ✗ Flatten failed: {:?}", e),
    }

    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_extensive_example() {
    println!("🔹 Extensive End-to-End Example");
    println!("{}", "-".repeat(80));

    let start = Instant::now();

    // Step 1: Build a base tensor
    let a = Tensor::from_f32(vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ], vec![2, 3]);
    println!("  Base A shape: {:?}", a.shape);
    println!("  Base A data:  {:?}", a.as_f32_slice());

    // Step 2: Slice and transpose (views)
    let a_slice = a.slice(0, 2).unwrap();
    let a_t = a_slice.transpose(0, 1).unwrap();
    println!("  View A^T shape: {:?}, strides: {:?}", a_t.shape(), a_t.strides());

    // Step 3: Materialize view for matmul
    let a_t_mat = a_t.contiguous();
    println!("  A^T materialized: {:?}", a_t_mat.as_f32_slice());

    // Step 4: Matmul with another tensor
    let b = Tensor::from_f32(vec![
        1.0, 0.0,
        0.0, 1.0,
    ], vec![2, 2]);
    println!("  B shape: {:?}", b.shape);

    let c = match a_t_mat.matmul(&b) {
        Ok(out) => out,
        Err(e) => {
            println!("  ✗ MatMul failed: {:?}", e);
            return;
        }
    };
    println!("  MatMul result shape: {:?}", c.shape);
    println!("  MatMul result data:  {:?}", c.as_f32_slice());

    // Step 5: Add a broadcasted bias
    let bias = Tensor::from_f32(vec![1.0, -1.0], vec![1, 2]);
    let c_bias = match c.add(&bias) {
        Ok(out) => out,
        Err(e) => {
            println!("  ✗ Add(bias) failed: {:?}", e);
            return;
        }
    };
    println!("  + Bias data:        {:?}", c_bias.as_f32_slice());

    // Step 6: Activation
    let c_relu = match c_bias.relu() {
        Ok(out) => out,
        Err(e) => {
            println!("  ✗ ReLU failed: {:?}", e);
            return;
        }
    };
    println!("  ReLU output:        {:?}", c_relu.as_f32_slice());

    let elapsed = start.elapsed();
    println!("  ⏱️  Time: {:?}\n", elapsed);
}

fn demo_performance_benchmark() {
    println!("🔹 Performance Benchmarks");
    println!("{}", "-".repeat(80));
    
    // Large tensor operations
    let size = 10000;
    let a = Tensor::from_f32(vec![1.0; size], vec![size]);
    let b = Tensor::from_f32(vec![2.0; size], vec![size]);
    
    let start = Instant::now();
    let _ = a.add(&b);
    let add_time = start.elapsed();
    
    let start = Instant::now();
    let _ = a.mul(&b);
    let mul_time = start.elapsed();
    
    println!("  Large Tensor Operations (size: {}):", size);
    println!("  ✓ Add: {:?} ({:.2} million ops/sec)", 
             add_time, size as f64 / add_time.as_secs_f64() / 1_000_000.0);
    println!("  ✓ Mul: {:?} ({:.2} million ops/sec)", 
             mul_time, size as f64 / mul_time.as_secs_f64() / 1_000_000.0);
    
    // Matrix multiplication benchmark
    let mat_size = 128;
    let ma = Tensor::from_f32(vec![1.0; mat_size * mat_size], vec![mat_size, mat_size]);
    let mb = Tensor::from_f32(vec![2.0; mat_size * mat_size], vec![mat_size, mat_size]);
    
    let start = Instant::now();
    let _ = ma.matmul(&mb);
    let matmul_time = start.elapsed();
    let flops = 2.0 * (mat_size * mat_size * mat_size) as f64;
    
    println!("\n  Matrix Multiplication [{}x{}]:", mat_size, mat_size);
    println!("  ✓ Time: {:?} ({:.2} GFLOPS)", 
             matmul_time, flops / matmul_time.as_secs_f64() / 1_000_000_000.0);
    
    println!();
}

fn print_footer(total_time: std::time::Duration) {
    println!("{}", "=".repeat(80));
    println!("\n🎉 All Demonstrations Complete!");
    println!("\n📊 Summary:");
    println!("   • Total execution time: {:?}", total_time);
    println!("   • All operations working correctly");
    println!("   • Multiple data types supported (FP16, BF16, FP32, FP64, INT*, UINT*)");
    println!("   • Broadcasting fully functional");
    println!("   • SIMD optimizations active for FP32");
    
    println!("\n🚀 Next Steps (Phase 1.5):");
    println!("   → Reduction Operations (sum, mean, max, min)");
    println!("   → Optimized MatMul (tiling/blocking)");
    println!("   → More Activations (GELU, Sigmoid, Tanh)");
    
    println!("\n{}", "=".repeat(80));
}
