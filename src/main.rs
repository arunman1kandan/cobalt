mod dtype;
mod device;
mod tensor;
mod backend;
mod ops;
mod errors;
mod broadcast;


//! Cobalt: A tiny deep learning framework for learning and exploration.
//!
//! This entry point serves as a scratchpad for testing new features and
//! verifying implementation logic manually.

use tensor::Tensor;

/// Main entry point.
///
/// Currently used to demonstrate datatype support and tensor operations.
/// Note: Some operations (mul, matmul, relu, softmax) are commented out
/// as they differ during the migration to the new backend system.
fn main() {
    let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    println!("a = {}", a);
    println!("b = {}", b);

    println!("a+b = {:?}", a.add(&b));
    println!("a+b = {:?}", a.add(&b));
    println!("a+b = {:?}", a.add(&b));
    // println!("a*b = {:?}", a.mul(&b));
    // println!("a@b = {:?}", a.matmul(&b));
    // println!("relu(a) = {:?}", a.relu());
    // println!("relu(a) = {:?}", a.relu());
    // println!("softmax(a) = {:?}", a.softmax());

    // Test INT32
    let i_a = Tensor::from_slice(&[1, 2, 3, 4], vec![2, 2]);
    let i_b = Tensor::from_slice(&[10, 20, 30, 40], vec![2, 2]);
    println!("i_a = {:?}", i_a);
    println!("i_a + i_b = {:?}", i_a.add(&i_b));

    // Test mixing (should fail)
    // println!("a + i_a = {:?}", a.add(&i_a)); // Should panic or error

}
