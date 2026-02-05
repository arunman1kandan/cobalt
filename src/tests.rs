//! Comprehensive test suite for Cobalt tensor operations.

#[cfg(test)]
mod tensor_tests {
    use crate::tensor::Tensor;
    use crate::dtype::DType;

    #[test]
    fn test_tensor_creation_fp32() {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.dtype, DType::FP32);
        assert_eq!(t.numel(), 4);
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn test_tensor_creation_int32() {
        let t = Tensor::from_slice(&[1, 2, 3, 4, 5, 6], vec![2, 3]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.dtype, DType::INT32);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_tensor_creation_fp64() {
        let t = Tensor::from_slice(&[1.0f64, 2.0, 3.0], vec![3]);
        assert_eq!(t.dtype, DType::FP64);
        assert_eq!(t.numel(), 3);
    }

    #[test]
    #[should_panic]
    fn test_tensor_creation_shape_mismatch() {
        // Should panic because data has 4 elements but shape expects 6
        let _t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 3]);
    }
}

#[cfg(test)]
mod add_tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_add_fp32_same_shape() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        let c = a.add(&b).unwrap();
        let result = c.as_f32_slice();
        
        assert_eq!(result, &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_add_int32_same_shape() {
        let a = Tensor::from_slice(&[1, 2, 3, 4], vec![4]);
        let b = Tensor::from_slice(&[10, 20, 30, 40], vec![4]);
        
        let c = a.add(&b).unwrap();
        let result = c.as_slice::<i32>();
        
        assert_eq!(result, &[11, 22, 33, 44]);
    }

    #[test]
    fn test_add_int64() {
        let a = Tensor::from_slice(&[100i64, 200, 300], vec![3]);
        let b = Tensor::from_slice(&[1i64, 2, 3], vec![3]);
        
        let c = a.add(&b).unwrap();
        let result = c.as_slice::<i64>();
        
        assert_eq!(result, &[101, 202, 303]);
    }

    #[test]
    fn test_add_fp64() {
        let a = Tensor::from_slice(&[1.5f64, 2.5], vec![2]);
        let b = Tensor::from_slice(&[0.5f64, 1.5], vec![2]);
        
        let c = a.add(&b).unwrap();
        let result = c.as_slice::<f64>();
        
        assert_eq!(result, &[2.0, 4.0]);
    }

    #[test]
    fn test_add_broadcast_scalar() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_f32(vec![10.0], vec![1]);
        
        let c = a.add(&b).unwrap();
        let result = c.as_f32_slice();
        
        assert_eq!(result, &[11.0, 12.0, 13.0, 14.0]);
    }

    #[test]
    fn test_add_broadcast_2d() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_f32(vec![10.0, 20.0, 30.0], vec![3]);
        
        let c = a.add(&b).unwrap();
        let result = c.as_f32_slice();
        
        assert_eq!(result, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }
}

#[cfg(test)]
mod mul_tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_mul_fp32() {
        let a = Tensor::from_f32(vec![2.0, 3.0, 4.0], vec![3]);
        let b = Tensor::from_f32(vec![5.0, 6.0, 7.0], vec![3]);
        
        let c = a.mul(&b).unwrap();
        let result = c.as_f32_slice();
        
        assert_eq!(result, &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_mul_int32() {
        let a = Tensor::from_slice(&[2, 3, 4], vec![3]);
        let b = Tensor::from_slice(&[10, 10, 10], vec![3]);
        
        let c = a.mul(&b).unwrap();
        let result = c.as_slice::<i32>();
        
        assert_eq!(result, &[20, 30, 40]);
    }

    #[test]
    fn test_mul_broadcast() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::from_f32(vec![2.0], vec![1]);
        
        let c = a.mul(&b).unwrap();
        let result = c.as_f32_slice();
        
        assert_eq!(result, &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_mul_fp64() {
        let a = Tensor::from_slice(&[1.5f64, 2.0, 2.5], vec![3]);
        let b = Tensor::from_slice(&[2.0f64, 2.0, 2.0], vec![3]);
        
        let c = a.mul(&b).unwrap();
        let result = c.as_slice::<f64>();
        
        assert_eq!(result, &[3.0, 4.0, 5.0]);
    }
}

#[cfg(test)]
mod matmul_tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_matmul_2x2() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_f32(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        
        let c = a.matmul(&b).unwrap();
        let result = c.as_f32_slice();
        
        // [1 2] @ [5 6]  =  [1*5+2*7  1*6+2*8]  =  [19 22]
        // [3 4]   [7 8]     [3*5+4*7  3*6+4*8]     [43 50]
        assert_eq!(result, &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_non_square() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_f32(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        
        let result = c.as_f32_slice();
        // [1 2 3] @ [7  8]   =  [1*7+2*9+3*11   1*8+2*10+3*12]  =  [58  64]
        // [4 5 6]   [9  10]     [4*7+5*9+6*11   4*8+5*10+6*12]     [139 154]
        //           [11 12]
        assert_eq!(result, &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_matmul_fp64() {
        let a = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_slice(&[1.0f64, 0.0, 0.0, 1.0], vec![2, 2]);
        
        let c = a.matmul(&b).unwrap();
        let result = c.as_slice::<f64>();
        
        // Identity matrix multiplication
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_matmul_shape_mismatch() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3, 1]);
        
        assert!(a.matmul(&b).is_err());
    }
}

#[cfg(test)]
mod relu_tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_relu_fp32() {
        let x = Tensor::from_f32(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let y = x.relu().unwrap();
        let result = y.as_f32_slice();
        
        assert_eq!(result, &[0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_int32() {
        let x = Tensor::from_slice(&[-10, -5, 0, 5, 10], vec![5]);
        let y = x.relu().unwrap();
        let result = y.as_slice::<i32>();
        
        assert_eq!(result, &[0, 0, 0, 5, 10]);
    }

    #[test]
    fn test_relu_fp64() {
        let x = Tensor::from_slice(&[-1.5f64, 0.0, 1.5], vec![3]);
        let y = x.relu().unwrap();
        let result = y.as_slice::<f64>();
        
        assert_eq!(result, &[0.0, 0.0, 1.5]);
    }

    #[test]
    fn test_relu_all_positive() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let y = x.relu().unwrap();
        let result = y.as_f32_slice();
        
        assert_eq!(result, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_relu_all_negative() {
        let x = Tensor::from_f32(vec![-1.0, -2.0, -3.0, -4.0], vec![4]);
        let y = x.relu().unwrap();
        let result = y.as_f32_slice();
        
        assert_eq!(result, &[0.0, 0.0, 0.0, 0.0]);
    }
}

#[cfg(test)]
mod softmax_tests {
    use crate::tensor::Tensor;

    fn assert_close(a: f32, b: f32, epsilon: f32) {
        assert!((a - b).abs() < epsilon, "{} != {} (within {})", a, b, epsilon);
    }

    #[test]
    fn test_softmax_1d() {
        let x = Tensor::from_f32(vec![2.0, 1.0, 0.1], vec![3]);
        let y = x.softmax().unwrap();
        let result = y.as_f32_slice();
        
        // Check sum is 1.0
        let sum: f32 = result.iter().sum();
        assert_close(sum, 1.0, 1e-6);
        
        // Check relative ordering (higher input -> higher output)
        assert!(result[0] > result[1]);
        assert!(result[1] > result[2]);
    }

    #[test]
    fn test_softmax_2d() {
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y = x.softmax().unwrap();
        let result = y.as_f32_slice();
        
        // Check each row sums to 1.0
        let row1_sum: f32 = result[0..3].iter().sum();
        let row2_sum: f32 = result[3..6].iter().sum();
        assert_close(row1_sum, 1.0, 1e-6);
        assert_close(row2_sum, 1.0, 1e-6);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values that would overflow without the max subtraction trick
        let x = Tensor::from_f32(vec![1000.0, 1001.0, 1002.0], vec![3]);
        let y = x.softmax().unwrap();
        let result = y.as_f32_slice();
        
        // Should not contain NaN or Inf
        assert!(result.iter().all(|&v| v.is_finite()));
        
        // Sum should still be 1.0
        let sum: f32 = result.iter().sum();
        assert_close(sum, 1.0, 1e-5);
    }

    #[test]
    fn test_softmax_uniform() {
        // All equal inputs should produce uniform distribution
        let x = Tensor::from_f32(vec![5.0, 5.0, 5.0, 5.0], vec![4]);
        let y = x.softmax().unwrap();
        let result = y.as_f32_slice();
        
        for &val in result {
            assert_close(val, 0.25, 1e-6);
        }
    }

    #[test]
    fn test_softmax_fp64() {
        let x = Tensor::from_slice(&[1.0f64, 2.0, 3.0], vec![3]);
        let y = x.softmax().unwrap();
        let result = y.as_slice::<f64>();
        
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod broadcast_tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_broadcast_shapes() {
        let a = Tensor::from_f32(vec![1.0; 6], vec![2, 3]);
        let b = Tensor::from_f32(vec![1.0; 3], vec![3]);
        
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape, vec![2, 3]);
    }

    #[test]
    fn test_broadcast_complex() {
        // [2, 1, 3] + [3] -> [2, 1, 3]
        let a = Tensor::from_f32(vec![1.0; 6], vec![2, 1, 3]);
        let b = Tensor::from_f32(vec![10.0, 20.0, 30.0], vec![3]);
        
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape, vec![2, 1, 3]);
        
        let result = c.as_f32_slice();
        assert_eq!(result, &[11.0, 21.0, 31.0, 11.0, 21.0, 31.0]);
    }
}

#[cfg(test)]
mod error_tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_dtype_mismatch() {
        let a = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_slice(&[1, 2], vec![2]);
        
        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_shape_mismatch_no_broadcast() {
        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_f32(vec![1.0, 2.0], vec![2]);
        
        assert!(a.add(&b).is_err());
    }
}
