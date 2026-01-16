mod tensor;
use tensor::Tensor;

fn main() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);

    println!("a = {}", a);
    println!("b = {}", b);

    let c = a.matmul(&b);
    println!("matmul = {}", c);

    let d = a.add(&b);
    println!("add = {}", d);

    let e = a.mul(&b);
    println!("mul = {}", e);
}
