mod tensor;
mod ops;

use tensor::Tensor;

fn main() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2,2]);

    println!("ReLU:\n{}", x.relu());
    println!("Softmax:\n{}", x.softmax());
}
