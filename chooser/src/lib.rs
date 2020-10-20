use rand::distributions::WeightedIndex;
use rand::prelude::*;

use cpython::{py_fn, py_module_initializer, PyResult, Python};

py_module_initializer!(chooser, |py, m| {
    m.add(py, "__doc__", "Rust implementation of TTST move-making")?;
    m.add(py, "choose", py_fn!(py, choose(weights: Vec<i32>)))?;
    Ok(())
});

fn choose(_: Python, weights: Vec<i32>) -> PyResult<i32> {
    let choices = [0, 1, 2, 3, 4, 5, 6, 7, 8];
    let dist = WeightedIndex::new(&weights).unwrap();
    let mut rng = thread_rng();

    Ok(choices[dist.sample(&mut rng)])
}
