# ttst-blaze

This is the code from my blog post [“Speeding up Python programs with Rust: a practical example”][blog].

## Contents
* `ttst.py` the original ttst script, copied from [this commit][docs]
* `ttst_blaze.py` the new script as described in the blog post, differs only in the `cogitate()` function
* `chooser/` Rust code

## Usage
The original `ttst.py` is ready to use, refer to `python ttst.py --help` and its [documentation][docs] for more information.

To run the updated `ttst_blaze.py` you must first compile the Rust library:

```
$ cd chooser
$ cargo build --release
$ mv target/release/libchooser.so ../chooser.so
```


[blog]: https://enricomiccoli.com/2020/10/21/speeding-up-python-with-rust-example.html
[docs]: https://github.com/EnricoMiccoli/tic-tac-steel-toe/tree/396c6359edc004af8df2856a8e6d3e8227a05630
