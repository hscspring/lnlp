#[macro_use]
extern crate failure;
extern crate tch;

pub fn main() -> failure::Fallible<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "usage: {} source.npz destination.ot", args[0]);

    let source_file = &args[1];
    let destination_file = &args[2];
    let tensors = tch::Tensor::read_npz(source_file)?;
    tch::Tensor::save_multi(&tensors, destination_file)?;

    Ok(())
}