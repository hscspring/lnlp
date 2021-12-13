use lnlp::task::generation;
use lnlp::Config;

use std::path::Path;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

use serde_json::json;

#[derive(Debug, Deserialize)]
struct Gpt2Config {
    attn_pdrop: f64,
    embd_pdrop: f64,
}

impl Config<Gpt2Config> for Gpt2Config {}

fn main() {
    let path = Path::new("/Users/HaoShaochun/Documents/Tools/gpt2-散文模型/config.json");
    let config: Gpt2Config = Gpt2Config::from_file(&path);
    println!("{:#?}", config);
}

