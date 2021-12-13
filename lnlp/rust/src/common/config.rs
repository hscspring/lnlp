use std::path::Path;
use std::io::BufReader;
use std::fs::File;
use serde::Deserialize;


pub trait Config<T>
    where for <'de> T: Deserialize<'de> {

    fn from_file(path: &Path) -> T {
        let file = File::open(path).expect("Could not open file.");
        let reader = BufReader::new(file);
        let config: T = serde_json::from_reader(reader).expect("Could not parse file.");
        config
    }
}