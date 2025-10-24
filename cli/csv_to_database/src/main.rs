use std::ffi::OsStr;
use std::path::PathBuf;

mod db;
mod csv;

fn main() {
    let mut csv_files: Vec<PathBuf> =  Vec::new();

    let files: Vec<PathBuf> = csv::retrieve_csv_files("./datasets/");
    for file in files {
        if file.extension() == Some(OsStr::new("csv")) {
            csv_files.push(file)
        }
    }

    for file in csv_files {
        csv::read_csv_file().expect("Cannot read CSV file.");
    }
}