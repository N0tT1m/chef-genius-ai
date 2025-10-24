extern crate walkdir;

use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use std::{error::Error, io, process};

pub(crate) fn retrieve_csv_files<P: AsRef<Path>>(root: P) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(|entry| entry.ok()) // Handle potential errors during iteration
        .filter(|entry| entry.file_type().is_file()) // Filter for files
        .map(|entry| entry.path().to_path_buf()) // Get PathBuf
        .collect()
}

pub(crate) fn read_csv_file() {

}

fn parse_csv_files() {

}

fn pass_data_to_db() {

}