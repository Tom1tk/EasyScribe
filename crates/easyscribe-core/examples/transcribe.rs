//! Development smoke test: transcribe a WAV with the real ONNX models.
//! `cargo run -p easyscribe-core --example transcribe -- <wav> <out.txt>`

use std::path::PathBuf;
use std::sync::Arc;

use easyscribe_core::diarizer::SpeakerNaming;
use easyscribe_core::transcriber::{self, TranscribeRequest};
use easyscribe_core::{CancelToken, Paths, Reporter, Status, TranscriptionEngine};

struct Console;

impl Reporter for Console {
    fn status(&self, status: Status) {
        println!("STATUS {}", status.name());
    }
    fn progress(&self, fraction: f64) {
        println!("PROGRESS {:.3}", fraction);
    }
    fn log(&self, line: &str) {
        println!("{line}");
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let input = PathBuf::from(args.next().expect("input wav"));
    let output = PathBuf::from(args.next().expect("output txt"));
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let paths = Paths::at(base).expect("paths");
    let engine = TranscriptionEngine::new(paths.clone());
    let cancel = CancelToken::new();
    let reporter: Arc<dyn Reporter> = Arc::new(Console);
    let request = TranscribeRequest {
        audio_path: &input,
        output_path: &output,
        add_timestamps: true,
        diarize: false,
    };
    let naming: Option<&dyn SpeakerNaming> = None;
    match transcriber::transcribe(&paths, &engine, &request, &cancel, &reporter, naming) {
        Ok(segments) => {
            println!("OK {} segments", segments.len());
            println!("{}", std::fs::read_to_string(&output).unwrap_or_default());
        }
        Err(e) => {
            eprintln!("ERR {e}");
            std::process::exit(1);
        }
    }
}
