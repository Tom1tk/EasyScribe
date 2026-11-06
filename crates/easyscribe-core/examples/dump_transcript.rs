use easyscribe_core::{CancelToken, Paths, Reporter, SilentReporter, whispercpp};
use std::sync::Arc;

struct Print;
impl Reporter for Print {
    fn log(&self, line: &str) {
        println!("LOG {line}");
    }
    fn progress(&self, f: f64) {
        if f >= 1.0 {
            println!("PROGRESS {f}");
        }
    }
}

fn main() {
    let base = Paths::at(easyscribe_core::config::dev_base_dir()).unwrap();
    let wav = std::env::args().nth(1).unwrap();
    let mode = std::env::args().nth(2).unwrap_or_else(|| "inproc".into());
    let reporter: Arc<dyn Reporter> = Arc::new(SilentReporter);
    let _ = Print;
    let segments = if mode == "cli" {
        whispercpp::transcribe_file(
            &base,
            std::path::Path::new(&wav),
            &CancelToken::new(),
            &reporter,
        )
    } else {
        whispercpp::transcribe_in_process(
            &base,
            std::path::Path::new(&wav),
            &CancelToken::new(),
            &reporter,
        )
    }
    .unwrap();
    for s in segments {
        println!("{:.2}\t{:.2}\t{}", s.start, s.end, s.text);
    }
}
