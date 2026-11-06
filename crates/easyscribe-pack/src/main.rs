//! `easyscribe-pack` — append the runtime payload to a built executable.
//!
//! The result is the single Windows `EasyScribe.exe`: a normal executable plus
//! a hashed ZIP overlay and a 128-byte footer. The app extracts the overlay
//! once into its per-user runtime cache.
//!
//! ```text
//! cargo run -p easyscribe-pack --release -- \
//!   --binary target/x86_64-pc-windows-msvc/release/easyscribe.exe \
//!   --payload payload \
//!   --out dist/EasyScribe.exe
//! ```

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use easyscribe_core::payload;

fn main() {
    if let Err(error) = run() {
        eprintln!("easyscribe-pack: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut binary = PathBuf::from("target/release/easyscribe.exe");
    let mut payload_dir = PathBuf::from("payload");
    let mut out = PathBuf::from("dist/EasyScribe.exe");
    let mut version = easyscribe_core::APP_VERSION.to_owned();
    let mut build_id = String::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--binary" => binary = PathBuf::from(value()?),
            "--payload" => payload_dir = PathBuf::from(value()?),
            "--out" => out = PathBuf::from(value()?),
            "--version" => version = value()?,
            "--build-id" => build_id = value()?,
            "-h" | "--help" => {
                println!(
                    "Usage: easyscribe-pack [--binary EXE] [--payload DIR] [--out EXE] \
                     [--version VERSION] [--build-id ID]"
                );
                return Ok(());
            }
            other => return Err(format!("unknown option {other}")),
        }
    }

    if !binary.is_file() {
        return Err(format!("binary not found: {}", binary.display()));
    }
    if !payload_dir.is_dir() {
        return Err(format!(
            "payload folder not found: {}",
            payload_dir.display()
        ));
    }
    if build_id.is_empty() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        build_id = format!("{version}-{stamp}");
    }

    println!("Packing {}", out.display());
    println!("  binary:  {}", binary.display());
    println!("  payload: {}", payload_dir.display());
    let footer = payload::pack_streaming(&binary, &payload_dir, &out, &version, &build_id)
        .map_err(|e| e.to_string())?;
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or_default();
    println!(
        "Wrote {} ({:.1} MB), build {}, payload {:.1} MB",
        out.display(),
        size as f64 / (1024.0 * 1024.0),
        footer.build_id,
        footer.payload_size as f64 / (1024.0 * 1024.0)
    );
    println!("  cache key: {}", footer.cache_key());
    Ok(())
}
