#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

//! EasyScribe 3.2.3 — Rune single-window app.
//!
//! Startup resolves the single-exe payload (or a development runtime), imports
//! v3 settings, runs crash recovery and dependency checks, then hands the
//! window to [`views::view`]. The app is fully offline.

mod app;
mod dialogs;
mod legacy;
mod platform;
mod theme;
mod views;
mod widgets;
mod workers;

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use easyscribe_core::{Paths, Status, payload};
use rune::{AppSpec, Cx, El, Tx, Waker};

use app::{AppState, FileEntry, FileState, Init, UiState};

static PATHS: OnceLock<Paths> = OnceLock::new();
/// Set before `rune::run`; `AppSpec::init` is a plain function pointer.
static FIRST_RUN: AtomicBool = AtomicBool::new(false);
/// The real window height in points. Layout decisions (file rows, short-window
/// naming, page scroll) must use the same value the window was created with.
static WINDOW_HEIGHT_BITS: AtomicU32 = AtomicU32::new(0);

fn main() {
    easyscribe_core::offline::install();
    if let Some(code) = worker_mode() {
        std::process::exit(code);
    }
    let headless = std::env::args().any(|a| {
        matches!(
            a.as_str(),
            "--snap" | "--scene" | "--list-scenes" | "--script" | "--desktop" | "--selftest"
        )
    }) || std::env::var_os("EASYSCRIBE_HEADLESS").is_some();

    platform::set_app_identity();

    // A first launch shows its unpacking progress in the window. Cached
    // launches, development builds and headless snapshots take the direct path
    // and behave exactly as before.
    let first_run = !headless && payload::needs_prepare().unwrap_or(false);
    let paths = if first_run {
        match Paths::at(payload::data_root()) {
            Ok(paths) => paths,
            Err(error) => {
                dialogs::error(
                    "EasyScribe cannot start",
                    &format!("EasyScribe could not prepare its program files:\n\n{error}"),
                );
                std::process::exit(1);
            }
        }
    } else {
        match Paths::discover() {
            Ok(paths) => paths,
            Err(error) => {
                let text = format!("EasyScribe could not prepare its program files:\n\n{error}");
                if headless {
                    eprintln!("{text}");
                } else {
                    dialogs::error("EasyScribe cannot start", &text);
                }
                std::process::exit(1);
            }
        }
    };

    if !headless && !first_run {
        app::startup(&paths);
        app::recover_before_ui(&paths);
        if let Some(message) = app::startup_error(&paths) {
            dialogs::error("EasyScribe cannot start", &message);
            std::process::exit(1);
        }
        if let Some(problem) = platform::processor_problem() {
            dialogs::error("EasyScribe cannot start", &problem);
            std::process::exit(1);
        }
    }

    if std::env::args().any(|arg| arg == "--selftest") {
        std::process::exit(run_selftest(&paths));
    }

    let _ = PATHS.set(paths);
    FIRST_RUN.store(first_run, Ordering::Relaxed);

    let height = initial_height();
    WINDOW_HEIGHT_BITS.store(height.to_bits(), Ordering::Relaxed);
    let spec = AppSpec {
        title: "EasyScribe 3.2.3",
        size: (900.0, height),
        min_size: Some((760.0, 600.0)),
        init,
        view,
        scenes: SCENES,
        fonts: &[],
        families: ("Segoe UI", "Consolas"),
        setup: Some(setup),
        poll: Some(poll),
        on_key: Some(on_key),
    };
    rune::App::new(spec).on_close(on_close).run();
}

/// Hidden child-process entry points. The main window is never created.
fn worker_mode() -> Option<i32> {
    let mut args = std::env::args().skip(1);
    let mut worker = false;
    let mut mode = None;
    let mut wav = None;
    let mut threads = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--worker" => {
                worker = true;
                mode = args.next();
            }
            "--wav" => wav = args.next(),
            "--threads" => {
                threads = args.next().and_then(|value| value.parse::<usize>().ok());
            }
            _ => {}
        }
    }
    if !worker {
        return None;
    }
    let Some(mode) = mode else {
        eprintln!("--worker needs a mode");
        return Some(2);
    };
    if mode != "diarize" {
        eprintln!("Unknown worker mode: {mode}");
        return Some(2);
    }
    let Some(wav) = wav else {
        eprintln!("--worker diarize needs --wav <path>");
        return Some(2);
    };
    if let Some(threads) = threads
        && threads > 0
    {
        // SAFETY: single-threaded startup, before any worker or UI thread exists.
        unsafe { std::env::set_var("EASYSCRIBE_NUM_THREADS", threads.to_string()) };
    }
    let paths = match Paths::discover() {
        Ok(paths) => paths,
        Err(error) => {
            eprintln!("The diarization worker could not prepare its files: {error}");
            return Some(1);
        }
    };
    Some(easyscribe_core::diarizer::run_diarize_worker(
        &paths,
        std::path::Path::new(&wav),
    ))
}

/// Load every native model once and decode a short local WAV. The release
/// workflow runs this on the packed executable; `--list-scenes` alone never
/// touches sherpa-onnx or whisper.cpp.
fn run_selftest(paths: &Paths) -> i32 {
    use easyscribe_core::diarizer::DiarizationEngine;
    use easyscribe_core::{
        CancelToken, SilentReporter, TranscriptionEngine, sherpa_engine, whispercli, whispercpp,
    };
    use std::sync::Arc;

    let budget = std::env::var("EASYSCRIBE_SELFTEST_BUDGET_SEC")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value > 0.0)
        .unwrap_or(180.0);

    let mut failures: Vec<String> = Vec::new();
    println!(
        "info: build profile: debug_assertions={}, cargo={}, whisper.cpp CMake={}",
        cfg!(debug_assertions),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        if cfg!(debug_assertions) {
            "RelWithDebInfo"
        } else {
            "Release"
        }
    );
    let info = whispercpp::system_info();
    println!("info: whisper.cpp system info: {info}");
    for feature in ["AVX2", "FMA", "F16C"] {
        let needle = format!("{feature} = 1");
        if !info.contains(&needle) {
            failures.push(format!("whisper.cpp CPU feature missing: {needle}"));
        }
    }
    println!(
        "info: {} logical CPU(s), whisper.cpp uses {} thread(s)",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(0),
        easyscribe_core::config::num_threads()
    );
    println!(
        "info: bundled whisper-cli: {}",
        if paths.whispercpp_bin.is_file() {
            paths.whispercpp_bin.display().to_string()
        } else {
            "no (in-process engine)".into()
        }
    );

    let fixture = match std::env::var_os("EASYSCRIBE_SELFTEST_WAV") {
        Some(value) => {
            let path = PathBuf::from(value);
            if !path.is_file() {
                println!(
                    "FAILED: EASYSCRIBE_SELFTEST_WAV is not a file: {}",
                    path.display()
                );
                return 1;
            }
            path
        }
        None => {
            let path = paths.assets_dir.join("selftest_speech.wav");
            if !path.is_file() {
                println!(
                    "FAILED: the selftest speech fixture is missing: {}",
                    path.display()
                );
                return 1;
            }
            path
        }
    };
    let seconds = wav_seconds(&fixture).unwrap_or(0.0);
    println!(
        "info: selftest speech: {} ({seconds:.1}s)",
        fixture.display()
    );
    let Some(expected) = expected_transcript(paths) else {
        println!("FAILED: the selftest golden transcript is missing or unreadable");
        return 1;
    };
    let wav = fixture;

    let have_cli = paths.whispercpp_bin.is_file() && paths.whispercpp_model.is_file();
    if cfg!(not(debug_assertions)) && !have_cli {
        failures.push("the bundled whisper-cli is missing from the release".into());
    }

    let reporter: Arc<dyn easyscribe_core::Reporter> = Arc::new(SilentReporter);
    // Decode before any ONNX Runtime model is loaded. The primary path must be
    // the bundled whisper-cli when it exists, so a spawn or DLL failure is a
    // selftest failure rather than a silent in-process decode. Debug builds
    // without a CLI still exercise the linked engine.
    let started = std::time::Instant::now();
    let cancel = CancelToken::new();
    let result = if have_cli {
        whispercli::transcribe_file(paths, &wav, &cancel, &reporter)
    } else {
        whispercpp::transcribe_in_process(paths, &wav, &cancel, &reporter)
    }
    .and_then(|segments| check_transcript(&segments, Some(&expected)));
    report_step(
        &mut failures,
        "whisper.cpp cold decode",
        started,
        Some(budget),
        result,
    );

    // Keep one in-process decode so CI tests the linked whisper.cpp build too,
    // not only the child process. It must reach the same transcript.
    let started = std::time::Instant::now();
    let cancel = CancelToken::new();
    let result = whispercpp::transcribe_in_process(paths, &wav, &cancel, &reporter)
        .and_then(|segments| check_transcript(&segments, Some(&expected)));
    report_step(
        &mut failures,
        "whisper.cpp in-process decode",
        started,
        // Diagnostic only: the fallback is not the shipping path.
        None,
        result,
    );

    let started = std::time::Instant::now();
    let engine = TranscriptionEngine::new(paths.clone());
    let result = engine.ensure_loaded(&SilentReporter);
    report_step(
        &mut failures,
        "sherpa-onnx whisper model",
        started,
        None,
        result,
    );
    // Load and drop the recognizer: the second whisper decode then tests
    // ONNX Runtime's global/thread-local side effects without also holding
    // the ~1 GB sherpa model in memory.
    drop(engine);

    let started = std::time::Instant::now();
    let result = sherpa_engine::build_vad(paths, 5.0).map(|_| ());
    report_step(&mut failures, "silero VAD model", started, None, result);

    let started = std::time::Instant::now();
    let cancel = CancelToken::new();
    // This step must stay in-process: it is the diagnostic that catches ONNX
    // Runtime's thread-local state poisoning a later whisper.cpp encode.
    let result = whispercpp::transcribe_in_process(paths, &wav, &cancel, &reporter)
        .and_then(|segments| check_transcript(&segments, Some(&expected)));
    report_step(
        &mut failures,
        "whisper.cpp decode after sherpa",
        started,
        None,
        result,
    );

    let started = std::time::Instant::now();
    let engine = DiarizationEngine::new(paths.clone());
    let result = require_turns(engine.diarize(&wav, &CancelToken::new(), Arc::new(SilentReporter)));
    report_step(
        &mut failures,
        "speaker diarization (in-process)",
        started,
        None,
        result,
    );

    let started = std::time::Instant::now();
    let result = require_turns(easyscribe_core::diarizer::diarize_in_child(
        paths,
        &wav,
        &CancelToken::new(),
        Arc::new(SilentReporter),
    ));
    report_step(
        &mut failures,
        "speaker diarization (child process)",
        started,
        None,
        result,
    );

    // Silence must produce an empty transcript and the clear "No speech
    // found" log, never a hallucinated "Thank you." and a long decode.
    let silence = paths.temp_dir.join("selftest_silence.wav");
    let started = std::time::Instant::now();
    let result = write_silence(&silence, 5).and_then(|()| {
        let logs = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let capture: Arc<dyn easyscribe_core::Reporter> =
            std::sync::Arc::new(CaptureReporter { logs: logs.clone() });
        let segments = whispercpp::transcribe_file(paths, &silence, &CancelToken::new(), &capture)?;
        let found = logs
            .lock()
            .map(|lines| lines.iter().any(|line| line.contains("No speech found")))
            .unwrap_or(false);
        if !segments.is_empty() {
            return Err(easyscribe_core::Error::Transcription(format!(
                "silence produced {} segment(s)",
                segments.len()
            )));
        }
        if !found {
            return Err(easyscribe_core::Error::Transcription(
                "silence did not log \"No speech found\"".into(),
            ));
        }
        Ok(())
    });
    let _ = std::fs::remove_file(&silence);
    report_step(
        &mut failures,
        "silence produces no speech",
        started,
        None,
        result,
    );

    let report = if failures.is_empty() {
        "EasyScribe selftest passed".to_owned()
    } else {
        format!("EasyScribe selftest failed:\n{}", failures.join("\n"))
    };
    println!("{report}");
    let _ = std::fs::write(paths.logs_dir.join("selftest.txt"), &report);
    if failures.is_empty() { 0 } else { 1 }
}

struct CaptureReporter {
    logs: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl easyscribe_core::Reporter for CaptureReporter {
    fn log(&self, line: &str) {
        if let Ok(mut logs) = self.logs.lock() {
            logs.push(line.to_owned());
        }
    }
}

fn expected_transcript(paths: &Paths) -> Option<String> {
    std::env::var_os("EASYSCRIBE_SELFTEST_TEXT")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            let path = paths.assets_dir.join("selftest_speech.txt");
            path.is_file().then_some(path)
        })
        .and_then(|path| std::fs::read_to_string(path).ok())
}

fn check_transcript(
    segments: &[easyscribe_core::Segment],
    expected: Option<&str>,
) -> easyscribe_core::Result<()> {
    if segments.is_empty() {
        return Err(easyscribe_core::Error::Transcription(
            "the speech sample produced no segments".into(),
        ));
    }
    let Some(expected) = expected else {
        return Ok(());
    };
    let actual = easyscribe_core::transcript::build_plain_transcript(segments, false);
    if actual.trim_end() != expected.trim_end() {
        return Err(easyscribe_core::Error::Transcription(format!(
            "the transcript does not match the golden text\n--- expected ---\n{}\n--- actual ---\n{}",
            expected.trim_end(),
            actual.trim_end()
        )));
    }
    Ok(())
}

fn require_turns(
    result: easyscribe_core::Result<Vec<easyscribe_core::Turn>>,
) -> easyscribe_core::Result<()> {
    result.and_then(|turns| {
        if turns.is_empty() {
            Err(easyscribe_core::Error::Diarization(
                "no speaker segments were found".into(),
            ))
        } else {
            Ok(())
        }
    })
}

fn report_step(
    failures: &mut Vec<String>,
    name: &str,
    started: std::time::Instant,
    budget: Option<f64>,
    result: easyscribe_core::Result<()>,
) {
    let seconds = started.elapsed().as_secs_f64();
    match result {
        Ok(()) => {
            println!("ok: {name} ({seconds:.1}s)");
            if let Some(budget) = budget
                && seconds > budget
            {
                let message = format!("{name} took {seconds:.1}s, over the {budget:.0}s budget");
                println!("FAILED: {message}");
                failures.push(message);
            }
        }
        Err(error) => {
            println!("FAILED: {name} ({seconds:.1}s): {error}");
            failures.push(format!("{name}: {error}"));
        }
    }
}

fn write_silence(path: &std::path::Path, seconds: u32) -> easyscribe_core::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| easyscribe_core::Error::Transcription(e.to_string()))?;
    for _ in 0..seconds * 16_000 {
        writer
            .write_sample(0_i16)
            .map_err(|e| easyscribe_core::Error::Transcription(e.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|e| easyscribe_core::Error::Transcription(e.to_string()))
}

fn wav_seconds(path: &std::path::Path) -> Option<f64> {
    let reader = hound::WavReader::open(path).ok()?;
    let spec = reader.spec();
    let frames = reader.duration() as f64;
    Some(frames / spec.sample_rate as f64)
}

fn init() -> AppState {
    let preparing = FIRST_RUN.load(Ordering::Relaxed);
    let paths = PATHS
        .get()
        .cloned()
        .unwrap_or_else(|| Paths::discover().expect("EasyScribe paths"));
    let bits = WINDOW_HEIGHT_BITS.load(Ordering::Relaxed);
    let window_height = if bits == 0 {
        scene_height()
    } else {
        f32::from_bits(bits)
    };
    AppState::new(Init {
        paths,
        window_height,
        preparing,
    })
}

fn scene_height() -> f32 {
    std::env::var("EASYSCRIBE_WINDOW_HEIGHT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(820.0)
}

fn view(s: &AppState, _cx: &mut Cx<AppState>) -> El {
    views::view(s)
}

fn setup(s: &mut AppState, waker: Waker) {
    s.set_waker(waker);
    if FIRST_RUN.load(Ordering::Relaxed) {
        s.begin_prepare();
    }
}

fn poll(s: &mut AppState, _now: f64, _tx: &mut Tx) -> bool {
    s.poll()
}

fn on_key(s: &mut AppState, event: &rune::KeyEvent, _tx: &mut Tx) -> bool {
    s.handle_key(event)
}

/// Python's `atexit` cleanup: temporary WAVs and decoder JSON never outlive a
/// session. Interrupted recordings live in `recordings/` and are handled by
/// the recovery prompt instead.
fn on_close(s: &mut AppState, _tx: &mut Tx) {
    if let Some(cancel) = &s.record_stop {
        cancel.cancel();
    }
    if let Some(cancel) = &s.cancel {
        cancel.cancel();
    }
    easyscribe_core::logger::clean_temp_files(&s.paths.temp_dir);
}

/// The window fits the work area like the Python app: 820 pt at 1080p, 678 pt
/// on a scaled 1366×768 laptop, never below 600.
fn initial_height() -> f32 {
    // Headless scene renders and visual QA use an explicit height even on
    // Windows, where the real screen would otherwise win.
    if let Some(height) = std::env::var("EASYSCRIBE_WINDOW_HEIGHT")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
    {
        return height.clamp(600.0, 820.0);
    }
    #[cfg(windows)]
    {
        #[link(name = "user32")]
        unsafe extern "system" {
            fn GetSystemMetrics(index: i32) -> i32;
            fn GetDpiForSystem() -> u32;
        }
        const SM_CYSCREEN: i32 = 1;
        unsafe {
            let screen = GetSystemMetrics(SM_CYSCREEN) as f32;
            let dpi = GetDpiForSystem() as f32;
            let logical = if dpi > 0.0 {
                screen * 96.0 / dpi
            } else {
                screen
            };
            return (logical - 90.0).clamp(600.0, 820.0);
        }
    }
    #[cfg(not(windows))]
    {
        scene_height()
    }
}

// ---------------------------------------------------------------------------
// Visual QA scenes (`cargo run -p easyscribe -- --snap out.png --scene files`)
// ---------------------------------------------------------------------------

type Scene = (&'static str, fn(&mut AppState));

const SCENES: &[Scene] = &[
    ("files-empty", scene_files_empty),
    ("files", scene_files),
    ("running", scene_running),
    ("done", scene_done),
    ("error", scene_error),
    ("record", scene_record),
    ("recording", scene_recording),
    ("live", scene_live),
    ("naming", scene_naming),
    ("find", scene_find),
    ("privacy", scene_privacy),
    ("details", scene_details),
    ("preparing", scene_preparing),
];

fn sample_files(s: &mut AppState) {
    s.files = vec![
        file_entry(
            "Team meeting 2026-09-14.m4a",
            29_569_024,
            "/home/user/demo",
            FileState::Done,
        ),
        file_entry(
            "Interview with Sam.mp3",
            12_845_056,
            "/home/user/demo",
            FileState::Done,
        ),
        file_entry(
            "Conference keynote.mp4",
            145_000_000,
            "/home/user/demo",
            FileState::Waiting,
        ),
    ];
}

fn file_entry(name: &str, size: u64, parent: &str, state: FileState) -> FileEntry {
    let mut entry = FileEntry::new(PathBuf::from(parent).join(name));
    entry.state = state;
    entry.display_name = Some(name.to_owned());
    entry.display_size = Some(size);
    entry
}

fn scene_files_empty(s: &mut AppState) {
    s.mode = app::Mode::Files;
    s.ui = UiState::Idle;
    s.activity = None;
    s.files.clear();
}

fn scene_files(s: &mut AppState) {
    scene_files_empty(s);
    sample_files(s);
}

fn scene_running(s: &mut AppState) {
    scene_files_empty(s);
    sample_files(s);
    s.files[0].state = FileState::Done;
    s.files[1].state = FileState::Working;
    s.files[2].state = FileState::Waiting;
    s.ui = UiState::Running;
    let mut activity = app::Activity::new(&s.theme);
    activity.apply_status(Status::Transcribing, &s.theme, true);
    activity.detail = "File 2 of 3: Interview with Sam.mp3".into();
    activity.bar = app::BarMode::Determinate;
    activity.progress = 0.42;
    activity.stop_visible = true;
    activity.stop_enabled = true;
    s.activity = Some(activity);
}

fn scene_done(s: &mut AppState) {
    scene_files_empty(s);
    sample_files(s);
    s.files[2].state = FileState::Done;
    let mut activity = app::Activity::new(&s.theme);
    activity.title = "Transcript ready".into();
    activity.tone = app::Tone::Green;
    activity.bar = app::BarMode::Determinate;
    activity.progress = 1.0;
    activity.bar_color = s.theme.green;
    activity.detail = "Saved in /home/user/demo".into();
    activity.open_transcript = true;
    activity.open_folder = true;
    s.activity = Some(activity);
    s.last_output_folder = Some(PathBuf::from("/home/user/demo"));
    s.last_transcript = Some(PathBuf::from("/home/user/demo/Team meeting 2026-09-14.txt"));
    s.rename_visible = true;
    s.rename_edit = rune::TextEdit::new("Team meeting 2026-09-14");
    s.rename_note = "Renames the transcript. Your original file is not changed.".into();
}

fn scene_error(s: &mut AppState) {
    scene_files_empty(s);
    s.files = vec![file_entry(
        "Damaged recording.wav",
        2_400_000,
        "/home/user/demo",
        FileState::Failed,
    )];
    let mut activity = app::Activity::new(&s.theme);
    activity.title = "Transcription failed".into();
    activity.tone = app::Tone::Crimson;
    activity.bar_color = s.theme.crimson;
    activity.message = Some((
        "Could not transcribe Damaged recording.wav. The file may be damaged or have no sound. Select \"Show details\" to see why.".into(),
        app::MessageKind::Error,
    ));
    activity.try_again = true;
    s.activity = Some(activity);
}

fn scene_record(s: &mut AppState) {
    s.mode = app::Mode::Record;
    s.ui = UiState::Idle;
    s.activity = None;
    s.record_mode = app::RecordMode::Best;
    s.timer_seconds = 0.0;
}

fn scene_recording(s: &mut AppState) {
    scene_record(s);
    s.ui = UiState::Recording;
    s.timer_seconds = 83.0;
    s.level = 0.62;
    s.live_text = app::LISTENING.into();
    let mut activity = app::Activity::new(&s.theme);
    activity.apply_status(Status::Recording, &s.theme, false);
    activity.stop_visible = false;
    activity.bar_color = s.theme.coral;
    s.activity = Some(activity);
}

fn scene_live(s: &mut AppState) {
    scene_record(s);
    s.record_mode = app::RecordMode::Live;
    s.ui = UiState::Recording;
    s.timer_seconds = 41.0;
    s.level = 0.35;
    s.live_has_text = true;
    s.live_text = "Welcome everyone. Thanks for joining the quarterly review.\nWe will start with the revenue numbers for the north region.\nThat is excellent news. Can you break down the figures by region?".into();
    let mut activity = app::Activity::new(&s.theme);
    activity.apply_status(Status::Recording, &s.theme, false);
    activity.bar_color = s.theme.coral;
    s.activity = Some(activity);
}

fn scene_find(s: &mut AppState) {
    scene_record(s);
    s.last_recording = Some(PathBuf::from(
        "/home/user/demo/recording_20261021_143207.wav",
    ));
    s.last_transcript = Some(PathBuf::from(
        "/home/user/demo/recording_20261021_143207.txt",
    ));
    s.last_output_folder = Some(PathBuf::from("/home/user/demo"));
    s.last_segments = vec![
        easyscribe_core::Segment::new(0.0, 4.2, "Hello and welcome."),
        easyscribe_core::Segment::new(4.6, 9.1, "Thanks for joining."),
    ];
    let mut activity = app::Activity::new(&s.theme);
    activity.title = "Transcript ready".into();
    activity.tone = app::Tone::Green;
    activity.bar = app::BarMode::Determinate;
    activity.progress = 1.0;
    activity.bar_color = s.theme.green;
    activity.detail = "Saved in /home/user/demo".into();
    activity.open_transcript = true;
    activity.open_folder = true;
    s.activity = Some(activity);
    s.find_card = Some(app::FindCard {
        estimate: "About 6 minutes".into(),
        basis: "Based on 27 min of audio, with 19 min of speech. You can keep using your computer and stop at any time."
            .into(),
    });
    s.rename_visible = true;
    s.rename_edit = rune::TextEdit::new("recording_20261021_143207");
    s.rename_note = "Renames the recording and its transcripts together.".into();
}

fn scene_naming(s: &mut AppState) {
    scene_files_empty(s);
    sample_files(s);
    s.ui = UiState::Running;
    let mut activity = app::Activity::new(&s.theme);
    activity.apply_status(Status::NamingSpeakers, &s.theme, true);
    activity.detail = "Team meeting 2026-09-14.m4a".into();
    s.activity = Some(activity);
    let rows = vec![
        app::NamingRow {
            raw: "SPEAKER_00".into(),
            default_name: "Speaker 1".into(),
            share: Some(0.42),
            clip: Some(PathBuf::from("/tmp/clip0.wav")),
            edit: rune::TextEdit::new(""),
        },
        app::NamingRow {
            raw: "SPEAKER_01".into(),
            default_name: "Speaker 2".into(),
            share: Some(0.31),
            clip: Some(PathBuf::from("/tmp/clip1.wav")),
            edit: rune::TextEdit::new(""),
        },
        app::NamingRow {
            raw: "SPEAKER_02".into(),
            default_name: "Speaker 3".into(),
            share: Some(0.27),
            clip: None,
            edit: rune::TextEdit::new(""),
        },
    ];
    s.naming = Some(app::NamingState { rows, reply: None });
}

fn scene_privacy(s: &mut AppState) {
    scene_files(s);
    s.privacy_open = true;
}

fn scene_preparing(s: &mut AppState) {
    s.mode = app::Mode::Files;
    s.ui = UiState::Idle;
    s.activity = None;
    s.preparing = Some(app::PreparingState {
        stage: "Unpacking the app files".into(),
        progress: 0.43,
        error: None,
    });
}

fn scene_details(s: &mut AppState) {
    scene_done(s);
    s.details_open = true;
    s.log = vec![
        "".into(),
        "──────────────────────────────────────────────────".into(),
        "[File 1/3] Team meeting 2026-09-14.m4a".into(),
        "[ffmpeg] Extracting audio to 16 kHz mono WAV".into(),
        "[Transcribe] Loading whisper.cpp model".into(),
        "[whisper.cpp] Engine: Vulkan (device 0: NVIDIA GeForce RTX 3060)".into(),
        "[Transcribe] Transcribed 412 segments".into(),
        "[Output] Writing transcript to: /home/user/demo/Team meeting 2026-09-14.txt".into(),
        "[Done] Saved: /home/user/demo/Team meeting 2026-09-14.txt  (28 KB)".into(),
        "".into(),
        "──────────────────────────────────────────────────".into(),
        "[File 2/3] Interview with Sam.mp3".into(),
        "[ffmpeg] Extracting audio to 16 kHz mono WAV".into(),
    ];
}
