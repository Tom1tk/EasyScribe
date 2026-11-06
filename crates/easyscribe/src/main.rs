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

use easyscribe_core::{Paths, Status};
use rune::{AppSpec, Cx, El, Tx, Waker};

use app::{AppState, FileEntry, FileState, Init, UiState};

static PATHS: OnceLock<Paths> = OnceLock::new();

fn main() {
    let headless = std::env::args().any(|a| {
        matches!(
            a.as_str(),
            "--snap" | "--scene" | "--list-scenes" | "--script" | "--desktop"
        )
    }) || std::env::var_os("EASYSCRIBE_HEADLESS").is_some();

    platform::set_app_identity();

    let paths = match Paths::discover() {
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
    };

    if !headless {
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

    let _ = PATHS.set(paths);

    let height = initial_height();
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
        on_key: None,
    };
    rune::run(spec);
}

fn init() -> AppState {
    let paths = PATHS
        .get()
        .cloned()
        .unwrap_or_else(|| Paths::discover().expect("EasyScribe paths"));
    AppState::new(Init {
        paths,
        window_height: scene_height(),
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
}

fn poll(s: &mut AppState, _now: f64, _tx: &mut Tx) -> bool {
    s.poll()
}

/// The window fits the work area like the Python app: 820 pt at 1080p, 678 pt
/// on a scaled 1366×768 laptop, never below 600.
fn initial_height() -> f32 {
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
            let logical = if dpi > 0.0 { screen * 96.0 / dpi } else { screen };
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

const SCENES: &[(&str, fn(&mut AppState))] = &[
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
];

fn sample_files(s: &mut AppState) {
    s.files = vec![
        file_entry("Team meeting 2026-09-14.m4a", 29_569_024, "/home/user/demo", FileState::Done),
        file_entry("Interview with Sam.mp3", 12_845_056, "/home/user/demo", FileState::Done),
        file_entry("Conference keynote.mp4", 145_000_000, "/home/user/demo", FileState::Waiting),
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
    s.last_recording = Some(PathBuf::from("/home/user/demo/recording_20261021_143207.wav"));
    s.last_transcript = Some(PathBuf::from("/home/user/demo/recording_20261021_143207.txt"));
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

