//! Small platform services: opening files/folders, WAV playback and Windows
//! process identity/priority. Each has a portable fallback.

use std::path::Path;
use std::process::Command;

pub fn set_app_identity() {
    #[cfg(windows)]
    unsafe {
        use std::os::windows::ffi::OsStrExt;
        let id: Vec<u16> = std::ffi::OsStr::new("EasyScribe.App")
            .encode_wide()
            .chain(Some(0))
            .collect();
        #[link(name = "shell32")]
        unsafe extern "system" {
            fn SetCurrentProcessExplicitAppUserModelID(id: *const u16) -> i32;
        }
        let _ = SetCurrentProcessExplicitAppUserModelID(id.as_ptr());
    }
}

pub fn open_path(path: &Path) -> Result<(), String> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        Command::new("cmd")
            .args(["/C", "start", ""])
            .arg(path)
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(path)
            .spawn()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        Command::new("xdg-open")
            .arg(path)
            .spawn()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

pub fn play_wav(path: &Path) {
    let path = path.to_path_buf();
    let _ = std::thread::Builder::new()
        .name("SamplePlayback".into())
        .spawn(move || {
            let Ok(stream) = rodio::DeviceSinkBuilder::open_default_sink() else {
                return;
            };
            let player = rodio::Player::connect_new(stream.mixer());
            let Ok(file) = std::fs::File::open(&path) else { return };
            let Ok(decoder) = rodio::Decoder::try_from(file) else { return };
            player.append(decoder);
            player.sleep_until_end();
        });
}

/// Lower a worker thread below normal so long decodes leave the UI responsive.
/// Best effort: unsupported platforms simply keep the default priority.
pub fn lower_thread_priority() {
    #[cfg(windows)]
    unsafe {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut std::ffi::c_void;
            fn SetThreadPriority(thread: *mut std::ffi::c_void, priority: i32) -> i32;
        }
        const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    unsafe {
        unsafe extern "C" {
            fn nice(increment: i32) -> i32;
        }
        let _ = nice(5);
    }
}

/// The CPU floor for the whisper.cpp GGML build. Without these the first
/// decode dies with an illegal instruction on old x86 CPUs.
pub fn processor_problem() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    {
        let missing: Vec<&str> = [
            ("AVX2", std::is_x86_feature_detected!("avx2")),
            ("FMA", std::is_x86_feature_detected!("fma")),
            ("F16C", std::is_x86_feature_detected!("f16c")),
            ("BMI2", std::is_x86_feature_detected!("bmi2")),
        ]
        .into_iter()
        .filter_map(|(name, present)| (!present).then_some(name))
        .collect();
        if missing.is_empty() {
            None
        } else {
            Some(format!(
                "This computer's processor is missing features EasyScribe needs ({}). Please use a newer computer or a CPU-only build of EasyScribe.",
                missing.join(", ")
            ))
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}
