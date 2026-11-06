//! Small platform services: opening files/folders, WAV playback and Windows
//! process identity/priority. Each has a portable fallback.

use std::path::Path;
#[cfg(not(windows))]
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

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
        use std::os::windows::ffi::OsStrExt;
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let operation: Vec<u16> = "open\0".encode_utf16().collect();
        #[link(name = "shell32")]
        unsafe extern "system" {
            fn ShellExecuteW(
                hwnd: *mut std::ffi::c_void,
                operation: *const u16,
                file: *const u16,
                parameters: *const u16,
                directory: *const u16,
                show: i32,
            ) -> *mut std::ffi::c_void;
        }
        // ShellExecuteW returns a value greater than 32 on success.
        let result = unsafe {
            ShellExecuteW(
                std::ptr::null_mut(),
                operation.as_ptr(),
                wide.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                1,
            )
        } as isize;
        if result > 32 {
            Ok(())
        } else {
            Err(format!("Could not open {}", path.display()))
        }
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

/// The sample currently playing in the naming panel. Holding `Arc<Player>`
/// lets the panel stop it when it closes; the playback thread owns the device
/// sink and exits after `sleep_until_end`.
static SAMPLE_PLAYER: OnceLock<Mutex<Option<Arc<rodio::Player>>>> = OnceLock::new();

pub fn play_wav(path: &Path) {
    stop_playback();
    let path = path.to_path_buf();
    let _ = std::thread::Builder::new()
        .name("SamplePlayback".into())
        .spawn(move || {
            let Ok(stream) = rodio::DeviceSinkBuilder::open_default_sink() else {
                return;
            };
            let player = Arc::new(rodio::Player::connect_new(stream.mixer()));
            let slot = SAMPLE_PLAYER.get_or_init(|| Mutex::new(None));
            if let Ok(mut current) = slot.lock() {
                *current = Some(player.clone());
            }
            let Ok(file) = std::fs::File::open(&path) else {
                return;
            };
            let Ok(decoder) = rodio::Decoder::try_from(file) else {
                return;
            };
            player.append(decoder);
            player.sleep_until_end();
            if let Ok(mut current) = slot.lock()
                && current
                    .as_ref()
                    .is_some_and(|playing| Arc::ptr_eq(playing, &player))
            {
                *current = None;
            }
        });
}

/// Stop the speaker sample, if one is playing. Called when the naming panel
/// closes so a clip never outlives the names it belongs to.
pub fn stop_playback() {
    if let Some(slot) = SAMPLE_PLAYER.get()
        && let Ok(mut current) = slot.lock()
        && let Some(player) = current.take()
    {
        player.stop();
    }
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
