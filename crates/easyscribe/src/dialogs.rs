//! Native file/folder pickers and message dialogs.
//!
//! File/folder selection uses Rusty File Dialog. On Windows and macOS it is
//! the same native dialog family as the Python app; on Linux it uses the
//! desktop portal. Headless tests never call these functions.

use std::path::{Path, PathBuf};

use easyscribe_core::config::SUPPORTED_EXTENSIONS;

pub fn pick_files() -> Vec<PathBuf> {
    let extensions: Vec<&str> = SUPPORTED_EXTENSIONS.to_vec();
    rfd::FileDialog::new()
        .set_title("Choose recordings to transcribe")
        .add_filter("Audio and video files", &extensions)
        .add_filter("All files", &["*"])
        .pick_files()
        .unwrap_or_default()
}

pub fn pick_folder(title: &str, initial: Option<&Path>) -> Option<PathBuf> {
    let mut dialog = rfd::FileDialog::new().set_title(title);
    if let Some(initial) = initial {
        dialog = dialog.set_directory(initial);
    }
    dialog.pick_folder()
}

pub fn info(title: &str, text: &str) {
    let _ = rfd::MessageDialog::new()
        .set_title(title)
        .set_description(text)
        .set_level(rfd::MessageLevel::Info)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

pub fn error(title: &str, text: &str) {
    let _ = rfd::MessageDialog::new()
        .set_title(title)
        .set_description(text)
        .set_level(rfd::MessageLevel::Error)
        .set_buttons(rfd::MessageButtons::Ok)
        .show();
}

pub fn confirm(title: &str, text: &str) -> bool {
    matches!(
        rfd::MessageDialog::new()
            .set_title(title)
            .set_description(text)
            .set_level(rfd::MessageLevel::Warning)
            .set_buttons(rfd::MessageButtons::YesNo)
            .show(),
        rfd::MessageDialogResult::Yes
    )
}
