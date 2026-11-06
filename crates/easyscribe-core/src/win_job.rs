//! Windows job object shared by every child process EasyScribe starts.
//!
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` means Windows terminates the children
//! if the main app exits, crashes or is killed, so a long GPU decode or
//! diarization can never survive as an orphan.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
    SetInformationJobObject,
};

static JOB: OnceLock<usize> = OnceLock::new();

fn job() -> Option<HANDLE> {
    let stored = *JOB.get_or_init(|| unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            return 0;
        }
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast::<c_void>(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok == 0 {
            return 0;
        }
        job as usize
    });
    (stored != 0).then_some(stored as HANDLE)
}

/// Assign a child to the kill-on-close job. Failure is non-fatal: the child
/// still runs, it just would not be reaped if the app were killed.
pub fn assign(child: &std::process::Child) {
    use std::os::windows::io::AsRawHandle;

    let Some(job) = job() else { return };
    let process = child.as_raw_handle() as HANDLE;
    unsafe {
        AssignProcessToJobObject(job, process);
    }
}
