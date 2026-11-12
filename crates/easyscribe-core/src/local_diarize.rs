//! Local copy of sherpa-rs 0.6.8's `Diarize` with configurable thread counts.
//!
//! Upstream hardcodes `num_threads: 1` for both the segmentation and embedding
//! models. That made "Find speakers" several times slower than the Python app,
//! which passed the machine's thread count to both. This wrapper keeps the
//! public behaviour (same clustering values, same progress callback) but sets
//! `num_threads` on both native model configs.
//!
//! It also returns an error for a model path that is not valid UTF-8 instead
//! of the upstream `to_str().unwrap()` panic.

use std::ffi::{CString, c_void};
use std::path::Path;
use std::ptr::null_mut;

use sherpa_rs::diarize::Segment;
use sherpa_rs::sherpa_rs_sys;

use crate::{Error, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub num_clusters: i32,
    pub threshold: f32,
    pub min_duration_on: f32,
    pub min_duration_off: f32,
    pub provider: String,
    pub num_threads: i32,
}

pub struct LocalDiarizer {
    sd: *const sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarization,
}

type ProgressCallback = Box<dyn Fn(i32, i32) -> i32 + Send + 'static>;

impl LocalDiarizer {
    pub fn new(segmentation_model: &Path, embedding_model: &Path, config: Config) -> Result<Self> {
        let segmentation_model = path_cstring(segmentation_model, "segmentation")?;
        let embedding_model = path_cstring(embedding_model, "embedding")?;
        let provider = CString::new(config.provider).map_err(|_| {
            Error::Diarization("The diarization provider name contains a NUL byte.".into())
        })?;

        let native = sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationConfig {
            embedding: sherpa_rs_sys::SherpaOnnxSpeakerEmbeddingExtractorConfig {
                model: embedding_model.as_ptr(),
                num_threads: effective_threads(config.num_threads),
                debug: 0,
                provider: provider.as_ptr(),
            },
            clustering: sherpa_rs_sys::SherpaOnnxFastClusteringConfig {
                num_clusters: config.num_clusters,
                threshold: config.threshold,
            },
            min_duration_off: config.min_duration_off,
            min_duration_on: config.min_duration_on,
            segmentation: sherpa_rs_sys::SherpaOnnxOfflineSpeakerSegmentationModelConfig {
                pyannote: sherpa_rs_sys::SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
                    model: segmentation_model.as_ptr(),
                },
                num_threads: effective_threads(config.num_threads),
                debug: 0,
                provider: provider.as_ptr(),
            },
        };

        // SAFETY: both C strings and the config stay alive for the call, and
        // sherpa-onnx copies the model paths it needs.
        let sd = unsafe { sherpa_rs_sys::SherpaOnnxCreateOfflineSpeakerDiarization(&native) };
        if sd.is_null() {
            return Err(Error::Diarization(
                "Failed to initialize offline speaker diarization".into(),
            ));
        }
        Ok(Self { sd })
    }

    pub fn compute(
        &mut self,
        mut samples: Vec<f32>,
        progress_callback: Option<ProgressCallback>,
    ) -> Result<Vec<Segment>> {
        let samples_ptr = samples.as_mut_ptr();
        // SAFETY: the sample buffer and callback box live until the native
        // call returns. The result and its segment array are destroyed here
        // after being copied into owned Rust values.
        unsafe {
            let mut callback_box =
                progress_callback.map(|cb| Box::new(cb) as Box<ProgressCallback>);
            let callback_ptr = callback_box
                .as_mut()
                .map(|boxed| boxed.as_mut() as *mut ProgressCallback as *mut c_void)
                .unwrap_or(null_mut());

            let result = sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                self.sd,
                samples_ptr,
                samples.len() as i32,
                if callback_box.is_some() {
                    Some(progress_callback_wrapper)
                } else {
                    None
                },
                callback_ptr,
            );
            if result.is_null() {
                return Err(Error::Diarization(
                    "Speaker diarization returned no result".into(),
                ));
            }

            let num_segments =
                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(result);
            let segments_ptr: *const sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationSegment =
                sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(result);

            let mut segments = Vec::new();
            if !segments_ptr.is_null() && num_segments > 0 {
                let slice = std::slice::from_raw_parts(segments_ptr, num_segments as usize);
                segments.reserve(slice.len());
                for segment in slice {
                    segments.push(Segment {
                        start: segment.start,
                        end: segment.end,
                        speaker: segment.speaker,
                    });
                }
            }

            sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(segments_ptr);
            sherpa_rs_sys::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(result);

            if segments.is_empty() {
                return Err(Error::Diarization(
                    "No segments found or invalid pointer.".into(),
                ));
            }
            Ok(segments)
        }
    }
}

/// sherpa-onnx treats zero or negative thread counts as "unspecified", which
/// has meant one thread in practice. Clamp to at least one so both models get
/// the count the caller asked for.
fn effective_threads(requested: i32) -> i32 {
    requested.max(1)
}

fn path_cstring(path: &Path, model: &str) -> Result<CString> {
    let text = path.to_str().ok_or_else(|| {
        Error::Diarization(format!(
            "The {model} model path is not valid UTF-8: {}",
            path.display()
        ))
    })?;
    CString::new(text).map_err(|_| {
        Error::Diarization(format!(
            "The {model} model path contains a NUL byte: {}",
            path.display()
        ))
    })
}

unsafe extern "C" fn progress_callback_wrapper(
    num_processed_chunk: i32,
    num_total_chunks: i32,
    arg: *mut c_void,
) -> i32 {
    if arg.is_null() {
        return 0;
    }
    // SAFETY: `compute` passes a pointer to a live `Box<ProgressCallback>`.
    let callback = unsafe { &mut *(arg as *mut ProgressCallback) };
    callback(num_processed_chunk, num_total_chunks)
}

// The native pointer is only used from the owning thread; upstream marks its
// equivalent type Send + Sync, and the worker may move it between threads.
unsafe impl Send for LocalDiarizer {}
unsafe impl Sync for LocalDiarizer {}

impl Drop for LocalDiarizer {
    fn drop(&mut self) {
        // SAFETY: `sd` is the non-null pointer returned by the create call and
        // is destroyed exactly once here.
        unsafe {
            sherpa_rs_sys::SherpaOnnxDestroyOfflineSpeakerDiarization(self.sd);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thread_count_is_passed_through_and_never_zero() {
        assert_eq!(effective_threads(8), 8);
        assert_eq!(effective_threads(2), 2);
        assert_eq!(effective_threads(0), 1);
        assert_eq!(effective_threads(-4), 1);
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_model_paths_return_an_error_instead_of_panicking() {
        use std::os::unix::ffi::OsStrExt;

        let path = Path::new(std::ffi::OsStr::from_bytes(b"/tmp/\xff-model.onnx"));
        let error = path_cstring(path, "segmentation").expect_err("invalid UTF-8 accepted");
        assert!(error.to_string().contains("not valid UTF-8"), "{error}");
    }
}
