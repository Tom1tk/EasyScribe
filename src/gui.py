"""
gui.py - CustomTkinter main window for EasyScribe v2.0.

All transcription and recording work runs in daemon background threads.
The GUI communicates with worker threads exclusively via self.after(0, ...) callbacks.
"""

import logging
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

import customtkinter as ctk  # type: ignore

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore
    _DND_AVAILABLE = True
except Exception:
    DND_FILES = None  # type: ignore
    _DND_AVAILABLE = False

_AppBase = ctk.CTk  # type: ignore

from config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_OUTPUT_DIR,
    MIN_FREE_DISK_BYTES,
    SUPPORTED_EXTENSIONS,
)
from ffmpeg_wrapper import (
    CancelledError as FFmpegCancelledError,
    FFmpegExtractionError,
    FFmpegNotFoundError,
    extract_audio,
)
from diarizer import DiarizationEngine  # type: ignore
from transcriber import (
    CancelledError as TranscribeCancelledError,
    ModelNotFoundError,
    TranscriptionEngine,
    TranscriptionError,
)
from mic_recorder import MicRecorder
from live_transcriber import LiveTranscriber

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_STATUS_COLOURS: dict[str, str] = {
    "Ready": "#4CAF50",
    "Loading Model": "#FF9800",
    "Extracting Audio": "#2196F3",
    "Transcribing": "#2196F3",
    "Recording": "#E91E63",
    "Identifying Speakers": "#2196F3",
    "Naming Speakers": "#9C27B0",
    "Writing Transcript": "#2196F3",
    "Done": "#4CAF50",
    "Cancelled": "#FF9800",
    "Failed": "#F44336",
}


# ─── Speaker naming dialog ────────────────────────────────────────────────────


class SpeakerNamingDialog(ctk.CTkToplevel):
    """Modal popup for renaming speakers after diarization."""

    def __init__(
        self,
        parent: ctk.CTk,
        speaker_map: dict[str, str],
        clips_dict: dict[str, Path],
        done_event: threading.Event,
    ) -> None:
        super().__init__(parent)
        self._speaker_map = speaker_map
        self._clips_dict = clips_dict
        self._done_event = done_event
        self._entries: dict[str, ctk.CTkEntry] = {}

        self.title("Name the Speakers")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        py = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{max(0, px)}+{max(0, py)}")

    def _build(self) -> None:
        pad = {"padx": 16, "pady": 6}
        ctk.CTkLabel(
            self, text="Identify the Speakers", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=16, pady=(16, 4), sticky="w")
        ctk.CTkLabel(
            self,
            text="Play a sample to identify each speaker, then enter their name.",
            text_color="gray60",
        ).grid(row=1, column=0, columnspan=3, padx=16, pady=(0, 12), sticky="w")

        for col, heading in enumerate(("Speaker", "Sample", "Name")):
            ctk.CTkLabel(self, text=heading, font=ctk.CTkFont(weight="bold")).grid(
                row=2, column=col, **pad, sticky="w" if col != 1 else ""
            )

        for i, (raw_label, default_name) in enumerate(self._speaker_map.items(), start=3):
            ctk.CTkLabel(self, text=default_name).grid(row=i, column=0, **pad, sticky="w")
            clip_path = self._clips_dict.get(raw_label)
            ctk.CTkButton(
                self,
                text="▶ Play",
                width=80,
                state="normal" if clip_path else "disabled",
                command=lambda p=clip_path: self._play_clip(p),
            ).grid(row=i, column=1, **pad)
            entry = ctk.CTkEntry(self, width=200, placeholder_text=default_name)
            entry.insert(0, default_name)
            entry.grid(row=i, column=2, **pad, sticky="ew")
            self._entries[raw_label] = entry

        confirm_row = 3 + len(self._speaker_map)
        ctk.CTkButton(
            self,
            text="Use These Names",
            font=ctk.CTkFont(weight="bold"),
            width=220,
            command=self._on_confirm,
        ).grid(row=confirm_row, column=0, columnspan=3, padx=16, pady=(12, 20))

    def _play_clip(self, clip_path: "Path | None") -> None:
        if clip_path is None:
            return
        try:
            import sys as _sys
            if _sys.platform == "win32":
                import winsound
                winsound.PlaySound(str(clip_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            logger.warning(f"Clip playback failed: {exc}")

    def _on_confirm(self) -> None:
        for raw_label, entry in self._entries.items():
            name = entry.get().strip()
            if name:
                self._speaker_map[raw_label] = name
        self._dismiss()

    def _on_close(self) -> None:
        self._dismiss()

    def _dismiss(self) -> None:
        self.grab_release()
        self.destroy()
        self._done_event.set()


# ─── Main window ─────────────────────────────────────────────────────────────


class TranscriberApp(_AppBase):  # type: ignore
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("820x720")
        self.minsize(700, 580)
        self.resizable(True, True)

        # ── State ────────────────────────────────────────────────────────────
        self._selected_files: list[Path] = []
        self._output_folder: Path | None = None
        self._cancel_event = threading.Event()
        self._stop_recording_event = threading.Event()
        self._engine = TranscriptionEngine()
        self._mic_recorder = MicRecorder()
        self._live_transcriber = LiveTranscriber()
        self._last_output_folder: Path | None = None

        # ── tkdnd ─────────────────────────────────────────────────────────────
        self._dnd_enabled: bool = False
        if _DND_AVAILABLE:
            try:
                TkinterDnD._require(self)  # type: ignore
                self._dnd_enabled = True
            except Exception as exc:
                logger.warning(f"tkdnd extension failed — drag-and-drop disabled: {exc}")

        # ── Diarization availability ──────────────────────────────────────────
        self._diarization_available: bool = DiarizationEngine().is_available()

        # ── Mic selector state ────────────────────────────────────────────────
        self._mic_options: list[str] = []
        self._mic_index_map: dict[str, int | None] = {}

        self._build_ui()
        self._set_ui_state("idle")
        self._update_status("Ready")

        logger.info("GUI initialised")

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # ── File selection ────────────────────────────────────────────────────
        file_frame = ctk.CTkFrame(self)
        file_frame.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        file_frame.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(
            file_frame, text="Input Files", font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w")

        self._select_files_btn = ctk.CTkButton(
            file_frame, text="Select File(s)", width=130, command=self._on_select_files
        )
        self._select_files_btn.grid(row=1, column=0, padx=10, pady=8, sticky="w")

        self._clear_files_btn = ctk.CTkButton(
            file_frame,
            text="Clear",
            width=70,
            fg_color="gray40",
            hover_color="gray30",
            command=self._on_clear_files,
        )
        self._clear_files_btn.grid(row=1, column=1, padx=(0, 10), pady=8, sticky="w")

        self._files_label = ctk.CTkLabel(
            file_frame, text="No files selected", anchor="w", text_color="gray70"
        )
        self._files_label.grid(row=1, column=2, padx=10, pady=8, sticky="ew")

        self._drop_zone = ctk.CTkLabel(
            file_frame,
            text=(
                "⬇  Drop media files here"
                if self._dnd_enabled
                else "Drag & drop unavailable — use Select File(s)"
            ),
            height=50,
            corner_radius=8,
            fg_color=("gray85", "gray20"),
            text_color=("gray40", "gray60"),
        )
        self._drop_zone.grid(row=2, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="ew")
        if self._dnd_enabled and DND_FILES is not None:
            self._drop_zone.drop_target_register(DND_FILES)
            self._drop_zone.dnd_bind("<<Drop>>", self._on_drop)

        # ── Output folder ─────────────────────────────────────────────────────
        out_frame = ctk.CTkFrame(self)
        out_frame.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        out_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            out_frame, text="Output Folder", font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 4), sticky="w")

        self._select_output_btn = ctk.CTkButton(
            out_frame, text="Select Folder", width=130, command=self._on_select_output_folder
        )
        self._select_output_btn.grid(row=1, column=0, padx=10, pady=8, sticky="w")

        self._output_label = ctk.CTkLabel(
            out_frame, text="Same folder as input file(s)", anchor="w", text_color="gray70"
        )
        self._output_label.grid(row=1, column=1, padx=10, pady=8, sticky="ew")

        # ── Options ───────────────────────────────────────────────────────────
        opts_frame = ctk.CTkFrame(self)
        opts_frame.grid(row=2, column=0, padx=12, pady=6, sticky="ew")

        ctk.CTkLabel(
            opts_frame, text="Options", font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, columnspan=4, padx=10, pady=(8, 4), sticky="w")

        self._timestamps_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opts_frame, text="Include timestamps", variable=self._timestamps_var
        ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")

        self._diarize_var = ctk.BooleanVar(value=False)
        _diarize_cb = ctk.CTkCheckBox(
            opts_frame,
            text="Identify speakers",
            variable=self._diarize_var,
            state="normal" if self._diarization_available else "disabled",
        )
        _diarize_cb.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="w")
        if not self._diarization_available:
            _diarize_cb.configure(text="Identify speakers  (model not found)")

        # Microphone selector
        ctk.CTkLabel(opts_frame, text="Microphone:", anchor="w").grid(
            row=2, column=1, padx=(20, 4), pady=(0, 10), sticky="w"
        )
        self._mic_options, self._mic_index_map = self._build_mic_options()
        self._mic_var = ctk.StringVar(value=self._mic_options[0])
        self._mic_menu = ctk.CTkOptionMenu(
            opts_frame,
            variable=self._mic_var,
            values=self._mic_options,
            width=220,
        )
        self._mic_menu.grid(row=2, column=2, padx=(0, 10), pady=(0, 10), sticky="w")

        # ── Actions ───────────────────────────────────────────────────────────
        action_frame = ctk.CTkFrame(self)
        action_frame.grid(row=3, column=0, padx=12, pady=6, sticky="ew")

        self._transcribe_btn = ctk.CTkButton(
            action_frame,
            text="Transcribe",
            width=140,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_transcribe,
        )
        self._transcribe_btn.grid(row=0, column=0, padx=10, pady=10)

        self._record_btn = ctk.CTkButton(
            action_frame,
            text="Record",
            width=140,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#E91E63",
            hover_color="#C2185B",
            command=self._on_record,
        )
        self._record_btn.grid(row=0, column=1, padx=(0, 10), pady=10)

        self._cancel_btn = ctk.CTkButton(
            action_frame,
            text="Cancel",
            width=100,
            fg_color="gray40",
            hover_color="gray30",
            command=self._on_cancel,
        )
        self._cancel_btn.grid(row=0, column=2, padx=(0, 10), pady=10)

        self._open_output_btn = ctk.CTkButton(
            action_frame,
            text="Open Output Folder",
            width=160,
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray25"),
            command=self._on_open_output_folder,
        )
        self._open_output_btn.grid(row=0, column=3, padx=(0, 10), pady=10)

        # ── Progress & status ─────────────────────────────────────────────────
        progress_frame = ctk.CTkFrame(self)
        progress_frame.grid(row=4, column=0, padx=12, pady=6, sticky="ew")
        progress_frame.grid_columnconfigure(0, weight=1)

        self._progress_bar = ctk.CTkProgressBar(progress_frame, height=14)
        self._progress_bar.set(0)
        self._progress_bar.grid(row=0, column=0, padx=10, pady=(10, 4), sticky="ew")

        status_row = ctk.CTkFrame(progress_frame, fg_color="transparent")
        status_row.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="ew")
        status_row.grid_columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(
            status_row,
            text="Ready",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self._status_label.grid(row=0, column=0, sticky="w")

        self._batch_label = ctk.CTkLabel(
            status_row, text="", font=ctk.CTkFont(size=12), anchor="e", text_color="gray60"
        )
        self._batch_label.grid(row=0, column=1, sticky="e")

        # ── Log box ───────────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=5, column=0, padx=12, pady=(6, 12), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_frame, text="Log", font=ctk.CTkFont(size=13, weight="bold")
        ).grid(row=0, column=0, padx=10, pady=(8, 2), sticky="w")

        self._log_box = ctk.CTkTextbox(
            log_frame,
            state="disabled",
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word",
        )
        self._log_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")

    # ─────────────────────────────────────────────────────────────────────────
    # Drag & drop
    # ─────────────────────────────────────────────────────────────────────────

    def _on_drop(self, event: object) -> None:
        raw = getattr(event, "data", "")
        try:
            paths_raw: list[str] = self.tk.splitlist(raw)  # type: ignore[attr-defined]
        except Exception:
            paths_raw = raw.split()

        valid = [
            Path(p) for p in paths_raw
            if Path(p).is_file() and Path(p).suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if valid:
            self._add_files(valid)
        else:
            messagebox.showwarning(
                "Unsupported Files",
                "None of the dropped files are supported media types.\n\n"
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Button handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_select_files(self) -> None:
        ext_list = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Select media file(s)",
            filetypes=[("Media files", ext_list), ("All files", "*.*")],
        )
        if paths:
            self._add_files([Path(p) for p in paths])

    def _on_clear_files(self) -> None:
        self._selected_files.clear()
        self._files_label.configure(text="No files selected", text_color="gray70")

    def _on_select_output_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder for transcripts")
        if folder:
            self._output_folder = Path(folder)
            self._output_label.configure(
                text=str(self._output_folder), text_color=("gray20", "gray90")
            )

    def _on_transcribe(self) -> None:
        if not self._selected_files:
            messagebox.showwarning("No Files", "Please select at least one media file first.")
            return

        check_dir = self._output_folder or self._selected_files[0].parent
        try:
            usage = shutil.disk_usage(check_dir)
            if usage.free < MIN_FREE_DISK_BYTES:
                if not messagebox.askyesno(
                    "Low Disk Space",
                    f"Available disk space is only {usage.free // (1024 * 1024)} MB.\n"
                    "Transcription may fail. Continue anyway?",
                ):
                    return
        except Exception:
            pass

        self._cancel_event.clear()
        self._set_ui_state("running")
        self._clear_log()
        self._progress_bar.set(0)

        threading.Thread(
            target=self._transcription_worker, daemon=True, name="TranscriptionWorker"
        ).start()

    def _on_record(self) -> None:
        self._stop_recording_event.clear()
        self._set_ui_state("recording")
        self._clear_log()
        self._safe_append_log("[Record] Starting live recording…")
        self._record_btn.configure(text="Stop Recording", command=self._on_stop_recording)

        threading.Thread(
            target=self._recording_worker, daemon=True, name="RecordingWorker"
        ).start()

    def _on_stop_recording(self) -> None:
        self._stop_recording_event.set()

    def _on_cancel(self) -> None:
        self._cancel_event.set()
        self._update_status("Cancelling…")
        self._cancel_btn.configure(state="disabled")

    def _on_open_output_folder(self) -> None:
        folder = self._last_output_folder or self._output_folder or DEFAULT_OUTPUT_DIR
        if folder and folder.exists():
            os.startfile(str(folder))
        else:
            messagebox.showinfo("No Folder", "No output folder to open yet.")

    # ─────────────────────────────────────────────────────────────────────────
    # Transcription worker thread
    # ─────────────────────────────────────────────────────────────────────────

    def _transcription_worker(self) -> None:
        files = list(self._selected_files)
        total = len(files)
        failed_count = 0

        for idx, input_file in enumerate(files, start=1):
            if self._cancel_event.is_set():
                break

            self._safe_set_batch_label(f"File {idx} / {total}: {input_file.name}")
            self._safe_append_log(f"\n{'─' * 50}")
            self._safe_append_log(f"[File {idx}/{total}] {input_file.name}")

            temp_wav: Path | None = None
            try:
                self._safe_update_status("Extracting Audio")
                self._safe_set_progress(0.0)

                temp_wav = extract_audio(input_file, self._cancel_event, self._safe_append_log)

                if self._cancel_event.is_set():
                    raise FFmpegCancelledError("Cancelled")

                output_path = self._resolve_output_path(input_file)
                self._last_output_folder = output_path.parent

                self._engine.transcribe(
                    audio_path=temp_wav,
                    output_path=output_path,
                    add_timestamps=self._timestamps_var.get(),
                    cancel_event=self._cancel_event,
                    status_callback=self._safe_update_status,
                    progress_callback=self._safe_set_progress,
                    log_callback=self._safe_append_log,
                    diarize=self._diarize_var.get(),
                    speaker_name_callback=(
                        self._make_speaker_naming_callback() if self._diarize_var.get() else None
                    ),
                )

            except (FFmpegCancelledError, TranscribeCancelledError):
                self._safe_append_log("[Cancelled] Operation stopped by user")
                break

            except (FFmpegNotFoundError, ModelNotFoundError) as exc:
                logger.exception("Fatal dependency missing")
                self._safe_append_log(f"[Fatal] {exc}")
                self.after(0, lambda e=str(exc): messagebox.showerror("Missing Dependency", e))
                failed_count += 1
                break

            except (FFmpegExtractionError, TranscriptionError) as exc:
                logger.error(f"File {input_file.name} failed: {exc}")
                self._safe_append_log(f"[Error] {input_file.name}: {exc}")
                failed_count += 1

            except Exception as exc:
                logger.exception(f"Unexpected error on {input_file.name}")
                self._safe_append_log(f"[Unexpected Error] {input_file.name}: {exc}")
                failed_count += 1

            finally:
                if temp_wav and temp_wav.exists():
                    try:
                        temp_wav.unlink()
                    except OSError:
                        pass

        if self._cancel_event.is_set():
            final_status = "Cancelled"
        elif failed_count == 0:
            final_status = "Done"
        elif failed_count == total:
            final_status = "Failed"
        else:
            final_status = "Done"
            self._safe_append_log(
                f"\n[Summary] Completed with {failed_count} error(s) out of {total} file(s)"
            )

        self._safe_update_status(final_status)
        self._safe_set_batch_label("")
        self.after(0, lambda: self._set_ui_state("idle"))

    # ─────────────────────────────────────────────────────────────────────────
    # Recording worker thread
    # ─────────────────────────────────────────────────────────────────────────

    def _recording_worker(self) -> None:
        transcript_lines: list[str] = []

        def on_segment(text: str) -> None:
            transcript_lines.append(text)
            self._safe_append_log(text)

        try:
            self._engine._ensure_model_loaded(self._safe_update_status)
            self._safe_update_status("Recording")

            mic_label = self._mic_var.get()
            device_index = self._mic_index_map.get(mic_label)

            DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            self._mic_recorder.start(device_index, DEFAULT_OUTPUT_DIR)
            mic_queue = self._mic_recorder.get_queue()

            self._live_transcriber.start(
                self._engine._recognizer,
                mic_queue,
                self._stop_recording_event,
                on_segment,
            )

            self._safe_append_log("[Record] Listening — click Stop Recording when done")
            self._stop_recording_event.wait()

            self._live_transcriber.stop()
            wav_path = self._mic_recorder.stop(convert_to_wav=True)

            # Write accumulated transcript
            if transcript_lines:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                txt_path = DEFAULT_OUTPUT_DIR / f"recording_{ts}.txt"
                try:
                    txt_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
                    self._last_output_folder = DEFAULT_OUTPUT_DIR
                    self._safe_append_log(f"[Done] Transcript saved: {txt_path}")
                except OSError as exc:
                    self._safe_append_log(f"[Error] Could not write transcript: {exc}")

            if wav_path:
                self._safe_append_log(f"[Done] Audio saved: {wav_path}")
                self._last_output_folder = DEFAULT_OUTPUT_DIR

            self._safe_update_status("Done")

        except ModelNotFoundError as exc:
            logger.exception("Model not found during recording")
            self._safe_append_log(f"[Fatal] {exc}")
            self.after(0, lambda e=str(exc): messagebox.showerror("Missing Model", e))
            self._safe_update_status("Failed")

        except Exception as exc:
            logger.exception("Recording worker error")
            self._safe_append_log(f"[Error] {exc}")
            self._safe_update_status("Failed")

        finally:
            self.after(
                0,
                lambda: (
                    self._record_btn.configure(text="Record", command=self._on_record),
                    self._set_ui_state("idle"),
                ),
            )

    def _make_speaker_naming_callback(self) -> Callable:
        def callback(speaker_map: dict, clips_dict: dict) -> None:
            done_event = threading.Event()
            self.after(
                0,
                lambda sm=speaker_map, cd=clips_dict, ev=done_event:
                    SpeakerNamingDialog(self, sm, cd, ev),
            )
            done_event.wait()

        return callback

    # ─────────────────────────────────────────────────────────────────────────
    # Thread-safe GUI helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _safe_update_status(self, status: str) -> None:
        self.after(0, lambda s=status: self._update_status(s))

    def _safe_set_progress(self, value: float) -> None:
        self.after(0, lambda v=value: self._progress_bar.set(v))

    def _safe_append_log(self, text: str) -> None:
        self.after(0, lambda t=text: self._append_log(t))

    def _safe_set_batch_label(self, text: str) -> None:
        self.after(0, lambda t=text: self._batch_label.configure(text=t))

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_mic_options(self) -> tuple[list[str], dict[str, int | None]]:
        options: list[str] = ["Default microphone"]
        index_map: dict[str, int | None] = {"Default microphone": None}

        try:
            devices = MicRecorder.list_devices()
            for d in devices:
                label = f"Mic {d['index']}: {d['name']}"
                if label not in index_map:
                    options.append(label)
                    index_map[label] = int(d["index"])
        except Exception as exc:
            logger.warning(f"Could not enumerate microphones: {exc}")

        return options, index_map

    def _add_files(self, paths: list[Path]) -> None:
        existing = set(self._selected_files)
        added = 0
        for p in paths:
            if p not in existing and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                self._selected_files.append(p)
                existing.add(p)
                added += 1

        count = len(self._selected_files)
        if count == 0:
            self._files_label.configure(text="No files selected", text_color="gray70")
        elif count == 1:
            self._files_label.configure(
                text=self._selected_files[0].name, text_color=("gray20", "gray90")
            )
        else:
            self._files_label.configure(
                text=f"{count} files selected", text_color=("gray20", "gray90")
            )

    def _resolve_output_path(self, input_file: Path) -> Path:
        folder = self._output_folder or input_file.parent
        return folder / (input_file.stem + ".txt")

    def _update_status(self, status: str) -> None:
        colour = _STATUS_COLOURS.get(status, "gray80")
        self._status_label.configure(text=status, text_color=colour)

    def _append_log(self, text: str) -> None:
        self._log_box.configure(state="normal")
        self._log_box.insert("end", text + "\n")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _set_ui_state(self, state: str) -> None:
        """Toggle widgets between 'idle', 'running', and 'recording' states."""
        is_running = state == "running"
        is_recording = state == "recording"
        is_busy = is_running or is_recording

        self._transcribe_btn.configure(state="disabled" if is_busy else "normal")
        self._select_files_btn.configure(state="disabled" if is_busy else "normal")
        self._select_output_btn.configure(state="disabled" if is_busy else "normal")
        self._clear_files_btn.configure(state="disabled" if is_busy else "normal")
        self._mic_menu.configure(state="disabled" if is_recording else "normal")

        # Record button: disabled while transcription runs; becomes Stop while recording
        self._record_btn.configure(state="disabled" if is_running else "normal")

        # Cancel button: only active during file transcription
        self._cancel_btn.configure(state="normal" if is_running else "disabled")

        self._open_output_btn.configure(
            state="normal" if (not is_busy and self._last_output_folder) else
            ("disabled" if is_busy else "normal")
        )
        if not is_busy:
            self._progress_bar.set(0.0)
