"""
gui.py - CustomTkinter main window for Transcriber7.

All transcription work runs in a daemon background thread. The GUI
communicates with the worker thread exclusively via thread-safe
`self.after(0, lambda: ...)` callbacks — widgets are never touched
from outside the main thread.
"""

import logging
import os
import shutil
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Callable

import customtkinter as ctk  # type: ignore

from config import APP_NAME, APP_VERSION, MIN_FREE_DISK_BYTES, SUPPORTED_EXTENSIONS
from ffmpeg_wrapper import (
    CancelledError as FFmpegCancelledError,
    FFmpegExtractionError,
    FFmpegNotFoundError,
    extract_audio,
)
from transcriber import (
    CancelledError as TranscribeCancelledError,
    ModelNotFoundError,
    TranscriptionEngine,
    TranscriptionError,
    list_gpus,
)

logger = logging.getLogger(__name__)

# ─── Appearance ───────────────────────────────────────────────────────────────

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Status colours
_STATUS_COLOURS: dict[str, str] = {
    "Ready": "#4CAF50",
    "Loading Model": "#FF9800",
    "Extracting Audio": "#2196F3",
    "Transcribing": "#2196F3",
    "Writing Transcript": "#2196F3",
    "Done": "#4CAF50",
    "Cancelled": "#FF9800",
    "Failed": "#F44336",
}


class TranscriberApp(ctk.CTk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()

        self.title(f"{APP_NAME} v{APP_VERSION}")
        self.geometry("820x680")
        self.minsize(700, 560)
        self.resizable(True, True)

        # ── State ─────────────────────────────────────────────────────────────
        self._selected_files: list[Path] = []
        self._output_folder: Path | None = None
        self._cancel_event = threading.Event()
        self._engine = TranscriptionEngine()
        self._last_output_folder: Path | None = None

        # Set initial GPU preference on engine (will be overridden once UI is built)
        # Build GPU options first so we know the default
        self._gpu_options: list[str] = []
        self._gpu_index_map: dict[str, int | None] = {}

        # ── Enable drag & drop (tkinterdnd2) ──────────────────────────────────
        self._dnd_enabled = self._try_enable_dnd()

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_ui()
        self._set_ui_state("idle")
        self._update_status("Ready")

        logger.info("GUI initialised")

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)  # log box expands

        # ── File selection ────────────────────────────────────────────────────
        file_frame = ctk.CTkFrame(self)
        file_frame.grid(row=0, column=0, padx=12, pady=(12, 6), sticky="ew")
        file_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(file_frame, text="Input Files", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w"
        )

        self._select_files_btn = ctk.CTkButton(
            file_frame,
            text="Select File(s)",
            width=130,
            command=self._on_select_files,
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
            file_frame,
            text="No files selected",
            anchor="w",
            text_color="gray70",
        )
        self._files_label.grid(row=1, column=2, padx=10, pady=8, sticky="ew")

        # Drop zone
        self._drop_zone = ctk.CTkLabel(
            file_frame,
            text="⬇  Drop media files here" if self._dnd_enabled else "Drag & drop unavailable — use Select File(s)",
            height=50,
            corner_radius=8,
            fg_color=("gray85", "gray20"),
            text_color=("gray40", "gray60"),
        )
        self._drop_zone.grid(row=2, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="ew")

        if self._dnd_enabled:
            self._drop_zone.drop_target_register("DND_Files")  # type: ignore[attr-defined]
            self._drop_zone.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore[attr-defined]

        # ── Output folder ─────────────────────────────────────────────────────
        out_frame = ctk.CTkFrame(self)
        out_frame.grid(row=1, column=0, padx=12, pady=6, sticky="ew")
        out_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(out_frame, text="Output Folder", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=10, pady=(8, 4), sticky="w"
        )

        self._select_output_btn = ctk.CTkButton(
            out_frame,
            text="Select Folder",
            width=130,
            command=self._on_select_output_folder,
        )
        self._select_output_btn.grid(row=1, column=0, padx=10, pady=8, sticky="w")

        self._output_label = ctk.CTkLabel(
            out_frame,
            text="Same folder as input file(s)",
            anchor="w",
            text_color="gray70",
        )
        self._output_label.grid(row=1, column=1, padx=10, pady=8, sticky="ew")

        # ── Options ───────────────────────────────────────────────────────────
        opts_frame = ctk.CTkFrame(self)
        opts_frame.grid(row=2, column=0, padx=12, pady=6, sticky="ew")

        ctk.CTkLabel(opts_frame, text="Options", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="w"
        )

        self._timestamps_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            opts_frame,
            text="Include timestamps in transcript",
            variable=self._timestamps_var,
        ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")

        # GPU selector
        ctk.CTkLabel(opts_frame, text="Device:", anchor="w").grid(
            row=1, column=1, padx=(20, 4), pady=(0, 10), sticky="w"
        )
        self._gpu_options, self._gpu_index_map = self._build_gpu_options()
        self._gpu_var = ctk.StringVar(value=self._gpu_options[0])
        self._gpu_menu = ctk.CTkOptionMenu(
            opts_frame,
            variable=self._gpu_var,
            values=self._gpu_options,
            width=220,
            command=self._on_gpu_changed,
        )
        self._gpu_menu.grid(row=1, column=2, padx=(0, 10), pady=(0, 10), sticky="w")

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

        self._cancel_btn = ctk.CTkButton(
            action_frame,
            text="Cancel",
            width=100,
            fg_color="gray40",
            hover_color="gray30",
            command=self._on_cancel,
        )
        self._cancel_btn.grid(row=0, column=1, padx=(0, 10), pady=10)

        self._open_output_btn = ctk.CTkButton(
            action_frame,
            text="Open Output Folder",
            width=160,
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray25"),
            command=self._on_open_output_folder,
        )
        self._open_output_btn.grid(row=0, column=2, padx=(0, 10), pady=10)

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
            status_row,
            text="",
            font=ctk.CTkFont(size=12),
            anchor="e",
            text_color="gray60",
        )
        self._batch_label.grid(row=0, column=1, sticky="e")

        # ── Log box ───────────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=5, column=0, padx=12, pady=(6, 12), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(log_frame, text="Log", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, padx=10, pady=(8, 2), sticky="w"
        )

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

    def _try_enable_dnd(self) -> bool:
        """Attempt to initialise tkinterdnd2. Returns True on success."""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD  # type: ignore  # noqa: F401
            # Replace the Tk root with a TkinterDnD-compatible one
            # (tkinterdnd2 patches the root widget in place when imported)
            logger.info("tkinterdnd2 drag-and-drop enabled")
            return True
        except Exception as exc:
            logger.warning(f"tkinterdnd2 not available, drag-and-drop disabled: {exc}")
            return False

    def _on_drop(self, event: object) -> None:
        """Handle files dropped onto the drop zone."""
        raw = getattr(event, "data", "")
        try:
            paths_raw: list[str] = self.tk.splitlist(raw)  # type: ignore[attr-defined]
        except Exception:
            paths_raw = raw.split()

        valid: list[Path] = []
        for p_str in paths_raw:
            p = Path(p_str)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                valid.append(p)

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
        """Open a file picker for one or more media files."""
        ext_list = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        paths = filedialog.askopenfilenames(
            title="Select media file(s)",
            filetypes=[
                ("Media files", ext_list),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self._add_files([Path(p) for p in paths])

    def _on_clear_files(self) -> None:
        self._selected_files.clear()
        self._files_label.configure(text="No files selected", text_color="gray70")
        logger.info("File selection cleared")

    def _on_select_output_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder for transcripts")
        if folder:
            self._output_folder = Path(folder)
            self._output_label.configure(
                text=str(self._output_folder), text_color=("gray20", "gray90")
            )
            logger.info(f"Output folder set: {self._output_folder}")

    def _on_transcribe(self) -> None:
        """Validate inputs and start the transcription worker thread."""
        if not self._selected_files:
            messagebox.showwarning("No Files", "Please select at least one media file first.")
            return

        # Disk space check
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

        thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
            name="TranscriptionWorker",
        )
        thread.start()
        logger.info(f"Started transcription worker for {len(self._selected_files)} file(s)")

    def _on_cancel(self) -> None:
        """Signal the worker thread to cancel."""
        logger.info("Cancel requested by user")
        self._cancel_event.set()
        self._update_status("Cancelling…")
        self._cancel_btn.configure(state="disabled")

    def _on_open_output_folder(self) -> None:
        """Open the last used output folder in Windows Explorer."""
        folder = self._last_output_folder or self._output_folder
        if folder and folder.exists():
            os.startfile(str(folder))
        elif self._selected_files:
            os.startfile(str(self._selected_files[0].parent))
        else:
            messagebox.showinfo("No Folder", "No output folder to open yet.")

    # ─────────────────────────────────────────────────────────────────────────
    # Worker thread
    # ─────────────────────────────────────────────────────────────────────────

    def _transcription_worker(self) -> None:
        """
        Background thread: processes all selected files.

        Never touches any widget directly — all GUI updates go through
        self.after(0, lambda: ...) to ensure thread safety.
        """
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
                # ── Step 1: Extract audio ─────────────────────────────────
                self._safe_update_status("Extracting Audio")
                self._safe_set_progress(0.0)

                temp_wav = extract_audio(
                    input_file,
                    self._cancel_event,
                    self._safe_append_log,
                )

                if self._cancel_event.is_set():
                    raise FFmpegCancelledError("Cancelled")

                # ── Step 2: Transcribe ────────────────────────────────────
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
                )

            except (FFmpegCancelledError, TranscribeCancelledError):
                self._safe_append_log("[Cancelled] Operation stopped by user")
                logger.info("Worker cancelled")
                break

            except (FFmpegNotFoundError, ModelNotFoundError) as exc:
                # Fatal: missing dependency — stop entire batch
                logger.exception("Fatal dependency missing")
                self._safe_append_log(f"[Fatal] {exc}")
                self.after(
                    0,
                    lambda e=str(exc): messagebox.showerror("Missing Dependency", e),
                )
                failed_count += 1
                break

            except (FFmpegExtractionError, TranscriptionError) as exc:
                # Per-file error: log and continue to next file
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

        # ── Final status ──────────────────────────────────────────────────────
        if self._cancel_event.is_set():
            final_status = "Cancelled"
        elif failed_count == 0:
            final_status = "Done"
        elif failed_count == total:
            final_status = "Failed"
        else:
            final_status = "Done"  # partial success
            self._safe_append_log(
                f"\n[Summary] Completed with {failed_count} error(s) out of {total} file(s)"
            )

        self._safe_update_status(final_status)
        self._safe_set_batch_label("")
        self.after(0, lambda: self._set_ui_state("idle"))

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

    def _build_gpu_options(self) -> tuple[list[str], dict[str, int | None]]:
        """
        Build the list of device options for the GPU selector dropdown.

        Returns (option_labels, label_to_device_index_map).
        device_index -1 means CPU, None means auto-select first GPU.
        """
        options: list[str] = ["CPU only"]
        index_map: dict[str, int | None] = {"CPU only": -1}

        gpus = list_gpus()
        if gpus:
            options.insert(0, "Auto (best GPU)")
            index_map["Auto (best GPU)"] = None  # engine will auto-select GPU 0
            for gpu in gpus:
                label = f"GPU {gpu['index']}: {gpu['name']}"
                options.append(label) if label not in options else None
                index_map[label] = int(gpu["index"])  # type: ignore[arg-type]
        else:
            # No GPU available; only show CPU
            logger.info("No CUDA GPUs found — device selector shows CPU only")

        return options, index_map

    def _on_gpu_changed(self, selection: str) -> None:
        """Called when the user changes the device dropdown."""
        gpu_index = self._gpu_index_map.get(selection, -1)
        self._engine.preferred_gpu_index = gpu_index
        # Unload the cached model so it reloads with the new device on next run
        self._engine.reload_model()
        logger.info(f"Device changed to: {selection!r} (device_index={gpu_index})")
        self._safe_append_log(f"[Device] Changed to: {selection}")

    def _add_files(self, paths: list[Path]) -> None:
        """Add files to the selection, filtering duplicates and unsupported types."""
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
        logger.info(f"Added {added} file(s); total: {count}")

    def _resolve_output_path(self, input_file: Path) -> Path:
        """Determine the .txt output path for a given input file."""
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
        """Toggle widgets between 'idle' and 'running' states."""
        running = state == "running"
        self._transcribe_btn.configure(state="disabled" if running else "normal")
        self._cancel_btn.configure(state="normal" if running else "disabled")
        self._select_files_btn.configure(state="disabled" if running else "normal")
        self._select_output_btn.configure(state="disabled" if running else "normal")
        self._clear_files_btn.configure(state="disabled" if running else "normal")
        self._open_output_btn.configure(
            state="normal" if (not running and self._last_output_folder) else
            ("disabled" if running else "normal")
        )
        if not running:
            self._progress_bar.set(0.0)
