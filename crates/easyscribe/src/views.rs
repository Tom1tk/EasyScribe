//! The complete Rune view: Files, Record, activity/results, naming, details and
//! the Privacy overlay. It is a pure function of [`AppState`].
//!
//! Layout follows `docs/rewrite/REFERENCE_UI.md`: 24 px gutter, 12 px card gap,
//! white 14-radius cards, 18 px inner padding, feature colours per task.

use rune::{Color, Cursor, Cx, El, ScrollSpec, Shape, col, el, row, text, text_input};

use crate::app::{
    Activity, AppState, BarMode, FileState, FindCard, MessageKind, Mode, NamingState,
    PreparingState, RecordMode, UiState,
};
use crate::theme::{Theme, metric, type_};
use crate::widgets::{
    Bar, ButtonKind, button, card, checkbox, empty_spacer, info_pill, label, link_text,
    message_banner, muted, progress_bar, radio, read_text, small_muted, status_pill, step_header,
    strong,
};

pub fn view(s: &AppState, cx: &Cx<AppState>) -> El {
    if let Some(preparing) = &s.preparing {
        return preparing_view(s, preparing);
    }
    let t = &s.theme;
    let mut main = col()
        .key("window")
        .w_full()
        .h_full()
        .fill(t.bg)
        .px(metric::GUTTER)
        .child(header(s))
        .child(mode_switch(s));

    let content = match s.mode {
        Mode::Files => files_view(s),
        Mode::Record => record_view(s),
    };
    // The files/options cards push the naming panel's Save button below the
    // glass even at the default 820 pt height. Naming is the only action that
    // can be taken at that moment, so give it the screen. Python clips this
    // panel; this is an intentional usability improvement.
    let focus_names = s.naming.is_some() && s.window_height <= 820.0;
    if !focus_names {
        main = main.child(content);
    }

    if let Some(activity) = &s.activity {
        main = main.child(activity_card(s, activity));
    }

    main = main.child(utility_row(s));

    if s.details_open {
        // Only the Files tab lets the log fill the leftover height. On Record
        // the record view owns that space; growing both made the record view
        // shrink under its own content and the utility row cover the Start
        // button.
        main = main.child(details_card(s, s.mode == Mode::Files));
    }

    // The page has no scrollbar at normal heights. On the 600–678 pt windows
    // the reference allows a last-resort page scroll so the utility row and a
    // tall result stay reachable; without it they are clipped below the glass.
    // The details panel on Record can also need more room than the window has.
    if s.window_height < 700.0 || (s.details_open && s.mode == Mode::Record) {
        main = main
            .min_h(0.0)
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .scroll_edge_fade(16.0);
    }

    let mut root = el().key("root").w_full().h_full().child(main);
    if let Some(caret) = rename_caret(s, cx) {
        root = root.child(caret);
    }
    if s.mode == Mode::Record && s.mic_menu_open && s.idle() {
        // A popup must not push the rest of the form down. Draw a dismiss
        // scrim and the menu over the page, anchored under the selector.
        root = root
            .child(
                el().key("mic-scrim")
                    .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
                    .on_press(|s: &mut AppState| s.mic_menu_open = false),
            )
            .child(mic_menu(s));
    }
    if s.privacy_open {
        root = root.child(privacy_overlay(s));
    }
    root
}

/// Rune paints a focused field's caret in one colour and does not blink it.
/// The rename box is the one place the app wants a word-processor caret, so
/// draw a slow-blink line over the field and leave Rune's own caret
/// transparent there. The rect comes from the focused field, in window points.
fn rename_caret(s: &AppState, cx: &Cx<AppState>) -> Option<El> {
    if !s.rename_visible || s.privacy_open || s.mic_menu_open {
        return None;
    }
    let [x, y, w, h] = cx.caret("rename-input")?;
    Some(
        el().key("rename-caret")
            .absolute(x, y)
            .size(w.max(1.0), h)
            .fill(s.theme.ink)
            .looping(rune::LoopSpec::cycle(rune::Cycle {
                period: 1.06,
                phase: 0.0,
                stops: &[(0.0, 1.0), (0.5, 1.0), (0.5, 0.0), (1.0, 0.0)],
                still: 1.0,
                fps: 30.0,
            })),
    )
}

/// First launch: the single executable is unpacking its runtime. This is a
/// standalone screen rather than a dialog so it can show real progress and can
/// never be buried behind another window.
fn preparing_view(s: &AppState, preparing: &PreparingState) -> El {
    let t = &s.theme;
    let percent = (preparing.progress * 100.0).round() as i32;

    let mut card = col()
        .key("preparing-card")
        .w(560.0)
        .radius(metric::RADIUS_LG)
        .fill(t.surface)
        .border(1.0, t.border)
        .shadow(14.0, 40.0, t.shadow)
        .padding(26.0)
        .gap(12.0)
        .child(
            row()
                .w_full()
                .center_y()
                .gap(12.0)
                .child(crate::widgets::logo(38.0))
                .child(text("Preparing EasyScribe").font(type_::NAMING_TITLE).color(t.ink).grow(1.0)),
        )
        .child(
            text(if preparing.error.is_some() {
                "EasyScribe could not unpack its program files."
            } else {
                "Unpacking the speech engine. This happens once; later launches start straight away."
            })
            .font(type_::BODY)
            .color(t.muted)
            .wrap(),
        );

    if let Some(error) = &preparing.error {
        card = card.child(message_banner(t, error, true)).child(
            row()
                .w_full()
                .justify_end()
                .gap(8.0)
                .pt(4.0)
                .child(
                    button(
                        t,
                        "Quit",
                        ButtonKind::Secondary,
                        100.0,
                        36.0,
                        type_::SMALL,
                        true,
                        |_s: &mut AppState| std::process::exit(1),
                    )
                    .key("prepare-quit"),
                )
                .child(
                    button(
                        t,
                        "Try again",
                        ButtonKind::Teal,
                        120.0,
                        36.0,
                        type_::SMALL,
                        true,
                        |s: &mut AppState| s.begin_prepare(),
                    )
                    .key("prepare-retry"),
                ),
        );
    } else {
        card = card
            .child(
                row()
                    .w_full()
                    .gap(12.0)
                    .center_y()
                    .child(
                        text(preparing.stage.clone())
                            .font(type_::STRONG)
                            .color(t.ink)
                            .grow(1.0),
                    )
                    .child(
                        text(format!("{percent}%"))
                            .font(type_::STRONG)
                            .color(t.muted),
                    ),
            )
            .child(progress_bar(
                t,
                Bar::Determinate(preparing.progress),
                t.teal,
            ))
            .child(
                text("Your recordings stay on this computer. EasyScribe never connects to the internet.")
                    .font(type_::SMALL)
                    .color(t.faint)
                    .wrap(),
            );
    }

    col()
        .key("preparing")
        .w_full()
        .h_full()
        .fill(t.bg)
        .px(metric::GUTTER)
        .child(header(s))
        .child(
            col()
                .w_full()
                .grow(1.0)
                .min_h(0.0)
                .center()
                .justify_center()
                .child(card),
        )
        .child(
            text("Works fully offline. No accounts, no telemetry, no update checks.")
                .font(type_::SMALL)
                .color(t.faint)
                .pb(18.0)
                .self_center(),
        )
}

fn header(s: &AppState) -> El {
    let t = &s.theme;
    row()
        .key("header")
        .w_full()
        .h(40.0)
        .pt(16.0)
        .pb(12.0)
        .center_y()
        .justify_between()
        .child(text("EasyScribe").font(type_::APP).color(t.ink))
        .child(crate::widgets::logo(40.0))
}

fn mode_switch(s: &AppState) -> El {
    let t = &s.theme;
    let enabled = s.idle();
    let mut container = row()
        .key("mode-switch")
        .h(38.0)
        .radius(metric::RADIUS_MD)
        .fill(t.surface_alt)
        .padding(4.0)
        .gap(4.0)
        .self_start();
    // `.pb` on the switch adds spacing under it while preserving the pill height.
    for (mode, label) in [(Mode::Files, "Transcribe files"), (Mode::Record, "Record")] {
        let selected = s.mode == mode;
        let (fill, hover) = match mode {
            Mode::Files => (t.teal, t.teal_hover),
            Mode::Record => (t.coral, t.coral_hover),
        };
        let mut segment = row()
            .key(if mode == Mode::Files {
                "mode-files"
            } else {
                "mode-record"
            })
            .role("button")
            .label(label)
            .h(30.0)
            .px(16.0)
            .radius(8.0)
            .center()
            .fill(if selected { fill } else { Color::TRANSPARENT })
            .focusable(enabled);
        if enabled {
            segment = segment
                .hover(if selected { hover } else { t.secondary_hover })
                .cursor(Cursor::Pointer)
                .focus_ring(2.0, 2.0, fill)
                .on_press(move |s: &mut AppState| s.switch_mode(mode));
        }
        let text_color = if selected {
            Color::WHITE
        } else if enabled {
            t.ink
        } else {
            t.faint
        };
        container = container.child(
            segment.child(
                text(label.to_owned())
                    .font(type_::STRONG)
                    .color(text_color)
                    .max_lines(1),
            ),
        );
    }
    col().w_full().pb(12.0).items_start().child(container)
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

fn files_view(s: &AppState) -> El {
    let _t = &s.theme;
    let busy = !s.idle();
    let mut files = col().key("files-view").w_full().child(files_add_card(s));

    if !s.files.is_empty() {
        files = files.child(expected_gap(12.0));
    }

    if s.activity.is_some() {
        files = files.child(options_folded(s));
    } else {
        files = files.child(files_options_card(s, busy));
        files = files.child(start_row(s));
    }

    files = files
        .accepts_files(easyscribe_core::outputs::is_supported_media)
        .on_drop_files(|s: &mut AppState, paths: &[std::path::PathBuf], _tx| {
            s.add_files(paths.to_vec());
        })
        .on_drop_rejected(|s: &mut AppState, paths: &[std::path::PathBuf], _tx| {
            s.add_files(paths.to_vec());
        })
        .on_drop_hover(|s: &mut AppState, over, _tx| {
            s.files_over = over && s.idle() && s.mode == Mode::Files;
        });

    files
}

fn expected_gap(amount: f32) -> El {
    el().w_full().h(amount)
}

fn files_add_card(s: &AppState) -> El {
    let t = &s.theme;
    let busy = !s.idle();
    let count = s.files.len();
    let mut header_right = row().gap(10.0).center_y();
    if count > 0 {
        header_right = header_right
            .child(
                text(easyscribe_core::outputs::plural(count, "file"))
                    .font(type_::BODY)
                    .color(t.muted),
            )
            .child(
                button(
                    t,
                    "Clear all",
                    ButtonKind::Link,
                    80.0,
                    28.0,
                    type_::SMALL,
                    !busy,
                    |s: &mut AppState| s.clear_files(),
                )
                .key("clear-all"),
            );
    }

    let mut card = card(t)
        .key("add-recordings")
        .w_full()
        .padding(18.0)
        .pt(16.0)
        .pb(16.0)
        .gap(10.0)
        .child(step_header(
            t,
            "1",
            "Add recordings",
            t.teal_soft,
            t.teal_ink,
            Some(header_right),
        ))
        .child(drop_zone(s));

    if !s.files.is_empty() {
        card = card.child(file_list(s));
    }

    if let Some(note) = &s.drop_note {
        card = card.child(
            text(note.clone())
                .font(type_::SMALL)
                .color(t.crimson)
                .wrap(),
        );
    }

    card
}

fn drop_zone(s: &AppState) -> El {
    let t = &s.theme;
    let busy = !s.idle();
    let populated = !s.files.is_empty();
    let over = s.files_over && !busy;
    let title = if populated {
        "Drag more files here"
    } else {
        "Drag audio or video files here"
    };
    let subtitle = if populated {
        "Or add more from your computer."
    } else {
        "Or choose them from your computer. MP3, WAV, M4A, MP4 and more."
    };
    let mut line = col().grow(1.0).gap(2.0);
    line = line.child(text(title.to_owned()).font(type_::STRONG).color(if over {
        t.teal_ink
    } else {
        t.ink
    }));
    if !populated {
        line = line.child(
            text(subtitle.to_owned())
                .font(type_::SMALL)
                .color(t.muted)
                .wrap(),
        );
    }

    let fill = if over { t.teal_soft } else { t.surface_alt };
    let border = if over { t.teal } else { t.border };
    let mut zone = row()
        .key("drop-zone")
        .w_full()
        .min_h(if populated { 52.0 } else { 64.0 })
        .radius(metric::RADIUS_MD)
        .fill(fill)
        .border(2.0, border)
        .padding(16.0)
        .py(if populated { 8.0 } else { 12.0 })
        .center_y()
        .gap(12.0)
        .child(line)
        .child(
            button(
                t,
                if populated {
                    "Add files"
                } else {
                    "Choose files"
                },
                ButtonKind::Teal,
                130.0,
                36.0,
                type_::STRONG,
                !busy,
                |s: &mut AppState| s.choose_files(),
            )
            .key(if populated {
                "add-files"
            } else {
                "choose-files"
            }),
        );
    zone = zone
        .on_drop_files(|s: &mut AppState, paths: &[std::path::PathBuf], _tx| {
            s.add_files(paths.to_vec());
        })
        .on_drop_rejected(|s: &mut AppState, paths: &[std::path::PathBuf], _tx| {
            s.add_files(paths.to_vec());
        });
    zone
}

fn file_list(s: &AppState) -> El {
    let t = &s.theme;
    let busy = !s.idle();
    let mut list = col().key("file-list").w_full().gap(0.0);
    // Before a batch starts every row would say "Waiting"; the activity card
    // is what marks a batch as running, so pills appear with it.
    let show_pills = s.activity.is_some();
    for (index, file) in s.files.iter().enumerate() {
        let path = &file.path;
        let name = file.display_name.clone().unwrap_or_else(|| {
            path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_owned()
        });
        let name = easyscribe_core::outputs::shorten(&name, 60);
        let size = file
            .display_size
            .or_else(|| std::fs::metadata(path).ok().map(|m| m.len()))
            .map(easyscribe_core::outputs::fmt_size)
            .unwrap_or_else(|| "Not found".into());
        let parent = path
            .parent()
            .map(|p| p.display().to_string())
            .unwrap_or_default();
        let parent = easyscribe_core::outputs::shorten(&parent, 60);
        let meta = format!("{size}  ·  {parent}");

        let mut line = col().grow(1.0).gap(2.0).child(
            text(name)
                .font(type_::BODY)
                .color(t.ink)
                .max_lines(1)
                .truncate_middle(),
        );
        line = line.child(
            text(meta)
                .font(type_::SMALL)
                .color(t.muted)
                .max_lines(1)
                .truncate_middle(),
        );

        let mut r = row()
            .key(format!("file-{index}"))
            .w_full()
            .h(46.0)
            .px(8.0)
            .py(3.0)
            .center_y()
            .gap(8.0)
            .child(line);

        if show_pills {
            r = r.child(status_pill(t, file.state));
        }
        r = r.child(button(
            t,
            "Remove",
            ButtonKind::LinkMuted,
            70.0,
            28.0,
            type_::SMALL,
            !busy,
            move |s: &mut AppState| s.remove_file(index),
        ));
        list = list.child(r);
    }
    // While the naming panel waits, keep the list shorter so its actions stay
    // reachable at the 820 pt window height (the extra row still scrolls).
    let rows = if s.naming.is_some() {
        s.visible_rows.clamp(1, 2)
    } else {
        s.visible_rows.max(1)
    };
    let visible = rows * 46;
    list.max_h(visible as f32).scroll(ScrollSpec {
        stick_bottom: false,
    })
}

fn files_options_card(s: &AppState, busy: bool) -> El {
    let t = &s.theme;
    let mut inner = col()
        .key("options-card")
        .w_full()
        .radius(metric::RADIUS_LG)
        .fill(t.surface)
        .border(1.0, t.border)
        .shadow(2.0, 10.0, t.shadow)
        .padding(18.0)
        .pt(16.0)
        .pb(16.0)
        .gap(10.0)
        .child(step_header(
            t,
            "2",
            "Choose options",
            t.teal_soft,
            t.teal_ink,
            None,
        ));

    inner = inner.child(field_label(t, "Save transcripts to"));
    inner = inner.child(files_output_row(s, busy));
    inner = inner.child(options_tiles(s, busy));
    inner
}

fn options_folded(s: &AppState) -> El {
    let t = &s.theme;
    col()
        .key("options-folded")
        .w_full()
        .radius(metric::RADIUS_LG)
        .fill(t.surface)
        .border(1.0, t.border)
        .shadow(2.0, 10.0, t.shadow)
        .padding(18.0)
        .py(14.0)
        .gap(10.0)
        .child(step_header(
            t,
            "2",
            "Choose options",
            t.teal_soft,
            t.teal_ink,
            Some(change_button(s)),
        ))
        .child(
            text(s.mode_summary_files())
                .font(type_::BODY)
                .color(t.muted)
                .pl(32.0)
                .wrap(),
        )
}

fn change_button(s: &AppState) -> El {
    let t = &s.theme;
    if s.activity.is_some() && s.idle() {
        button(
            t,
            "Change",
            ButtonKind::Link,
            80.0,
            28.0,
            type_::SMALL,
            true,
            |s: &mut AppState| s.hide_activity(),
        )
    } else {
        el().w(0.0).h(0.0)
    }
}

fn field_label(t: &Theme, value: &str) -> El {
    text(value.to_owned()).font(type_::STRONG).color(t.ink)
}

fn files_output_row(s: &AppState, busy: bool) -> El {
    let t = &s.theme;
    let custom = s.output_folder.is_some();
    let mut path_text = if let Some(folder) = &s.output_folder {
        let shown = easyscribe_core::outputs::shorten(&folder.display().to_string(), 70);
        text(shown).font(type_::BODY).color(t.ink)
    } else {
        text("The same folder as each recording")
            .font(type_::BODY)
            .color(t.muted)
    };
    path_text = path_text.grow(1.0).max_lines(1).truncate_middle();

    let mut r = row()
        .w_full()
        .min_h(32.0)
        .center_y()
        .gap(8.0)
        .child(path_text);
    if custom {
        r = r.child(button(
            t,
            "Use default",
            ButtonKind::Link,
            100.0,
            32.0,
            type_::SMALL,
            !busy,
            |s: &mut AppState| s.output_folder = None,
        ));
    }
    r.child(button(
        t,
        "Change folder",
        ButtonKind::Secondary,
        130.0,
        32.0,
        type_::SMALL,
        !busy,
        |s: &mut AppState| s.choose_output_folder(),
    ))
}

fn options_tiles(s: &AppState, busy: bool) -> El {
    let t = &s.theme;
    let timestamp_enabled = !busy;
    let speakers_enabled = !busy && s.diarization_available;

    let mut timestamp_tile = col()
        .grow(1.0)
        .radius(metric::RADIUS_MD)
        .fill(t.sky_soft)
        .padding(18.0)
        .py(6.0)
        .gap(4.0);
    let mut timestamp_label =
        text("Add timestamps")
            .font(type_::STRONG)
            .color(if timestamp_enabled {
                t.sky_ink
            } else {
                t.muted
            });
    if timestamp_enabled {
        timestamp_label = timestamp_label
            .cursor(Cursor::Pointer)
            .on_press(|s: &mut AppState| s.timestamps = !s.timestamps);
    }
    timestamp_tile = timestamp_tile.child(
        row()
            .key("timestamps-checkbox")
            .gap(12.0)
            .center_y()
            .child(
                checkbox(
                    t,
                    s.timestamps,
                    timestamp_enabled,
                    t.sky,
                    t.sky_hover,
                    |s: &mut AppState| s.timestamps = !s.timestamps,
                )
                .key("timestamp-toggle")
                .label("Add timestamps"),
            )
            .child(timestamp_label),
    );
    timestamp_tile = timestamp_tile.child(
        text("Shows the time, like [00:01:23], at the start of each paragraph.")
            .font(type_::SMALL)
            .color(t.muted)
            .wrap(),
    );

    let speaker_accent = if s.diarization_available {
        t.amber
    } else {
        t.border_strong
    };
    let mut speaker_tile = col()
        .grow(1.0)
        .radius(metric::RADIUS_MD)
        .fill(if s.diarization_available {
            t.amber_soft
        } else {
            t.surface_alt
        })
        .padding(18.0)
        .py(6.0)
        .gap(4.0);
    let mut speaker_label =
        text("Name the speakers")
            .font(type_::STRONG)
            .color(if s.diarization_available {
                t.amber_ink
            } else {
                t.muted
            });
    if speakers_enabled {
        speaker_label = speaker_label
            .cursor(Cursor::Pointer)
            .on_press(|s: &mut AppState| {
                if s.diarization_available {
                    s.name_speakers = !s.name_speakers;
                }
            });
    }
    speaker_tile = speaker_tile.child(
        row()
            .key("speakers-checkbox")
            .gap(12.0)
            .center_y()
            .child(
                checkbox(
                    t,
                    s.name_speakers && s.diarization_available,
                    speakers_enabled,
                    speaker_accent,
                    t.amber_hover,
                    |s: &mut AppState| {
                        if s.diarization_available {
                            s.name_speakers = !s.name_speakers;
                        }
                    },
                )
                .key("speaker-toggle")
                .label("Name the speakers"),
            )
            .child(speaker_label),
    );
    let body = if s.diarization_available {
        "Finds who is talking. You hear a short sample of each voice and type their name. Takes longer."
    } else {
        "Not available: the speaker files are missing from this copy of EasyScribe."
    };
    speaker_tile = speaker_tile.child(
        text(body.to_owned())
            .font(type_::SMALL)
            .color(t.muted)
            .wrap(),
    );

    row()
        .key("options")
        .w_full()
        .gap(12.0)
        .items_stretch()
        .child(timestamp_tile)
        .child(speaker_tile)
}

fn start_row(s: &AppState) -> El {
    let t = &s.theme;
    let count = s.files.len();
    let running = s.ui == UiState::Running;
    let label = if running {
        "Transcribing…".to_owned()
    } else if count == 0 {
        "Transcribe".to_owned()
    } else {
        format!(
            "Transcribe {}",
            easyscribe_core::outputs::plural(count, "file")
        )
    };
    let enabled = s.idle() && count > 0;
    let mut r = row()
        .key("start-row")
        .w_full()
        .pt(12.0)
        .gap(12.0)
        .center_y()
        .child(
            button(
                t,
                label,
                ButtonKind::Teal,
                metric::CTA_W,
                metric::CTA_H,
                type_::PRIMARY,
                enabled,
                |s: &mut AppState| s.start(),
            )
            .key("start-transcribe"),
        );
    if count == 0 {
        r = r.child(muted(t, "Add a recording to start."));
    } else if let Some((hint, error)) = &s.action_hint {
        r = r.child(text(hint.clone()).font(type_::BODY).color(if *error {
            t.crimson
        } else {
            t.muted
        }));
    }
    r
}

// ---------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------

fn record_view(s: &AppState) -> El {
    let t = &s.theme;
    let busy = !s.idle();
    let folded = s.activity.is_some();
    let live_mode = s.record_mode == RecordMode::Live;

    let mut record_card = col()
        .key("record-card")
        .w_full()
        .radius(metric::RADIUS_LG)
        .fill(t.surface)
        .border(1.0, t.border)
        .shadow(2.0, 10.0, t.shadow)
        .padding(18.0)
        .pt(16.0)
        .pb(16.0)
        .gap(10.0);

    if folded {
        record_card = record_card
            .child(step_header(
                t,
                "1",
                "Choose a microphone",
                t.coral_soft,
                t.coral_ink,
                None,
            ))
            .child(
                text(s.mic_summary())
                    .font(type_::BODY)
                    .color(t.muted)
                    .pl(32.0)
                    .wrap(),
            )
            .child(step_header(
                t,
                "2",
                "What do you want?",
                t.coral_soft,
                t.coral_ink,
                Some(change_button(s)),
            ))
            .child(
                text(s.record_mode_summary().to_owned())
                    .font(type_::BODY)
                    .color(t.muted)
                    .pl(32.0)
                    .wrap(),
            )
            .child(step_header(
                t,
                "3",
                "Record",
                t.coral_soft,
                t.coral_ink,
                None,
            ))
            .child(record_controls(s));
    } else {
        record_card = record_card
            .child(step_header(
                t,
                "1",
                "Choose a microphone",
                t.coral_soft,
                t.coral_ink,
                None,
            ))
            .child(mic_selector(s))
            .child(row().w_full().pt(6.0).gap(0.0).child(step_header(
                t,
                "2",
                "What do you want?",
                t.coral_soft,
                t.coral_ink,
                None,
            )))
            .child(record_tiles(s, busy))
            .child(field_label(t, "Save recordings to"))
            .child(record_folder_row(s, busy))
            .child(row().w_full().pt(6.0).gap(0.0).child(step_header(
                t,
                "3",
                "Record",
                t.coral_soft,
                t.coral_ink,
                None,
            )))
            .child(record_controls(s));
    }

    let mut column = col()
        .key("record-view")
        .w_full()
        .grow(1.0)
        .child(record_card);

    if live_mode {
        column = column.child(live_card(s));
    } else {
        // Parity: best-quality mode leaves the weighted space blank under the card.
        column = column.child(empty_spacer());
    }
    column
}

fn mic_selector(s: &AppState) -> El {
    let t = &s.theme;
    let enabled = s.idle();
    let selected = s
        .mic_choices
        .get(s.selected_mic)
        .map(|m| m.label.clone())
        .unwrap_or_else(|| "Default microphone".into());

    let mut selector = row()
        .key("mic-select")
        .w(340.0)
        .h(36.0)
        .radius(metric::RADIUS_MD)
        .fill(t.secondary)
        .px(12.0)
        .center_y()
        .gap(8.0)
        .role("combobox")
        .label("Microphone")
        .expanded(s.mic_menu_open)
        .focusable(enabled);
    if enabled {
        selector = selector
            .hover(t.secondary_hover)
            .cursor(Cursor::Pointer)
            .focus_ring(2.0, 2.0, t.coral)
            .on_press(|s: &mut AppState| s.mic_menu_open = !s.mic_menu_open)
            .on_key(|s: &mut AppState, key: &rune::KeyEvent, _tx| {
                if key.key == rune::Key::Escape && s.mic_menu_open {
                    s.mic_menu_open = false;
                    true
                } else {
                    false
                }
            });
    }
    selector
        .child(
            text(selected)
                .font(type_::BODY)
                .color(if enabled { t.ink } else { t.faint })
                .grow(1.0)
                .max_lines(1)
                .truncate_middle(),
        )
        .child(chevron(t, if enabled { t.ink } else { t.faint }))
}

fn mic_menu(s: &AppState) -> El {
    let t = &s.theme;
    let mut menu = col()
        .key("mic-menu")
        .w(340.0)
        .radius(metric::RADIUS_MD)
        .fill(t.surface)
        .border(1.0, t.border)
        .shadow(6.0, 18.0, t.shadow)
        .padding(4.0)
        .focus_group()
        .on_key(|s: &mut AppState, key: &rune::KeyEvent, _tx| {
            if key.key == rune::Key::Escape {
                s.mic_menu_open = false;
                true
            } else {
                false
            }
        })
        .anchored("mic-select", rune::Side::Bottom, rune::Align::Start, 4.0);
    for (index, choice) in s.mic_choices.iter().enumerate() {
        let label = easyscribe_core::outputs::shorten(&choice.label, 48);
        let selected = index == s.selected_mic;
        menu = menu.child(
            row()
                .key(format!("mic-{index}"))
                .h(32.0)
                .px(10.0)
                .radius(metric::RADIUS_SM)
                .center_y()
                .role("option")
                .label(label.clone())
                .focusable(true)
                .focus_selected(selected)
                .position_in_set(index + 1, s.mic_choices.len())
                .fill(if selected {
                    t.coral_soft
                } else {
                    Color::TRANSPARENT
                })
                .hover(t.coral_soft)
                .focus_ring(2.0, 1.0, t.coral)
                .cursor(Cursor::Pointer)
                .on_press(move |s: &mut AppState| {
                    s.selected_mic = index;
                    s.mic_menu_open = false;
                })
                .child(
                    text(label)
                        .font(type_::BODY)
                        .color(if selected { t.coral_ink } else { t.ink })
                        .max_lines(1)
                        .truncate_middle(),
                ),
        );
    }
    menu
}

fn chevron(_t: &Theme, color: Color) -> El {
    el().size(12.0, 12.0).shape(Shape::new().stroke(
        vec![[2.0, 4.0], [6.0, 8.0], [10.0, 4.0]],
        1.8,
        color,
    ))
}

fn record_tiles(s: &AppState, busy: bool) -> El {
    let t = &s.theme;
    let enabled = !busy;
    let mut tiles = row()
        .key("record-mode")
        .w_full()
        .gap(12.0)
        .items_stretch()
        .focus_group();

    let best_selected = s.record_mode == RecordMode::Best;
    let mut best = col()
        .key("record-best")
        .grow(1.0)
        .radius(metric::RADIUS_MD)
        .fill(if best_selected {
            t.coral_soft
        } else {
            t.surface_alt
        })
        .border(2.0, if best_selected { t.coral } else { t.border })
        .padding(18.0)
        .py(10.0)
        .gap(4.0);
    if enabled {
        best = best
            .cursor(Cursor::Pointer)
            .hover(t.coral_soft)
            .on_press(|s: &mut AppState| s.record_mode = RecordMode::Best);
    }
    best = best
        .role("radio")
        .label("Best quality transcript after recording")
        .focusable(enabled)
        .focus_selected(best_selected)
        .focus_ring(2.0, 2.0, t.coral)
        .child(
            row()
                .gap(12.0)
                .center_y()
                .child(radio(t, best_selected, enabled, t.coral))
                .child(
                    text("Best quality transcript after recording")
                        .font(type_::STRONG)
                        .color(if enabled { t.ink } else { t.muted }),
                ),
        )
        .child(
            col()
                .pl(32.0)
                .gap(1.0)
                .child(small_muted(t, "Saves the recording."))
                .child(small_muted(
                    t,
                    "Adds times. Most accurate. Find speakers after.",
                )),
        );

    let live_selected = s.record_mode == RecordMode::Live;
    let mut live = col()
        .key("record-live")
        .grow(1.0)
        .radius(metric::RADIUS_MD)
        .fill(if live_selected {
            t.coral_soft
        } else {
            t.surface_alt
        })
        .border(2.0, if live_selected { t.coral } else { t.border })
        .padding(18.0)
        .py(10.0)
        .gap(4.0);
    if enabled {
        live = live
            .cursor(Cursor::Pointer)
            .hover(t.coral_soft)
            .on_press(|s: &mut AppState| s.record_mode = RecordMode::Live);
    }
    live = live
        .role("radio")
        .label("Show words as I speak")
        .focusable(enabled)
        .focus_selected(live_selected)
        .focus_ring(2.0, 2.0, t.coral)
        .child(
            row()
                .gap(12.0)
                .center_y()
                .child(radio(t, live_selected, enabled, t.coral))
                .child(
                    text("Show words as I speak")
                        .font(type_::STRONG)
                        .color(if enabled { t.ink } else { t.muted }),
                ),
        )
        .child(
            col()
                .pl(32.0)
                .gap(1.0)
                .child(small_muted(t, "Saves the recording."))
                .child(small_muted(t, "Quick text while you talk. Less accurate.")),
        );

    tiles = tiles.child(best).child(live);
    tiles
}

fn record_folder_row(s: &AppState, busy: bool) -> El {
    let t = &s.theme;
    let custom = s.recordings_dir.is_some();
    let folder = s.effective_recordings_dir();
    let shown = easyscribe_core::outputs::shorten(&folder.display().to_string(), 70);
    let mut r = row().w_full().min_h(32.0).center_y().gap(8.0).child(
        text(shown)
            .font(type_::BODY)
            .color(if custom { t.ink } else { t.muted })
            .grow(1.0)
            .max_lines(1)
            .truncate_middle(),
    );
    if custom {
        r = r.child(button(
            t,
            "Use default",
            ButtonKind::Link,
            100.0,
            32.0,
            type_::SMALL,
            !busy,
            |s: &mut AppState| s.reset_recordings_folder(),
        ));
    }
    r.child(button(
        t,
        "Change folder",
        ButtonKind::Secondary,
        130.0,
        32.0,
        type_::SMALL,
        !busy,
        |s: &mut AppState| s.choose_recordings_folder(),
    ))
}

fn record_controls(s: &AppState) -> El {
    let t = &s.theme;
    let recording = s.ui == UiState::Recording;
    let enabled = s.idle() || (recording && !s.record_saving);
    let label = if s.record_saving {
        "Saving…".to_owned()
    } else if recording {
        "Stop recording".to_owned()
    } else {
        "Start recording".to_owned()
    };
    // Keep the timer coral while the recording is being saved: the stop
    // request has not finished yet, and the timer is still meaningful.
    let timer_color = if s.record_paused {
        t.amber_ink
    } else if recording {
        t.coral_ink
    } else {
        t.faint
    };
    let level = if recording && !s.record_paused {
        s.level.clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut meter = row()
        .w(120.0)
        .h(8.0)
        .radius(4.0)
        .fill(t.border)
        .clip_rounded(4.0);
    if recording && level > 0.0 {
        meter = meter.child(
            el().w((120.0 * level).max(1.0))
                .h_full()
                .radius(4.0)
                .fill(t.coral),
        );
    }

    let mut controls = row().w_full().gap(16.0).center_y().child(
        button(
            t,
            label,
            ButtonKind::Coral,
            metric::CTA_W,
            metric::CTA_H,
            type_::PRIMARY,
            enabled,
            |s: &mut AppState| s.toggle_recording(),
        )
        .key("record-button"),
    );
    if recording && s.record_capturing && !s.record_saving {
        controls = controls.child(
            button(
                t,
                if s.record_paused { "Resume" } else { "Pause" },
                ButtonKind::Secondary,
                110.0,
                metric::CTA_H,
                type_::STRONG,
                true,
                |s: &mut AppState| s.toggle_pause(),
            )
            .key("pause-button"),
        );
    }
    controls
        .child(
            text(easyscribe_core::outputs::fmt_timer(s.timer_seconds))
                .font(type_::TIMER)
                .color(timer_color),
        )
        .child(meter)
}

fn live_card(s: &AppState) -> El {
    let t = &s.theme;
    let color = if s.live_has_text { t.ink } else { t.muted };
    col()
        .key("live-card")
        .w_full()
        .grow(1.0)
        .min_h(0.0)
        .mt(12.0)
        .radius(metric::RADIUS_LG)
        .fill(t.surface)
        .border(1.0, t.border)
        .shadow(2.0, 10.0, t.shadow)
        .padding(18.0)
        .pt(14.0)
        .pb(16.0)
        .gap(8.0)
        .child(text("Live transcript").font(type_::TITLE).color(t.ink))
        .child(read_text(
            "live-text",
            s.live_text.clone(),
            type_::BODY,
            color,
            t.surface_alt,
            14.0,
            120.0,
            true,
        ))
}

// ---------------------------------------------------------------------------
// Activity and results
// ---------------------------------------------------------------------------

fn activity_card(s: &AppState, activity: &Activity) -> El {
    let t = &s.theme;
    let mut inner = col()
        .key("activity-card")
        .w_full()
        .mt(12.0)
        .radius(metric::RADIUS_LG)
        .fill(t.surface)
        .border(1.0, t.border)
        .shadow(2.0, 10.0, t.shadow)
        .padding(18.0)
        .pt(14.0)
        .pb(14.0)
        .gap(8.0);

    let mut top = row().w_full().center_y().gap(12.0).child(
        text(activity.title.clone())
            .font(type_::STATUS)
            .color(activity.tone.color(t))
            .grow(1.0)
            .wrap(),
    );

    if s.ui == UiState::Running && activity.bar == BarMode::Determinate {
        top = top.child(
            text(format!(
                "{}%",
                (activity.progress * 100.0).round_ties_even() as i32
            ))
            .font(type_::STRONG)
            .color(t.muted),
        );
    }
    if activity.stop_visible {
        top = top.child(button(
            t,
            "Stop",
            ButtonKind::Secondary,
            80.0,
            32.0,
            type_::SMALL,
            activity.stop_enabled,
            |s: &mut AppState| s.cancel_job(),
        ));
    }
    inner = inner.child(top);

    if !activity.detail.is_empty() {
        inner = inner.child(
            text(activity.detail.clone())
                .font(type_::BODY)
                .color(t.muted)
                .wrap(),
        );
    }

    let bar = match activity.bar {
        BarMode::Idle => Bar::Idle,
        BarMode::Determinate => Bar::Determinate(activity.progress),
        BarMode::Busy => Bar::Busy,
    };
    inner = inner.child(progress_bar(t, bar, activity.bar_color));

    if let Some((message, kind)) = &activity.message {
        // While naming speakers on a short window, the panel repeats this
        // instruction and every pixel keeps another row reachable.
        let duplicate = s.naming.is_some() && s.window_height < 900.0;
        if !duplicate {
            inner = inner.child(message_banner(t, message, *kind == MessageKind::Error));
        }
    }

    let has_results = activity.open_transcript || activity.open_folder || activity.try_again;
    if has_results {
        let mut results = row().w_full().gap(8.0).pt(2.0);
        if activity.open_transcript {
            results = results.child(button(
                t,
                "Open transcript",
                ButtonKind::Teal,
                150.0,
                36.0,
                type_::SMALL,
                true,
                |s: &mut AppState| s.open_transcript(),
            ));
        }
        if activity.open_folder {
            results = results.child(button(
                t,
                "Open folder",
                ButtonKind::Secondary,
                120.0,
                36.0,
                type_::SMALL,
                true,
                |s: &mut AppState| s.open_folder(),
            ));
        }
        if activity.try_again {
            results = results.child(button(
                t,
                "Try again",
                ButtonKind::Secondary,
                110.0,
                36.0,
                type_::SMALL,
                true,
                |s: &mut AppState| s.hide_activity(),
            ));
        }
        inner = inner.child(results);
    }

    if s.rename_visible {
        inner = inner.child(rename_row(s));
    }
    if activity.best_row {
        inner = inner.child(best_row(s));
    }
    if let Some(find) = &s.find_card {
        inner = inner.child(find_card(s, find));
    }
    if let Some(naming) = &s.naming {
        inner = inner.child(naming_panel(s, naming));
    }

    inner
}

fn rename_row(s: &AppState) -> El {
    let t = &s.theme;
    let enabled = s.idle();
    col()
        .key("rename-row")
        .w_full()
        .radius(metric::RADIUS_MD)
        .fill(t.surface_alt)
        .padding(12.0)
        .px(14.0)
        .gap(6.0)
        .child(
            row()
                .w_full()
                .gap(10.0)
                .center_y()
                .child(text("Name").font(type_::STRONG).color(t.ink))
                .child(
                    row()
                        .grow(1.0)
                        .h(36.0)
                        .radius(metric::RADIUS_SM)
                        .fill(t.surface)
                        .border(1.0, t.border_strong)
                        .px(10.0)
                        .center_y()
                        .child(
                            text_input(&s.rename_edit, type_::BODY)
                                .key("rename-input")
                                .label("New name")
                                .w_full()
                                .single_line()
                                .color(t.ink)
                                // `rename_caret` draws the blinking line; hide
                                // Rune's solid white default on the light field.
                                .caret(Color::TRANSPARENT, t.coral_soft)
                                .placeholder(
                                    "Type a new name, for example Team meeting 21 Oct",
                                    t.faint,
                                )
                                .on_edit(|s: &mut AppState| &mut s.rename_edit)
                                .on_submit(|s: &mut AppState, _tx| s.commit_rename()),
                        ),
                )
                .child(button(
                    t,
                    "Rename",
                    ButtonKind::Teal,
                    100.0,
                    36.0,
                    type_::STRONG,
                    enabled,
                    |s: &mut AppState| s.commit_rename(),
                )),
        )
        .child(
            text(s.rename_note.clone())
                .font(type_::SMALL)
                .color(s.rename_note_color())
                .wrap(),
        )
}

fn best_row(s: &AppState) -> El {
    let t = &s.theme;
    let note = if s.live_has_text {
        "Adds times. More accurate. Your live transcript is kept. You can find the speakers after."
    } else {
        "Adds times. Most accurate. You can find the speakers after."
    };
    row()
        .key("best-row")
        .w_full()
        .radius(metric::RADIUS_MD)
        .fill(t.surface_alt)
        .padding(12.0)
        .px(14.0)
        .center_y()
        .gap(12.0)
        .child(button(
            t,
            "Make best quality transcript",
            ButtonKind::Teal,
            240.0,
            36.0,
            type_::STRONG,
            s.idle(),
            |s: &mut AppState| {
                if let Some(audio) = s.last_recording.clone() {
                    s.start_best_quality(audio);
                }
            },
        ))
        .child(
            text(note.to_owned())
                .font(type_::SMALL)
                .color(t.muted)
                .grow(1.0)
                .wrap(),
        )
}

fn find_card(s: &AppState, find: &FindCard) -> El {
    let t = &s.theme;
    let mut card = col()
        .key("find-card")
        .w_full()
        .mt(4.0)
        .radius(metric::RADIUS_MD)
        .fill(t.surface)
        .border(2.0, t.amber)
        .padding(16.0)
        .gap(6.0)
        .child(text("Who said what?").font(type_::TITLE).color(t.ink))
        .child(
            text("Find the different voices in this recording, then type each person's name. The names are added to a copy of the transcript.")
                .font(type_::BODY)
                .color(t.muted)
                .wrap(),
        );

    let mut bottom = row()
        .w_full()
        .pt(4.0)
        .gap(16.0)
        .center_y()
        .child(
            col()
                .grow(1.0)
                .gap(4.0)
                .child(
                    row()
                        .gap(8.0)
                        .center_y()
                        .child(
                            text(find.estimate.clone())
                                .font(type_::STRONG)
                                .color(t.amber_ink),
                        )
                        .child(info_pill(t, "rough estimate", t.amber_ink, t.amber_soft)),
                )
                .child(
                    text(find.basis.clone())
                        .font(type_::SMALL)
                        .color(t.muted)
                        .wrap(),
                ),
        )
        .child(button(
            t,
            "Find speakers",
            ButtonKind::Amber,
            170.0,
            40.0,
            type_::STRONG,
            s.idle(),
            |s: &mut AppState| s.start_find_speakers(),
        ));
    bottom = bottom.child(el().w(0.0));
    card = card.child(bottom);
    card
}

fn naming_panel(s: &AppState, naming: &NamingState) -> El {
    let t = &s.theme;
    let short_window = s.window_height < 900.0;
    let row_height = if short_window { 50.0 } else { 62.0 };
    let count = naming.rows.len();
    let title = format!(
        "We found {count} speaker{}",
        if count == 1 { "" } else { "s" }
    );
    let mut panel = col()
        .key("naming-panel")
        .w_full()
        .mt(6.0)
        .radius(metric::RADIUS_MD)
        .fill(t.surface)
        .border(2.0, t.amber)
        .padding(18.0)
        .pt(16.0)
        .pb(18.0)
        .gap(7.0)
        .pb(12.0)
        .child(text(title).font(type_::NAMING_TITLE).color(t.ink))
        .child(
            text("Play a short sample of each voice, then type the person's name.\nLeave a box empty to keep \"Speaker 1\", \"Speaker 2\" and so on.")
                .font(type_::BODY)
                .color(t.muted)
                .wrap(),
        );

    let mut rows = col().w_full().gap(4.0);
    for (index, naming_row) in naming.rows.iter().enumerate() {
        let clip = naming_row.clip.clone();
        let has_clip = clip.is_some();
        let share = naming_row
            .share
            .map(|share| {
                format!(
                    "{}% of the talking",
                    (share * 100.0).round_ties_even() as i32
                )
            })
            .unwrap_or_default();
        let mut r = row()
            .key(format!("speaker-{index}"))
            .w_full()
            .h(row_height)
            .px(8.0)
            .gap(10.0)
            .center_y()
            .radius(metric::RADIUS_MD)
            .fill(t.surface_alt)
            .border(1.0, t.border)
            .child(
                row()
                    .w(92.0)
                    .h(28.0)
                    .radius(metric::RADIUS_SM)
                    .fill(t.amber_soft)
                    .center()
                    .child(
                        text(naming_row.default_name.clone())
                            .font(type_::STRONG)
                            .color(t.amber_ink),
                    ),
            )
            .child(button(
                t,
                if has_clip { "Play sample" } else { "No sample" },
                ButtonKind::Secondary,
                110.0,
                32.0,
                type_::SMALL,
                has_clip,
                move |s: &mut AppState| s.play_clip(clip.as_ref()),
            ))
            .child(
                row()
                    .grow(1.0)
                    .h(34.0)
                    .radius(metric::RADIUS_SM)
                    .fill(t.surface)
                    .border(1.0, t.border_strong)
                    .px(10.0)
                    .center_y()
                    .child(
                        text_input(&naming_row.edit, type_::BODY)
                            .key(format!("speaker-name-{index}"))
                            .label(format!("Name for {}", naming_row.default_name))
                            .w_full()
                            .single_line()
                            .color(t.ink)
                            .caret(t.ink, t.coral_soft)
                            .placeholder("Type a name", t.faint)
                            .on_edit(move |s: &mut AppState| {
                                &mut s.naming.as_mut().expect("naming panel").rows[index].edit
                            })
                            .on_submit(|s: &mut AppState, _tx| s.save_naming()),
                    ),
            );
        if !share.is_empty() {
            r = r.child(
                text(share)
                    .font(type_::SMALL)
                    .color(t.muted)
                    .w(130.0)
                    .max_lines(1),
            );
        }
        rows = rows.child(r);
    }
    if short_window {
        // On an 820 pt window the full panel would put Save/Skip off-screen.
        // Three compact rows fit, so a typical conversation shows every voice;
        // a larger cast scrolls inside the same panel.
        rows = rows
            .max_h(row_height * 3.0 + 8.0)
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .scroll_edge_fade(14.0);
    } else if naming.rows.len() > 4 {
        rows = rows.max_h(248.0).scroll(ScrollSpec {
            stick_bottom: false,
        });
    }
    panel = panel.child(rows);
    panel.child(
        row()
            .w_full()
            .justify_end()
            .gap(8.0)
            .pt(4.0)
            .child(
                button(
                    t,
                    "Skip names",
                    ButtonKind::Subtle,
                    110.0,
                    36.0,
                    type_::SMALL,
                    true,
                    |s: &mut AppState| s.skip_naming(),
                )
                .key("skip-names"),
            )
            .child(
                button(
                    t,
                    "Save transcript with names",
                    ButtonKind::Amber,
                    230.0,
                    36.0,
                    type_::STRONG,
                    true,
                    |s: &mut AppState| s.save_naming(),
                )
                .key("save-names"),
            ),
    )
}

// ---------------------------------------------------------------------------
// Details, utility and privacy
// ---------------------------------------------------------------------------

fn utility_row(s: &AppState) -> El {
    let t = &s.theme;
    row()
        .key("utility")
        .w_full()
        .py(12.0)
        .center_y()
        .justify_between()
        .child(
            button(
                t,
                if s.details_open {
                    "Hide details"
                } else {
                    "Show details"
                },
                ButtonKind::Subtle,
                110.0,
                30.0,
                type_::SMALL,
                true,
                |s: &mut AppState| s.details_open = !s.details_open,
            )
            .key("toggle-details"),
        )
        .child(
            button(
                t,
                "Privacy",
                ButtonKind::Subtle,
                80.0,
                30.0,
                type_::SMALL,
                true,
                |s: &mut AppState| s.privacy_open = true,
            )
            .key("open-privacy"),
        )
}

fn details_card(s: &AppState, grow: bool) -> El {
    let t = &s.theme;
    let body = if s.log.is_empty() {
        String::new()
    } else {
        s.log.join("\n")
    };
    let mut card = col()
        .key("details-card")
        .w_full()
        .min_h(140.0)
        .pb(16.0)
        .radius(metric::RADIUS_LG)
        .fill(t.surface)
        .border(1.0, t.border)
        .shadow(2.0, 10.0, t.shadow)
        .padding(10.0);
    if grow {
        card = card.grow(1.0);
    }
    card.child(read_text(
        "details-log",
        body,
        type_::MONO,
        t.muted,
        Color::TRANSPARENT,
        0.0,
        140.0,
        true,
    ))
}

fn privacy_overlay(s: &AppState) -> El {
    let t = &s.theme;
    let logs = s.paths.logs_dir.display().to_string();
    let recordings = s.paths.default_recordings_dir.display().to_string();

    let mut rows = col().w_full().gap(10.0);
    for (title, body) in [
        (
            "Transcripts",
            "Next to each recording, or in the folder you choose.".to_owned(),
        ),
        (
            "Recordings",
            format!("{recordings}, or the folder you choose."),
        ),
        (
            "Temporary audio",
            "Deleted after each file, and again when EasyScribe closes.".to_owned(),
        ),
        (
            "Technical logs",
            format!(
                "{logs}\nFile names and progress only, never the transcript text. Only the last 10 are kept."
            ),
        ),
    ] {
        rows = rows.child(
            row()
                .w_full()
                .gap(16.0)
                .items_start()
                .child(
                    text(title.to_owned())
                        .font(type_::STRONG)
                        .color(t.ink)
                        .w(150.0),
                )
                .child(text(body).font(type_::BODY).color(t.muted).grow(1.0).wrap()),
        );
    }

    let card_body = col()
        .key("privacy-card")
        .w(620.0)
        .radius(metric::RADIUS_LG)
        .fill(t.bg)
        .border(1.0, t.border)
        .shadow(18.0, 48.0, Color::BLACK.alpha(0.18))
        .padding(24.0)
        .pt(22.0)
        .pb(22.0)
        .gap(14.0)
        .child(text("Your privacy").font(type_::PRIVACY_TITLE).color(t.ink))
        .child(
            col()
                .w_full()
                .radius(metric::RADIUS_MD)
                .fill(t.green_soft)
                .padding(10.0)
                .px(12.0)
                .child(
                    text("EasyScribe works fully offline. It never connects to the internet,\nand it blocks any attempt to do so. Nothing is uploaded, shared\nor tracked. There are no accounts, analytics or update checks.")
                        .font(type_::BODY)
                        .color(t.green_ink)
                        .wrap(),
                ),
        )
        .child(text("Where your data is saved").font(type_::STRONG).color(t.ink))
        .child(
            col()
                .w_full()
                .radius(metric::RADIUS_LG)
                .fill(t.surface)
                .border(1.0, t.border)
                .padding(16.0)
                .child(rows),
        )
        .child(
            text("To remove everything, delete the transcripts you made and the\nEasyScribe folder. Nothing is stored anywhere else.")
                .font(type_::BODY)
                .color(t.muted)
                .wrap(),
        )
        .child(
            row()
                .w_full()
                .justify_end()
                .gap(8.0)
                .pt(4.0)
                .child(button(
                    t,
                    "Open logs folder",
                    ButtonKind::Secondary,
                    150.0,
                    36.0,
                    type_::SMALL,
                    true,
                    |s: &mut AppState| s.open_logs_folder(),
                ))
                .child(button(
                    t,
                    "Close",
                    ButtonKind::Teal,
                    100.0,
                    36.0,
                    type_::SMALL,
                    true,
                    |s: &mut AppState| s.privacy_open = false,
                ).key("close-privacy")),
        );

    el().key("privacy-scrim")
        .pin(Some(0.0), Some(0.0), Some(0.0), Some(0.0))
        .fill(t.scrim)
        .center()
        .modal(true, |s: &mut AppState| s.privacy_open = false)
        .on_press(|s: &mut AppState| s.privacy_open = false)
        .child(col().on_press(|_: &mut AppState| {}).child(card_body))
}

#[allow(dead_code)]
fn _unused(t: &Theme) {
    let _ = (
        FileState::Waiting,
        label(t, ""),
        link_text(t, ""),
        strong(t, ""),
        empty_spacer(),
    );
}
