//! Reusable controls painted in the EasyScribe design language.

use rune::{
    Color, Cursor, El, ScrollSpec, Shape, Tx, col, el, row, text,
};
use rune::LoopSpec;

use crate::app::AppState;
use crate::theme::{Theme, metric, type_};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ButtonKind {
    Teal,
    Coral,
    Amber,
    Secondary,
    Subtle,
    Link,
    LinkMuted,
    White,
}

impl ButtonKind {
    fn colors(self, t: &Theme) -> (Color, Color, Color, bool) {
        // fill, hover, text, has_border
        match self {
            Self::Teal => (t.teal, t.teal_hover, Color::WHITE, false),
            Self::Coral => (t.coral, t.coral_hover, Color::WHITE, false),
            Self::Amber => (t.amber, t.amber_hover, Color::WHITE, false),
            Self::Secondary => (t.secondary, t.secondary_hover, t.ink, false),
            Self::Subtle => (t.surface, t.surface_alt, t.muted, true),
            Self::Link => (Color::TRANSPARENT, t.surface_alt, t.teal_ink, false),
            Self::LinkMuted => (Color::TRANSPARENT, t.surface_alt, t.muted, false),
            Self::White => (t.surface, t.surface_alt, t.ink, true),
        }
    }

    fn ring(self, t: &Theme) -> Color {
        match self {
            Self::Teal => t.teal,
            Self::Coral => t.coral,
            Self::Amber => t.amber,
            Self::Secondary | Self::Subtle | Self::White | Self::Link | Self::LinkMuted => t.teal,
        }
    }
}

pub fn button<S: 'static>(
    t: &Theme,
    label: impl Into<String>,
    kind: ButtonKind,
    width: f32,
    height: f32,
    font: rune::tok::TextStyle,
    enabled: bool,
    on_press: impl Fn(&mut S) + 'static,
) -> El {
    let label = label.into();
    let (fill, hover, text_color, bordered) = kind.colors(t);
    let mut b = row()
        .role("button")
        .label(label.clone())
        .h(height)
        .center()
        .justify_center()
        .radius(metric::RADIUS_MD)
        .fill(if enabled { fill } else { t.secondary })
        .cursor(if enabled { Cursor::Pointer } else { Cursor::Default })
        .focusable(enabled);
    if width > 0.0 {
        b = b.w(width);
    }
    if bordered {
        b = b.border(1.0, if enabled { t.border } else { t.border });
    }
    if enabled {
        b = b
            .hover(hover)
            .press(Color::BLACK.alpha(0.05))
            .focus_ring(2.0, 2.0, kind.ring(t))
            .on_press(on_press);
    }
    b.child(
        text(label)
            .font(font)
            .color(if enabled { text_color } else { t.faint })
            .max_lines(1)
            .truncate(),
    )
}

pub fn card(t: &Theme) -> El {
    col()
        .fill(t.surface)
        .radius(metric::RADIUS_LG)
        .border(1.0, t.border)
        .shadow(2.0, 10.0, t.shadow)
}

pub fn step_badge(_t: &Theme, number: &str, soft: Color, ink: Color) -> El {
    col()
        .size(metric::STEP, metric::STEP)
        .radius(metric::STEP / 2.0)
        .fill(soft)
        .center()
        .child(text(number.to_owned()).font(type_::STRONG).color(ink))
}

pub fn step_header(t: &Theme, number: &str, title: &str, soft: Color, ink: Color, right: Option<El>) -> El {
    let mut r = row()
        .w_full()
        .center_y()
        .gap(10.0)
        .child(step_badge(t, number, soft, ink))
        .child(text(title.to_owned()).font(type_::TITLE).color(t.ink));
    if let Some(right) = right {
        r = r.child(el().grow(1.0)).child(right);
    }
    r
}

pub fn label(t: &Theme, value: impl Into<String>) -> El {
    text(value.into()).font(type_::BODY).color(t.ink)
}

pub fn strong(t: &Theme, value: impl Into<String>) -> El {
    text(value.into()).font(type_::STRONG).color(t.ink)
}

pub fn muted(t: &Theme, value: impl Into<String>) -> El {
    text(value.into()).font(type_::BODY).color(t.muted)
}

pub fn small_muted(t: &Theme, value: impl Into<String>) -> El {
    text(value.into()).font(type_::SMALL).color(t.muted)
}

pub fn small_ink(t: &Theme, value: impl Into<String>) -> El {
    text(value.into()).font(type_::SMALL).color(t.ink)
}

pub fn link_text(t: &Theme, value: impl Into<String>) -> El {
    text(value.into()).font(type_::SMALL).color(t.teal_ink)
}

pub fn checkbox(
    t: &Theme,
    checked: bool,
    enabled: bool,
    accent: Color,
    accent_hover: Color,
    on_toggle: impl Fn(&mut AppState) + 'static,
) -> El {
    let fill = if checked { accent } else { t.surface };
    let mut box_el = el()
        .role("checkbox")
        .checked(checked)
        .size(22.0, 22.0)
        .radius(metric::RADIUS_SM)
        .fill(if enabled { fill } else { t.surface_alt })
        .border(if checked { 0.0 } else { 2.0 }, if enabled { accent } else { t.border })
        .focusable(enabled);
    if checked && enabled {
        box_el = box_el.shape(
            Shape::new().stroke(vec![[5.0, 11.0], [9.5, 15.5], [17.0, 6.0]], 2.4, Color::WHITE),
        );
    }
    if enabled {
        box_el = box_el
            .hover(if checked { accent_hover } else { accent.alpha(0.10) })
            .press(Color::BLACK.alpha(0.05))
            .focus_ring(2.0, 2.0, accent)
            .cursor(Cursor::Pointer)
            .on_press(move |s: &mut AppState| on_toggle(s));
    }
    box_el
}

pub fn radio(t: &Theme, selected: bool, enabled: bool, accent: Color) -> El {
    let mut circle = el()
        .size(20.0, 20.0)
        .radius(10.0)
        .fill(t.surface)
        .border(2.0, if enabled { if selected { accent } else { t.border_strong } } else { t.border });
    if selected {
        circle = circle.center().child(el().size(10.0, 10.0).radius(5.0).fill(if enabled { accent } else { t.faint }));
    }
    circle
}

pub fn status_pill(t: &Theme, state: crate::app::FileState) -> El {
    let (label, fg, bg) = state.pill(t);
    row()
        .size(72.0, 24.0)
        .radius(metric::RADIUS_SM)
        .fill(bg)
        .center()
        .justify_center()
        .child(text(label).font(type_::SMALL).color(fg))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bar {
    Idle,
    Determinate(f32),
    Busy,
}

pub fn progress_bar(t: &Theme, bar: Bar, color: Color) -> El {
    let frac = match bar {
        Bar::Determinate(value) => Some(value.clamp(0.0, 1.0)),
        Bar::Idle => Some(0.0),
        Bar::Busy => None,
    };
    let mut track = row()
        .w_full()
        .h(8.0)
        .radius(4.0)
        .fill(t.border)
        .clip_rounded(4.0);
    if let Some(frac) = frac {
        let mut fill = el().h_full().radius(4.0).fill(color);
        if frac > 0.0 {
            fill = fill.w_frac(frac);
        } else {
            fill = fill.w(0.0);
        }
        track = track.child(fill);
    } else {
        track = track.child(
            el()
                .w(220.0)
                .h_full()
                .radius(4.0)
                .fill(color)
                .looping(LoopSpec::slide(1.25, [-240.0, 0.0], [820.0, 0.0])),
        );
    }
    track
}

pub fn message_banner(t: &Theme, text_value: &str, error: bool) -> El {
    col()
        .w_full()
        .radius(metric::RADIUS_SM)
        .fill(if error { t.crimson_soft } else { t.surface_alt })
        .padding(8.0)
        .px(10.0)
        .child(
            text(text_value.to_owned())
                .font(type_::BODY)
                .color(if error { t.crimson } else { t.ink })
                .wrap(),
        )
}

pub fn info_pill(_t: &Theme, value: &str, fg: Color, bg: Color) -> El {
    row()
        .h(22.0)
        .px(6.0)
        .radius(metric::RADIUS_SM)
        .fill(bg)
        .center()
        .child(text(value.to_owned()).font(type_::SMALL).color(fg))
}

pub fn logo(size: f32) -> El {
    let scale = size / 40.0;
    let bar_w = 3.6 * scale;
    let gap = 2.8 * scale;
    let left = (size - (bar_w * 5.0 + gap * 4.0)) / 2.0;
    let heights = [8.8, 18.4, 24.8, 15.2, 7.2];
    let mut tile = el()
        .key("logo")
        .size(size, size)
        .radius(size * 0.28)
        .fill(crate::theme::hx("#0B7A75"))
        .clip_rounded(size * 0.28);
    for (i, height) in heights.iter().enumerate() {
        let x = left + i as f32 * (bar_w + gap);
        let y = (size - height * scale) / 2.0;
        tile = tile.child(
            el()
                .w(bar_w)
                .h(height * scale)
                .radius(bar_w / 2.0)
                .fill(Color::WHITE)
                .absolute(x, y),
        );
    }
    tile
}

pub fn chevron(t: &Theme) -> El {
    el().size(12.0, 12.0).shape(
        Shape::new().stroke(vec![[2.0, 4.0], [6.0, 8.0], [10.0, 4.0]], 1.8, t.ink),
    )
}

/// A scrollable, selectable read-only text block. `stick_bottom` follows appended
/// text like the live transcript and details log.
pub fn read_text(
    key: &'static str,
    value: String,
    font: rune::tok::TextStyle,
    color: Color,
    fill: Color,
    padding: f32,
    min_height: f32,
    stick_bottom: bool,
) -> El {
    el()
        .key(key)
        .w_full()
        .grow(1.0)
        .min_h(min_height)
        .scroll(ScrollSpec { stick_bottom })
        .radius(if fill == Color::TRANSPARENT { 0.0 } else { metric::RADIUS_MD })
        .fill(fill)
        .padding(padding)
        .child(
            text(value)
                .font(font)
                .color(color)
                .wrap()
                .selectable(true)
                .selection_group(true),
        )
}

/// A soft feature tile that can be pressed as a whole (record choices).
pub fn pressable_tile(
    fill: Color,
    border: Color,
    on_press: impl Fn(&mut AppState) + 'static,
) -> El {
    row()
        .w_full()
        .grow(1.0)
        .radius(metric::RADIUS_MD)
        .fill(fill)
        .border(2.0, border)
        .padding(12.0)
        .gap(10.0)
        .cursor(Cursor::Pointer)
        .focusable(true)
        .press(Color::BLACK.alpha(0.035))
        .on_press(move |s: &mut AppState| on_press(s))
}

pub fn empty_spacer() -> El {
    el().grow(1.0)
}
