//! The v3.2.3 palette, type scale and shared metrics, translated to Rune.
//!
//! Every value here comes from `REFERENCE_UI.md` §3. The light theme is fixed:
//! the product has no dark mode, so no OS appearance is consulted.

use rune::tok::{Family, TextStyle};
use rune::Color;

pub fn hx(value: &str) -> Color {
    rune::color::hex(value).expect("valid theme hex")
}

#[derive(Clone, Debug)]
pub struct Theme {
    pub bg: Color,
    pub surface: Color,
    pub surface_alt: Color,
    pub border: Color,
    pub border_strong: Color,
    pub ink: Color,
    pub muted: Color,
    pub faint: Color,
    pub secondary: Color,
    pub secondary_hover: Color,

    pub teal: Color,
    pub teal_hover: Color,
    pub teal_soft: Color,
    pub teal_ink: Color,

    pub coral: Color,
    pub coral_hover: Color,
    pub coral_soft: Color,
    pub coral_ink: Color,

    pub amber: Color,
    pub amber_hover: Color,
    pub amber_soft: Color,
    pub amber_ink: Color,

    pub sky: Color,
    pub sky_hover: Color,
    pub sky_soft: Color,
    pub sky_ink: Color,

    pub green: Color,
    pub green_soft: Color,
    pub green_ink: Color,

    pub crimson: Color,
    pub crimson_soft: Color,

    pub shadow: Color,
    pub scrim: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            bg: hx("#F4F6FB"),
            surface: hx("#FFFFFF"),
            surface_alt: hx("#EEF1F7"),
            border: hx("#DCE1EC"),
            border_strong: hx("#C3CAD9"),
            ink: hx("#131A2B"),
            muted: hx("#55607A"),
            faint: hx("#9AA3B8"),
            secondary: hx("#E8ECF4"),
            secondary_hover: hx("#DCE1EC"),
            teal: hx("#0B7A75"),
            teal_hover: hx("#08625E"),
            teal_soft: hx("#DDF3F1"),
            teal_ink: hx("#075E5A"),
            coral: hx("#C8402A"),
            coral_hover: hx("#A8341F"),
            coral_soft: hx("#FDE7E1"),
            coral_ink: hx("#A3341C"),
            amber: hx("#B87400"),
            amber_hover: hx("#946000"),
            amber_soft: hx("#FFF2D9"),
            amber_ink: hx("#8A5300"),
            sky: hx("#2F74C0"),
            sky_hover: hx("#255F9E"),
            sky_soft: hx("#E2EEFB"),
            sky_ink: hx("#1D5FA0"),
            green: hx("#1F8A53"),
            green_soft: hx("#E1F4E8"),
            green_ink: hx("#16613A"),
            crimson: hx("#A61F35"),
            crimson_soft: hx("#FCE8EB"),
            shadow: hx("#131A2B").alpha(0.055),
            scrim: Color { r: 0.0, g: 0.0, b: 0.0, a: 0.24 },
        }
    }
}

pub mod type_ {
    use super::*;

    pub const APP: TextStyle = TextStyle::new(22.0, 700.0).lh(28.0);
    pub const TITLE: TextStyle = TextStyle::new(15.0, 700.0).lh(20.0);
    pub const STATUS: TextStyle = TextStyle::new(16.0, 700.0).lh(22.0);
    pub const BODY: TextStyle = TextStyle::new(13.0, 400.0).lh(19.5);
    pub const STRONG: TextStyle = TextStyle::new(13.0, 700.0).lh(19.5);
    pub const SMALL: TextStyle = TextStyle::new(12.0, 400.0).lh(17.0);
    pub const SMALL_STRONG: TextStyle = TextStyle::new(12.0, 700.0).lh(17.0);
    pub const PRIMARY: TextStyle = TextStyle::new(14.0, 700.0).lh(19.0);
    pub const MONO: TextStyle = TextStyle::new(11.0, 400.0).lh(16.5).family(Family::SystemMono);
    pub const TIMER: TextStyle = TextStyle::new(22.0, 700.0).lh(27.0).family(Family::SystemMono);
    pub const NAMING_TITLE: TextStyle = TextStyle::new(18.0, 700.0).lh(24.0);
    pub const PRIVACY_TITLE: TextStyle = TextStyle::new(18.0, 700.0).lh(24.0);
}

pub mod metric {
    pub const GUTTER: f32 = 24.0;
    pub const CARD_PAD: f32 = 18.0;
    pub const GAP: f32 = 12.0;
    pub const RADIUS_SM: f32 = 6.0;
    pub const RADIUS_MD: f32 = 10.0;
    pub const RADIUS_LG: f32 = 14.0;
    pub const BUTTON_H: f32 = 36.0;
    pub const CTA_W: f32 = 200.0;
    pub const CTA_H: f32 = 44.0;
    pub const ROW_H: f32 = 46.0;
    pub const STEP: f32 = 26.0;
}

/// Number of file rows shown at once, from the window height at launch.
///
/// The oracle shows three rows when the available height is at least 760 logical
/// points, and two otherwise (1366×768 laptops get two).
pub fn visible_rows(window_height: f32) -> usize {
    if window_height >= 760.0 { 3 } else { 2 }
}
