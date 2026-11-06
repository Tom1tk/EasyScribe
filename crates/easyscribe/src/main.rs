//! Placeholder entry point; replaced by the full Rune UI as it is built.

use rune::tok::{self, type_};
use rune::{AppSpec, Cx, El, col, text};

#[derive(Default)]
struct State;

fn view(_state: &State, _cx: &mut Cx<State>) -> El {
    let t = tok::theme(true);
    col()
        .w_full()
        .h_full()
        .center()
        .gap(12.0)
        .fill(t.bg)
        .child(text("EasyScribe").font(type_::TITLE).color(t.text))
        .child(text("Rune rewrite loading…").font(type_::BODY).color(t.text_secondary))
}

fn main() {
    rune::run(AppSpec::new("EasyScribe 3.2.3", (900.0, 700.0), State::default, view));
}
