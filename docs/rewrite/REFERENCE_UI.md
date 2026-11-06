# EasyScribe 3.2.3 — UI/UX Functional Specification

**Purpose:** implementation reference for the Rust/Rune rewrite of EasyScribe.
This document describes the complete user-visible behaviour, information
architecture, copy, sizing, colour and state model of the v3.2.3 Python GUI.

**Behavioural oracle:** `/home/user/EasyScribe/src/gui.py` (2455 lines), plus the
supporting modules that `gui.py` calls:

- `src/rename.py` — name validation and atomic file renaming.
- `src/settings.py` — remembered recordings folder.
- `src/transcriber.py` — transcript output format, status strings, speaker
  naming callback.
- `src/diarizer.py` — estimate formatting, diarization status/progress.
- `src/config.py` — app version, paths, supported extensions, limits.
- `src/main.py` / `src/recovery.py` — startup checks and the crash-recovery
  prompt.
- `src/mic_recorder.py` / `src/live_transcriber.py` — recording, timer level and
  live transcription semantics.
- `src/transitions.py` — the Windows screen-change fade.

**Rewrite target:** Rune native UI. The workflow, information architecture,
copy, colour meaning and all functional outcomes must match. Visual treatment
may be slightly more polished (see §16), but no user-visible step may be added,
removed, renamed or reordered.

**Reading conventions:** All literal user-visible text is in `code spans`. A
trailing `…` is U+2026. `·` is U+00B7. `✓` is U+2713. `"` are straight ASCII
quotes. Sizes are logical pixels at 100% Windows scaling unless stated
otherwise. "Card" means the white rounded surface with a 1 px border.

Where the checked-in screenshots disagree with the current code, **the code
wins**; see Appendix B for the one known stale screenshot.

---

## 1. Product frame and UX rules

### 1.1 Audience and tone

- Non-technical people transcribing private recordings on their own computer.
- Calm, plain language. One clear next step at a time.
- Designed copy uses sentence case. No em or en dashes, no exclamation marks.
  Engine log lines in the details panel may contain their own punctuation.
- Every control says what it does; no jargon, no file-format talk except the
  supported-extension list.
- The app is fully offline. Nothing is uploaded, shared, tracked or checked for
  updates. The Privacy panel reinforces this.

### 1.2 Theme

- **Light theme only.** `ctk.set_appearance_mode("light")`. There is no dark
  mode, no theme picker, and the UI must not follow the OS dark setting.
- Base CustomTkinter theme is irrelevant; every widget overrides its colours.
- Body and heading text pairs target WCAG AA (4.5:1) or better; disabled
  text is exempt.

### 1.3 Feature colour language

Every feature owns one colour, and that colour follows the feature everywhere:
mode switch, step badge, option chip, progress bar, status text, action button.

| Feature / meaning | Token family | Primary hex | Soft background | Ink/text |
|---|---|---|---|---|
| Transcribe files | teal | `#0B7A75` | `#DDF3F1` | `#075E5A` |
| Record | coral | `#C8402A` | `#FDE7E1` | `#A3341C` |
| Speakers | amber | `#B87400` | `#FFF2D9` | `#8A5300` |
| Timestamps | sky/blue | `#2F74C0` | `#E2EEFB` | `#1D5FA0` |
| Offline / privacy / success | green | `#1F8A53` | `#E1F4E8` | `#16613A` |
| Errors | crimson | `#A61F35` | `#FCE8EB` | `#A61F35` |

Colour is always a reinforcement, never the only carrier of meaning: status
pills, titles and messages also carry text.

### 1.4 Two top-level tasks

The app has exactly two modes, selected by a segmented switch under the header:

1. **Transcribe files** — choose audio/video files, choose timestamp and
   speaker options, transcribe them one after another.
2. **Record** — choose a microphone, record, and either get a best-quality
   transcript after stopping or watch words appear live.

The selected mode keeps its state when switching, but the activity/result card
is hidden on every mode switch (see §7.7).

### 1.5 Visual character

- Calm and airy: a cool off-white slate canvas, white cards, generous 24 px
  outer margins and 12 px gaps. Nothing is dense or dashboard-like.
- Flat in the oracle: 1 px hairline borders, no gradients, no drop shadows, no
  glass effects. Rune may add only the subtle depth described in §16.
- Colour appears as solid primary buttons, soft tinted option chips/step badges
  and thin accent borders. Large neutral areas dominate.
- Dark, bold headings; readable 13 pt body; small muted metadata and paths.
- Rounded rectangular controls, not sharp corners; no icon-only actions except
  the brand mark.
- One vertical task per screen: files above, activity/result below; record setup
  above, live transcript/result below.
- Screenshots at 900×820 show the intended breathing room; at 900×678 the same
  hierarchy compresses by folding options and showing two file rows.

---

## 2. Window, chrome and global layout

### 2.1 Window metrics

| Property | Value |
|---|---|
| Title | `f"{APP_NAME} {APP_VERSION}"` = `EasyScribe 3.2.3` |
| Default width | `900` logical px |
| Default height | `max(600, min(820, floor(screen_height / display_scale) - 90))` |
| Minimum size | `760 × 600` logical px |
| Resizable | yes, both axes; no maximum, no fixed aspect |
| Initial position | OS default; the app does not centre itself |
| Background | `#F4F6FB` |
| Appearance | light only |
| Icon | `assets/EasyScribe.ico` on Windows, else `assets/EasyScribe.png`; set on the main window and the Privacy window |
| Windows taskbar identity | `EasyScribe.App` (set before the first window) |

Height examples:

- 1920×1080 at 100% → `available = 990`, height `820`.
- 1366×768 at 100% → `available = 678`, height `678`; the file list shows
  2 rows before scrolling.
- If `available >= 760`, the file list shows up to 3 rows; otherwise 2.

All dimensions are logical and are multiplied by the Windows display scale.
The Python code computes the screen height in unscaled units first so that the
window fits a scaled 1366×768 laptop. Rune must do the same and remain usable
from 100% to 200% scaling.

### 2.2 Main grid

The root is one column, weight 1.

| Row | Content | Visibility / weight |
|---|---|---|
| 0 | Header: app name left, 40×40 logo right | always |
| 1 | Segmented mode switch | always; left aligned |
| 2 | Active view: Files or Record | exactly one; column weight 1 |
| 3 | Activity / result card | hidden by default; shown while working or showing a result |
| 4 | Bottom utility row: `Show details` left, `Privacy` right | always |
| 5 | Details log card | hidden by default; when open, row weight 1 |

- Outer horizontal gutter is **24 px** on both sides for the header, mode
  switch, cards and utility row.
- Card-to-card vertical gap is **12 px**.
- In **Files** mode, row 2 has weight 0: content sits at its natural height and
  the activity card follows immediately below it.
- In **Record** mode, row 2 has weight 1. The record view therefore claims all
  remaining vertical space. The live-transcript card grows into it; when the
  live card is hidden (best-quality mode), the same space becomes a blank
  spacer between the record card and the activity panel. This produces the
  official look in `transcript-ready.png`; see §6.1 and §15.
- The page itself has **no outer scrollbar**. Long content is kept usable by
  folding the options/setup rows, hiding the action row and giving the file
  list a capped height. At the 600 px minimum some tall result combinations can
  clip; Rune may add a page scroll only as a last-resort polish, without
  changing the order of content.

### 2.3 Header

- Frame transparent, `padx=24`, `pady=(16, 12)`, sticky `ew`.
- Left: `EasyScribe` in 22 pt bold, ink `#131A2B`, vertically centred.
- Right: EasyScribe mark, 40×40, sticky `e`.
- No version text in the header; the version appears only in the window title.

**Logo mark.** A teal rounded tile (corner radius = 28% of size) containing five
white, round-capped vertical bars. At 40 px: bar width 3.6 px, gap 2.8 px, bar
heights 8.8, 18.4, 24.8, 15.2, 7.2 px, group horizontally centred. The tile is
`#0B7A75`; bars are `#FFFFFF`; the canvas background is the window background.
Use the vector/PNG mark from `assets/` in Rune; do not replace it with a generic
microphone icon.

### 2.4 Segmented mode switch

| Property | Value |
|---|---|
| Values | `Transcribe files`, `Record` |
| Height | 38 |
| Corner radius | 10 |
| Container / unselected fill | `#EEF1F7` |
| Unselected hover | `#DCE1EC` |
| Selected fill | teal `#0B7A75` in Files mode; coral `#C8402A` in Record mode |
| Selected hover | `#08625E` teal / `#A8341F` coral |
| Text | `#131A2B` unselected; `#FFFFFF` selected; `#9AA3B8` disabled |
| Font | 13 bold |
| Width | natural width based on the two labels; left aligned |

Behaviour:

- Switching while `_ui_state == "idle"`: show the chosen view, change the
  selected colour to that mode's feature colour, hide the activity card, un-fold
  the options/setup rows, restore the Files action row.
- Switching while running, recording or naming speakers is blocked: the
  segmented button is disabled, and any programmatic change resets it to the
  current mode.
- Mode switch hides any visible result card. Results are not restored when
  switching back; the user must produce a new result. This is existing
  behaviour.

### 2.5 Screen-change motion

On Windows, every "screen" change is wrapped in a short fade (see
`transitions.py`):

- The area below the mode switch is covered by a background-coloured veil.
- Fade out **90 ms**, hold **40 ms**, fade in **140 ms**; ticks every 15 ms.
- Used for: mode switch, show/hide details, show/hide activity, showing results,
  opening/closing the speaker naming panel, showing the Find speakers card.
- Only when the app window is foreground and in `normal`/`zoomed` state. If the
  Windows "Animation effects" setting is off, drawing is still held for the
  90 ms fade-out phase and then released in one go, without the veil. On
  non-Windows platforms changes are immediate.
- `EASYSCRIBE_TRANSITIONS=0` disables the whole effect.

Rune: replace with a 120–160 ms crossfade/slide if desired; honour the OS
reduced-motion setting. The change must never reorder content or cause a layout
jump.

### 2.6 Focus, keyboard and close

- No global keyboard shortcuts. `Return` commits the Name box and the speaker
  naming panel. `Escape` closes only the Privacy dialog.
- Tab order follows widget creation: mode switch, Files content, Record
  content, activity controls, then bottom utility row. Hidden widgets are
  skipped by the native focus system.
- The speaker naming panel focuses its first name entry 100 ms after opening.
- The Privacy dialog takes focus 50 ms after opening.
- Rune should add visible keyboard focus rings (2 px feature-coloured ring or
  equivalent) without changing layout.

**Window close (oracle behaviour):**

- There is **no** `WM_DELETE_WINDOW` handler and **no** close confirmation.
- The native title-bar close destroys the Tk root immediately. Worker threads
  are daemons and are terminated when the process exits.
- `atexit` deletes temporary extracted audio; `main.py` also cleans `temp/` at
  the next start.
- An in-progress microphone recording is left as crash-safe
  `recording_*.pcm` + `.json` in `BASE_DIR/recordings`. The next launch offers
  recovery (§13.6).
- Rune parity: no new close prompt. On close, cancel workers, make a bounded
  best effort to stop capture and flush the PCM/JSON pair, and leave any
  unfinished recording for recovery. Never delete the raw PCM on close.

---

## 3. Design tokens

### 3.1 Palette

Named tokens live in `gui.py` class `C`; the two hover values marked
hard-coded are literals in the widget calls, not class attributes.

#### Neutrals

| Token | Hex | Use |
|---|---|---|
| `BG` | `#F4F6FB` | window background, transparent header canvas |
| `SURFACE` | `#FFFFFF` | cards, inputs, primary button text, white text |
| `SURFACE_ALT` | `#EEF1F7` | drop zone, folded rows, info banner, live textbox, rename/best rows, radio tiles, segmented container |
| `BORDER` | `#DCE1EC` | card border, drop-zone border, radio tile border, progress track |
| `BORDER_STRONG` | `#C3CAD9` | input borders, record radio border, scrollbar thumb |
| `INK` | `#131A2B` | primary text, headings |
| `MUTED` | `#55607A` | secondary text, hints, paths, folded summary, info banner text |
| `FAINT` | `#9AA3B8` | disabled text, idle timer |
| `SECONDARY` | `#E8ECF4` | secondary button fill, disabled primary fill, option menu fill |
| `SECONDARY_HOVER` | `#DCE1EC` | secondary button hover, option menu button |

#### Feature accents

| Token | Hex | Use |
|---|---|---|
| `TEAL` | `#0B7A75` | Files primary, selected mode, progress, logo tile |
| `TEAL_HOVER` | `#08625E` | teal button hover |
| `TEAL_SOFT` | `#DDF3F1` | step-1/2 badges, Working status pill |
| `TEAL_INK` | `#075E5A` | teal-family text, link text, Transcribing status, step numbers |
| `CORAL` | `#C8402A` | Record primary, selected Record segment, recording progress, level meter |
| `CORAL_HOVER` | `#A8341F` | coral button hover |
| `CORAL_SOFT` | `#FDE7E1` | record step badges, selected record tile, mic dropdown hover |
| `CORAL_INK` | `#A3341C` | record-family text, recording status |
| `AMBER` | `#B87400` | speaker primary, Find speakers, Save names, speakers card border, progress |
| `AMBER_HOVER` | `#946000` | amber button hover, speaker-checkbox hover (hard-coded, not a `C` token) |
| `AMBER_SOFT` | `#FFF2D9` | speaker option tile, speaker badges, "rough estimate" |
| `AMBER_INK` | `#8A5300` | speaker text, estimate, partial-success title |
| `SKY` | `#2F74C0` | timestamps checkbox and accents |
| `SKY_HOVER` | `#255F9E` | timestamps checkbox hover (hard-coded) |
| `SKY_SOFT` | `#E2EEFB` | timestamps option tile |
| `SKY_INK` | `#1D5FA0` | timestamps title text |
| `GREEN` | `#1F8A53` | success progress bar |
| `GREEN_SOFT` | `#E1F4E8` | Done status pill, offline/privacy info box |
| `GREEN_INK` | `#16613A` | success title, Done pill text, privacy body |
| `CRIMSON` | `#A61F35` | errors, failed pill, error title, drop warning |
| `CRIMSON_SOFT` | `#FCE8EB` | failed pill, error message background |

Headings and status text on the soft tints use the matching `*_INK`; body copy
on tinted option tiles stays `MUTED`. All text on solid feature buttons is
white. Disabled primary buttons are filled `SECONDARY` with `FAINT` text.

### 3.2 Typography

Family: `Segoe UI` on Windows; the Tk default elsewhere. Monospace: `Consolas`
on Windows; `DejaVu Sans Mono` elsewhere. Rune on Windows should use
`Segoe UI` (or Segoe UI Variable if metrically identical) and `Consolas`.

The sizes below are the values passed to CTk `CTkFont(size=…)`; they are point
sizes. Rune should reproduce the same visual proportions at 100% display scale.

| Token | Size | Weight | Use |
|---|---|---|---|
| `f_app` | 22 | bold | `EasyScribe` header |
| `f_title` | 15 | bold | card titles, step titles, `Live transcript`, `Who said what?` |
| `f_status` | 16 | bold | activity status headline |
| `f_body` | 13 | regular | body text, entries, live transcript, paragraphs |
| `f_strong` | 13 | bold | button labels, option titles, step numbers, strong labels |
| `f_small` | 12 | regular | hints, file metadata, notes, small buttons, share percentages |
| `f_primary` | 14 | bold | `Transcribe` and `Start recording` primary CTAs |
| `f_mono` | 11 | regular | details log |
| `f_timer` | 22 | bold | recording timer |
| Naming title | 18 | bold | `We found N speakers` |
| Naming body | 13 | regular | naming instructions and entries |
| Privacy title | 18 | bold | `Your privacy` |
| Privacy head | 13 | bold | row labels / section heading |

Unspecified CTk button labels use the framework default (~13 regular). In Rune
use 13 regular/13 semibold consistently; do not introduce new sizes.

### 3.3 Radii and spacing

| Token | Value | Use |
|---|---|---|
| `R_SM` | 6 | chips, entries, checkboxes, state pills, message banner |
| `R_MD` | 10 | buttons, cards inside cards, drop zone, option tiles, rename/best rows, live textbox |
| `R_LG` | 14 | top-level cards |
| Step badge | 13 | half of 26 px circle |
| Progress bar | 4 | half of 8 px height |
| Logo tile | 28% of size | 11.2 px at 40 px |

Spacing rhythm:

- Window gutter: 24.
- Card inner padding: 18 left/right, generally 16 top / 12–16 bottom.
- Row padding inside cards: 18; compact rows 12–14.
- Gap between major cards: 12.
- Gap between paired buttons: 8.
- Default control gap: 8–14.
- Drop zone border, record-choice border, speakers card border and naming
  panel border: **2 px**.
- Standard card/section border: **1 px**.
- Use an 8 px spacing grid for any Rune-added internal alignment; preserve the
  listed metrics where exact.

### 3.4 Control sizes

| Control | Size / note |
|---|---|
| Header logo | 40 × 40 |
| Segmented switch | height 38, radius 10 |
| Standard button | height 36, radius 10 |
| Small button | height 28 or 30 or 32 depending on context |
| Primary CTA | height 44, width 200 (Transcribe / Start recording) |
| Step badge | 26 × 26 circle |
| `Choose files` / `Add files` | width 130, height 36 |
| `Clear all`, Files `Change` | width 80, height 28 |
| Record summary `Change` | width 70, height 28 |
| `Use default` | width 100, height 32 |
| `Change folder` | width 130, height 32 |
| File row | 46 px high (two text lines + padding) |
| File name | height 20 (single line, shortened) |
| File metadata | height 18 |
| State pill | width 72, height 24 |
| `Remove` | width 70, height 28 |
| Checkbox | 22 × 22 |
| Radio | 20 × 20 |
| Microphone menu | width 340, height 36 |
| Level meter | width 120, height 8, radius 4 |
| Timer | 22 pt bold mono; no fixed box |
| Live transcript textbox | min height 120 |
| Activity `Stop` | width 80, height 32 |
| Progress bar | height 8, radius 4 |
| `Open transcript` | width 150, height 36 |
| `Open folder` | width 120, height 36 |
| `Try again` | width 110, height 36 |
| Rename entry | height 36 |
| `Rename` | width 100, height 36 |
| `Make best quality transcript` | width 240, height 36 |
| `Find speakers` | width 170, height 40 |
| Speaker badge | width 92, height 28 |
| `Play sample` / `No sample` | width 110, height 32 |
| Speaker entry | height 34 |
| Share label | width 130, right/left as described |
| Speaker list scroll area | height 248 (4 × 62) when >4 speakers |
| `Skip names` | width 110, height 36 |
| `Save transcript with names` | width 230, height 36 |
| `Show details` | width 110, height 30 |
| `Privacy` | width 80, height 30 |
| Details log | min height 140, expands with row weight |
| Privacy `Open logs folder` | width 150, height 36 |
| Privacy `Close` | width 100, height 36 |

### 3.5 Interaction states

| State | Treatment |
|---|---|
| Primary teal | fill `#0B7A75`, hover `#08625E`, white text |
| Primary coral | fill `#C8402A`, hover `#A8341F`, white text |
| Primary amber | fill `#B87400`, hover `#946000`, white text |
| Secondary | fill `#E8ECF4`, hover `#DCE1EC`, text `#131A2B` |
| Subtle | white fill, 1 px `#DCE1EC` border, hover `#EEF1F7`, text `#55607A` |
| Link | transparent fill, hover `#EEF1F7`, text `#075E5A` (Remove overrides text to `#55607A`; Clear all keeps `#075E5A`) |
| Disabled primary | fill `#E8ECF4`, text `#9AA3B8`, no hover |
| Disabled generic | framework dim, text `#9AA3B8`/`#55607A`; no hover |
| Selected record tile | bg `#FDE7E1`, 2 px border `#C8402A` |
| Unselected record tile | bg `#EEF1F7`, 2 px border `#DCE1EC` |
| Checked option | checkbox fill feature colour, white checkmark; tile background stays soft feature tint |
| Unchecked option | border feature colour; tile background stays soft feature tint |
| Drag over drop zone | background `#DDF3F1`, 2 px border `#0B7A75` |
| Error banner | `#A61F35` text on `#FCE8EB` |
| Info banner | `#131A2B` text on `#EEF1F7` |

The Python app does not define a separate pressed state; Rune may darken a
pressed button by ~5% for tactile feedback without changing the palette.

---

## 4. Shared components

### 4.1 Card

- White `#FFFFFF`, radius 14, 1 px `#DCE1EC` border, no shadow in the oracle.
- Rune may add a very subtle shadow: `0 1px 2px rgba(19,26,43,.06)` plus
  `0 8px 24px rgba(19,26,43,.04)`; keep the 1 px border and white fill.
- Content is inset 18 px left/right unless a nested element says otherwise.
- Cards themselves never scroll; the file list, speaker list, live transcript
  and details log have their own scrollable areas.

### 4.2 Step header

Used identically in both tabs:

- A 26 × 26 circular badge with the step number, radius 13, soft feature
  background, feature ink text, 13 bold.
- 10 px gap, then a 15 bold ink title.
- The rest of the row expands; optional right-side controls sit at the far
  right (count/link or Change).
- Header padding inside the card: Files steps use 18 px sides and `(16, 10)`
  top/bottom; Record step 1 uses `(16, 8)`, Record steps 2 and 3 use `(0, 8)`.

Files steps: `1` `Add recordings`; `2` `Choose options`.
Record steps: `1` `Choose a microphone`; `2` `What do you want?`; `3` `Record`.

### 4.3 Buttons

- Rectangular rounded buttons, no icons unless noted.
- `_button()` defaults: height 36, radius 10, no border, disabled text
  `#9AA3B8`.
- Variants: teal, coral, amber, secondary, subtle, link (see §3.5).
- Primary CTAs (`Transcribe`, `Start recording`) are 200 × 44 and 14 bold.
- Small utility buttons use 12 pt; `Clear all`/`Change` are 80 × 28.
- Focus: Rune should draw a 2 px ring in the current feature colour, offset 2 px.

### 4.4 Labels, entries and text areas

- Body labels: 13 regular, ink; secondary: 13 regular, muted.
- Meta: 12 regular, muted; paths use the same style and are middle-shortened.
- Field labels: 13 bold, ink.
- Entry: white fill, 1 px `#C3CAD9` border, radius 6, ink text, 13 regular;
  placeholder text muted/faint; height 36 (34 in speaker row).
- No field has a separate visible label association in the oracle, but Rune
  should expose accessible names from the adjacent text.
- Read-only text areas use disabled state (selection/copy allowed if the
  framework permits) and never show a caret.

### 4.5 Progress bar

- Track: `#DCE1EC`; height 8; radius 4; full width inside 18 px card padding.
- Fill uses the current status colour:
  - teal for files, loading, extraction, transcription, writing;
  - amber for speakers/diarization;
  - coral for recording/finishing/saving;
  - green for success;
  - crimson/unchanged for failure;
  - faint for cancel.
- Modes: determinate percentage or indeterminate "busy". Indeterminate speed
  0.6 in CTk. Percent label only in determinate running states.
- Idle/empty track has no visible leading dot (fill colour equals track when
  idle in the record level meter).

### 4.6 Message banner

- Shown between the batch label/progress and the result actions.
- Full width inside 18 px card padding, radius 6.
- Error: `#A61F35` text on `#FCE8EB`.
- Info/neutral: `#131A2B` text on `#EEF1F7`.
- 13 regular, left aligned, wrap ~760 px, internal padding `10 × 8`.
- Hidden entirely when message text is empty.

### 4.7 Checkbox and radio tiles

- Tile radius 10; no border for option tiles; 2 px border for record choice
  tiles.
- Checkbox 22 × 22, radius 6, white checkmark, feature fill when checked.
- Radio 20 × 20, coral fill/dot when selected.
- Title 13 bold; body 12 regular muted, wraplength 330 (options) / 320
  (record), left inset 46 (checkbox) or 44 (radio) to align under the title.
- Record tiles: the whole tile and its body note are clickable and select the
  choice, not only the radio circle. Option tiles: only the checkbox toggles.
- Option tile background is the feature soft tint whenever the option is
  available, regardless of checked state. If unavailable, it becomes
  `#EEF1F7`, title muted, checkbox disabled.

### 4.8 Drop zone

- Fill `#EEF1F7`, radius 10, 2 px `#DCE1EC` border.
- Two-left-line text plus a right-aligned button, height ~56–72 depending on
  whether the subtitle is shown.
- Title 13 bold ink; subtitle 12 muted; left padding 16.
- Whole add card, drop zone, title and subtitle are drag-and-drop targets.
- On drag enter (idle only): fill `#DDF3F1`, border `#0B7A75`. On leave/drop:
  reset to `#EEF1F7` / `#DCE1EC`.
- If native drag-and-drop is unavailable, the UI still works via the file
  picker; see copy table §5.2.

### 4.9 Status pill

- 12 regular, radius 6, width 72, height 24, centred in the file row.
- Backgrounds/text:

| State | Text | Text colour | Background |
|---|---|---|---|
| Waiting | `Waiting` | `#55607A` | `#EEF1F7` |
| Working | `Working` | `#075E5A` | `#DDF3F1` |
| Done | `Done` | `#16613A` | `#E1F4E8` |
| Failed | `Failed` | `#A61F35` | `#FCE8EB` |
| Stopped | `Stopped` | `#55607A` | `#EEF1F7` |

### 4.10 Read-only text boxes

- Live transcript: fill `#EEF1F7`, radius 10, no border, ink text, 13 regular,
  word wrap, min height 120; muted placeholder text; auto-scroll to end.
- Details log: white fill, no border, 11 mono, muted text, min height 140,
  word wrap; auto-scroll to end.
- Both are read-only to the user but can be selected/copied if Rune allows.

### 4.11 Scrollbars

- File list scrollbar explicitly uses `#C3CAD9` thumb, `#9AA3B8` hover.
- The file list's scrollbar is auto-sized (height forced to 0 in the Python
  implementation so one row does not reserve 200 px); Rune should auto-hide the
  scrollbar when all rows fit.
- Speaker list >4 uses the framework scrollbar; no custom colours.
- No horizontal scrolling anywhere.

---

## 5. Transcribe files view

### 5.1 Information architecture

Normal, populated state:

```text
┌ EasyScribe                                             [logo] ┐
│ [ Transcribe files ]  Record                                  │
│ ┌ ① Add recordings                         3 files  Clear all ┐│
│ │ ┌ Drag more files here                          [Add files] ┐││
│ │ │ Team meeting 2026-09-14.m4a                  Done Remove │││
│ │ │ 28.2 MB  ·  /home/user/demo                             │││
│ │ │ Interview with Sam.mp3                       Done Remove │││
│ │ │ ...                                                       │││
│ │ └───────────────────────────────────────────────────────────┘││
│ └──────────────────────────────────────────────────────────────┘│
│ ┌ ② Choose options                                     Change ┐│
│ │ Timestamps on, speaker names on, saved next to each recording.│
│ └──────────────────────────────────────────────────────────────┘│
│ [ Transcribe 3 files ]                                          │
│ [Show details]                                         [Privacy]│
└─────────────────────────────────────────────────────────────────┘
```

With the activity card visible, the options / setup rows fold to one summary
line and the action row is removed; the activity card takes its place below the
view (§7.7). The add-recordings card and file rows stay visible.

### 5.2 Empty state

| Element | State |
|---|---|
| Step header | badge `1`, title `Add recordings`; count label empty; `Clear all` hidden |
| Drop-zone title (DND available) | `Drag audio or video files here` |
| Drop-zone subtitle | `Or choose them from your computer. MP3, WAV, M4A, MP4 and more.` |
| Drop-zone title (DND unavailable) | `Choose the recordings to transcribe` |
| Drop-zone subtitle (DND unavailable) | `MP3, WAV, M4A, MP4 and other audio and video files.` |
| Drop button | `Choose files`; teal; 130 × 36 |
| File list | hidden |
| Count label | empty |
| `Clear all` | hidden |
| Options | fully expanded; default `Add timestamps` off, `Name the speakers` off |
| Output | `The same folder as each recording`, muted; `Use default` hidden |
| Start button | `Transcribe`, disabled (secondary fill, faint text) |
| Start hint | `Add a recording to start.` muted |
| Bottom row | `Show details` left, `Privacy` right |

### 5.3 Populated state

As soon as `count > 0`:

- Drop title becomes `Drag more files here` (DND) or remains the no-DND title
  `Choose the recordings to transcribe`.
- Drop subtitle is hidden; the title gets 14 px top and bottom padding, making
  the drop zone one line high.
- Button text becomes `Add files`.
- Count label becomes `_plural(count, "file")` → `1 file`, `2 files`, `3 files`.
- `Clear all` appears at the right of the step header.
- File list appears, height `min(count, visible_rows) × 46`, with rows as below.
- Start button becomes `Transcribe N files` and is enabled; hint blank.

**File row** (46 px, transparent, 8 px horizontal padding inside list, 3 px
vertical padding so rows sit 6 px apart):

- Line 1: middle-shortened filename to 60 characters, 13 regular ink, height
  20.
- Line 2: `"{size}  ·  {middle-shortened parent path to 60}"`, 12 regular
  muted, height 18. Two spaces surround `·`. If `stat()` fails, size is
  `Not found` and the parent path is still shown.
- Optional status pill, 72 × 24, spanning both lines, 8 px gap before the
  Remove link. No pill is shown before a batch starts; once a batch starts,
  every row gets `Waiting` and then updates as the worker reports.
- `Remove` link button, 70 × 28, muted text, right aligned; disabled while a
  job runs.

No per-row progress bar and no click action on the row. Removing a file is only
through `Remove`.

### 5.4 Adding, removing and clearing files

**Choose files**

- Opens a native multi-select file dialog:
  - title `Choose recordings to transcribe`;
  - filter label `Audio and video files`, patterns generated from the supported
    extension set in sorted order: `*.aac *.avi *.flac *.m4a *.mkv *.mov
    *.mp3 *.mp4 *.ogg *.opus *.wav *.webm`;
  - second filter `All files` with `*.*`.
- Selected files are filtered by supported extension in `_add_files`; unsupported
  selections are listed in the drop note (not silently ignored).
- Duplicate paths already in the list are ignored.
- Adding while a result is visible and the app is idle in Files mode hides the
  activity card and removes all file-state pills, keeping only rows that were
  still `Waiting`. The list itself is preserved and the new files appended.

**Drag and drop**

- Available only if tkinterdnd2/Tcl DnD initialised. If not available, the UI
  uses the file-picker copy in §5.2.
- Targets: the whole add-recordings card, drop zone, drop title and drop
  subtitle.
- Uses `tk.splitlist` to parse the platform list; each path must exist, be a
  file, and have a supported suffix. Directories and unsupported files are
  collected as "skipped".
- Valid files are added; the skipped list produces the red drop note:

  `Skipped {names}{more}. EasyScribe reads audio and video files: {extensions}.`

  where `names` is up to 3 middle-shortened names (limit 40), joined with
  `, `; `more` is ` and N more` when more than 3; extensions are sorted
  uppercase without dots, joined `, `:
  `AAC, AVI, FLAC, M4A, MKV, MOV, MP3, MP4, OGG, OPUS, WAV, WEBM`.
- If nothing was skipped, the note is hidden. The note persists until another
  drop/selection with no skips, or `Clear all`.
- During any busy state, drop enter does not highlight and drop does nothing.
- Full file paths from the OS are used; no dialogs are opened on drop.

**Clear all**

- Visible only when at least one file is present; no confirmation.
- Allowed only when idle (button disabled otherwise).
- Hides the activity card when idle, clears `_selected_files` and all file
  states, hides the drop note, re-renders the empty list. Options, output folder
  and record settings are untouched.

**Remove**

- Allowed only when idle.
- Removes that path from the selection and its state, re-renders. It does not
  hide an already-visible result card or clear `last_transcript`.

**Supported extensions**

`.mp4 .mkv .mov .avi .webm .mp3 .wav .m4a .flac .ogg .opus .aac`
(case-insensitive suffix match). The drop note/help displays them uppercase.

### 5.5 Step 2: Choose options

Expanded card contents, top to bottom:

1. Step header: badge `2`, title `Choose options`; right `Change` link only
   when the card is folded and idle (see §7.7).
2. Field label `Save transcripts to` (13 bold ink).
3. Output row:
   - default: `The same folder as each recording` (13 muted);
     `Use default` hidden.
   - custom: middle-shortened absolute path (limit 70) in ink;
     `Use default` link 100 × 32 on the right.
   - `Change folder` secondary 130 × 32 at the far right; opens a native folder
     picker titled `Choose where to save transcripts`. No initial directory.
   - `Use default` clears the custom choice; no persistence across launches.
4. Two option tiles side by side, equal width, 18 px outer / 6 px inner
   paddings, 12 px between:

**Add timestamps (sky)**

- Checkbox title `Add timestamps`, 13 bold, text `#1D5FA0`.
- Body: `Shows the time, like [00:01:23], at the start of each paragraph.`
- Default off. Available always. Checkbox fill `#2F74C0`, hover `#255F9E`,
  border same, white checkmark; tile `#E2EEFB`.

**Name the speakers (amber)**

- Checkbox title `Name the speakers`, 13 bold, text `#8A5300` when available.
- Available body:
  `Finds who is talking. You hear a short sample of each voice and type their name. Takes longer.`
- Unavailable body:
  `Not available: the speaker files are missing from this copy of EasyScribe.`
- Default off. If diarization models are missing at startup, the tile is
  `#EEF1F7`, title is muted, the checkbox is disabled, and the unavailable body
  is shown. Availability is not re-checked until restart.
- Checkbox fill `#B87400`, hover `#946000`, border amber, white checkmark; tile
  `#FFF2D9`.

Both option checkboxes and the output-folder buttons are disabled while a job is
running/recording. Ticking `Name the speakers` makes file transcription slower;
the UI gives no extra confirmation.

### 5.6 Start row

- Left: primary `Transcribe` / `Transcribe N files` button, 200 × 44, teal,
  14 bold.
- Empty selection: button disabled, hint `Add a recording to start.`
- Non-empty idle: button enabled; text `Transcribe N files` using `_plural`;
  hint empty.
- Running: button text `Transcribing…` (U+2026), disabled; hint empty.
- Defensive click with no files (normally impossible because disabled) sets the
  hint to `Add at least one recording first.` in crimson.
- Start has no confirmation step; the only possible prompt is the low-disk check
  below. Before starting, it checks free disk space in the chosen output folder,
  or the first file's parent when using the default output:
  - if free < **512 MB**, a native yes/no dialog:
    - title `Low disk space`
    - text `Only {N} MB of disk space is free.\nTranscription may fail. Continue anyway?`
    - `No` aborts; `Yes` starts. A failure to read disk usage is ignored.
- The options snapshot is `{timestamps: bool, diarize: bool}`.
- The batch always uses a snapshot of the selected file list; the original
  files are never written to.

### 5.7 Running state

When transcription starts:

- `_ui_state = "running"`; `_cancel_event` is cleared.
- All selected files get `Waiting`.
- Log is cleared.
- Activity card is shown below the view, options/setup collapse to summaries,
  the Files action row is hidden.
- File list rows update `Working → Done / Failed / Stopped`, one current file
  at a time.
- Mode switch, choose/add files, clear all, remove, output folder controls,
  checkboxes and record controls are disabled. `Privacy` and `Show details`
  remain available. Drop is ignored.
- Activity shows the status (`Reading the audio`, `Transcribing`, etc.),
  current file line, progress bar and `Stop` at the right of the status row.
- `Stop` sets the shared cancel event, changes the status to
  `Stopping…`/`Stopped` (title mapping), and disables itself. It does not close
  the window or discard partial results.

### 5.8 Result states (files)

The activity card becomes the result card. `Open transcript` appears only when
exactly one file was saved, because there is a single "last transcript" target.
`Open folder` appears whenever at least one file was saved. `Try again` appears
for any failure or cancellation (except a fatal dependency failure). All
buttons are idle-only actions.

| Outcome | Status title | Detail line | Message banner | Buttons |
|---|---|---|---|---|
| All files succeeded, 1 saved | `Transcript ready` (green) | `Saved in {path}` | none | `Open transcript`, `Open folder` |
| All files succeeded, N > 1 saved | `N transcripts ready` (green) | `Saved in {path}` | none | `Open folder` |
| Some failed, ≥1 saved | `{saved} of {total} transcripts ready` (amber) | `Saved in {path}` | `Could not transcribe {names}{more}. The file may be damaged or have no sound. Select "Show details" to see why.` | `Open transcript` if exactly 1 saved, `Open folder`, `Try again` |
| All failed | `Transcription failed` (crimson) | empty | same "Could not transcribe" message | `Try again` |
| Stopped, ≥1 saved | `Stopped` (muted) | `1 transcript saved before you stopped.` or `{N} transcripts saved before you stopped.` | none | `Open transcript` if exactly 1 saved, `Open folder`, `Try again` |
| Stopped, none saved | `Stopped` (muted) | `No transcript was saved.` | none | `Try again` |
| Fatal dependency missing | `EasyScribe could not continue` (crimson) | `A program file is missing. Please download EasyScribe again.` | raw fatal error text | Open buttons only if earlier files saved; no retry |

- Progress bar: green and full when at least one file was saved; otherwise 0.
- Percent label is cleared on every result.
- `Try again` and `Change` both call `_on_change_options`: they only hide the
  result card and return to the editable options/action row. They do **not**
  automatically start another run.
- The rename row is offered when there is a single result target (one saved
  transcript, or a recording set in Record mode); see §9.
- Multiple saved files never offer `Open transcript` or the Name box because
  `last_transcript` is intentionally `None`.
- When one file fails, later files continue. A fatal `FFmpegNotFoundError` or
  `ModelNotFoundError` stops the whole batch immediately.
- The batch detail uses the last saved output folder, shortened to 90
  characters. If files went to different folders (default "next to each
  recording"), the detail names only the last one.
- The fatal message banner shows the exception verbatim, for example
  `Model validation failed:\nMissing: {path}` or
  `ffmpeg.exe not found at: {path}`.

### 5.9 Batch worker algorithm (exact order)

The worker runs on a daemon thread `TranscriptionWorker`; only safe `after(0)`
callbacks touch widgets.

For each input file `i` of `n`, sequential:

1. If cancel is already set, break.
2. Batch label:
   - `n > 1`: `File {i} of {n}: {filename shortened to 70}`;
   - `n == 1`: `{filename shortened to 80}`.
3. File state `Working`.
4. Append log separator: a blank line, then 50 `─` characters; then
   `[File {i}/{n}] {input_file.name}`.
5. Status `Extracting Audio`, progress 0. Extract a temporary mono 16 kHz WAV.
   If cancelled, terminate ffmpeg and raise `CancelledError`.
6. Resolve output path (§12). Set `last_output_folder` to its parent.
7. Status/progress come from the engine:
   `Transcribing` → optional `Identifying Speakers` → optional
   `Naming Speakers` (waits for the panel) → `Writing Transcript`.
8. On success: append output to `saved`, store `last_segments` when this batch
   is a recording (`_batch_recording`), set file state `Done`.
9. On `CancelledError`: log `[Cancelled] Operation stopped by user`, set current
   state `Stopped`, break.
10. On `FFmpegNotFoundError` or `ModelNotFoundError`: log `[Fatal] {exc}`, set
    state `Failed`, add to `failed`, remember `fatal`, break.
11. On `FFmpegExtractionError` or `TranscriptionError`: log
    `[Error] {filename}: {exc}`, state `Failed`, add to `failed`, continue.
12. On any other exception: log `[Unexpected Error] {filename}: {exc}`, state
    `Failed`, add to `failed`, continue.
13. Always delete the temporary WAV.

Final status:

- cancel event set → `Cancelled`;
- no failures → `Done`;
- every file failed or fatal → `Failed`;
- non-fatal partial failures → `Done`, plus log
  `[Summary] Completed with {N} error(s) out of {M} file(s)`.

The UI then calls `_finish_batch` on the main thread, which sets idle, stores the
single last transcript (only if exactly one saved), sets the last output folder,
overrides the status/detail/message as in §5.8, renders the result actions, and
offers the Name box / Find speakers card where applicable.

---

## 6. Record view

### 6.1 Information architecture

```text
┌ EasyScribe                                             [logo] ┐
│ Transcribe files  [ Record ]                                  │
│ ┌ ① Choose a microphone                                      ┐│
│ │ [ Default microphone                               ▼ ]     ││
│ │ ② What do you want?                                        ││
│ │ ┌ ● Best quality transcript after recording ┐ ┌ ○ Show words as I speak ┐
│ │ │ Saves the recording.                      │ │ Saves the recording.   │
│ │ │ Adds times. Most accurate. Find speakers after. │ Quick text while you talk. Less accurate. │
│ │ └───────────────────────────────────────────┘ └────────────────────────┘
│ │ Save recordings to                                         ││
│ │ /home/user/EasyScribe                     [Change folder]  ││
│ │ ③ Record                                                   ││
│ │ [ Start recording ]  00:00  ▭▭▭                            ││
│ └────────────────────────────────────────────────────────────┘│
│                    (blank flex space in best-quality mode)     │
│ [Show details]                                         [Privacy]│
└─────────────────────────────────────────────────────────────────┘
```

In live mode, a `Live transcript` card appears directly below the record card
and grows to fill the record view's weighted row. The activity/result card then
sits below the live card.

### 6.2 Step 1: microphone

- Badge `1`, title `Choose a microphone` coral.
- `CTkOptionMenu`, width 340, height 36, radius 10, 13 regular; dynamic
  resizing disabled so the control does not jump when a long device name is
  selected.
- Fill `#E8ECF4`, dropdown button `#DCE1EC`, button hover `#C3CAD9`, text ink,
  disabled text faint; dropdown white with coral-soft hover `#FDE7E1`.
- Values are built once at startup:
  - always first: `Default microphone` mapped to OS default device (`None`);
  - then every input device from `MicRecorder.list_devices()`, name stripped
    and middle-shortened to 48 characters; duplicate visible names are shown
    once, using the first device index.
- If enumeration fails, only `Default microphone` is present. There is no
  refresh control and no live meter before recording.
- Disabled while any job runs/records.

### 6.3 Step 2: what do you want?

Two equal-width radio tiles, 12 px gap, 18 px outer / 6 px inner paddings,
2 px border, radius 10:

**Best quality transcript after recording** (default)

- Title 13 bold ink.
- Body, exactly two lines:
  `Saves the recording.`
  `Adds times. Most accurate. Find speakers after.`
- Default selected. Selected tile `#FDE7E1` with coral border; unselected
  `#EEF1F7` with `#DCE1EC` border.
- Radio 20 × 20 coral; whole tile and body note clickable.
- Effect: record only; at Stop, run the accurate file pipeline immediately,
  with timestamps, no diarization, next to the recording. Then offer Find
  speakers.

**Show words as I speak**

- Title 13 bold ink.
- Body, exactly two lines:
  `Saves the recording.`
  `Quick text while you talk. Less accurate.`
- Effect: load the live recognizer, record, show VAD segments in the live
  transcript as they are decoded, save the recording and a plain live
  transcript at Stop.

Selection is allowed only while idle. The chosen radio mode is not persisted;
each launch starts on Best quality.

### 6.4 Save recordings to

- Label `Save recordings to`, 13 bold ink.
- Path line:
  - default: middle-shortened `DEFAULT_RECORDINGS_DIR` (limit 70), muted;
    `Use default` hidden.
  - custom: middle-shortened chosen folder, ink; `Use default` link visible.
- `Change folder` secondary 130 × 32; opens a native folder picker titled
  `Choose where to save recordings`, initial directory is the current
  recordings folder. Choosing calls `settings.save_recordings_dir()`.
- `Use default` clears the setting via `save_recordings_dir(None)` and shows the
  default path again.
- Persistence: JSON key `recordings_dir` in `BASE_DIR/recordings/settings.json`.
  On load, a stored folder that no longer exists is treated as unset (default).
  This is the only persisted user choice.
- Default is `Documents/EasyScribe` (the real Documents known folder on Windows,
  including OneDrive redirection; fallback `~/Documents` or home then
  `EasyScribe`).
- If a recording cannot be written to the chosen folder at Stop, the WAV is
  saved next to the crash-safe PCM in `BASE_DIR/recordings`, and only a warning
  is written to the log; the UI still reports the actual parent folder.

### 6.5 Step 3: record row

- Badge `3`, title `Record` coral.
- `Start recording` primary coral, 200 × 44, 14 bold.
- Timer: `00:00`, 22 bold mono, faint at idle; while recording coral ink.
  - Starts at 0 when capture actually starts (after the optional live model
    load).
  - Format `MM:SS`, or `H:MM:SS` once the hour is reached (hours not
    zero-padded).
  - Updates every 500 ms.
  - On stop, the text stays at the last duration but returns to faint. It is
    not reset to `00:00` until the next recording starts.
- Level meter: 120 × 8, radius 4, track `#DCE1EC`; while recording the fill is
  coral and set to `sqrt(peak)` capped at 1.0, updated every 100 ms. Peak is the
  loudest sample since the previous read. When not recording it resets to 0 and
  the fill colour is the track colour (so the empty bar shows no coral dot).
  It exists to reassure that the microphone hears sound, especially when no
  words appear.
- No input-level label, no dB scale, no mute control.

### 6.6 Live transcript card

Only present when `Show words as I speak` is selected (otherwise the card is
removed, leaving the flex blank space described in §6.1).

- Card title `Live transcript`, 15 bold ink, 18 px padding, `(14, 6)`.
- Textbox fills the card, min height 120, `#EEF1F7` fill, no border, radius 10,
  13 regular ink, word wrap, read-only, 18 px sides / 16 px bottom.
- Idle placeholder:
  `Select Start recording. The words appear here as people speak.`
- When recording starts, the box is reset to:
  `Listening. The words appear here as people speak.`
- Placeholder text is muted. On the first real segment, the placeholder is
  cleared, text becomes ink, and each decoded segment is appended stripped plus
  a newline; the view auto-scrolls to the end.
- No timestamps or speaker labels in live text.
- If the live decoder falls behind, the log gets:
  `[Warning] transcription falling behind. Some audio was skipped in the live view; the full recording is still saved`
  (lowercase after Warning, exactly as written). The live view may miss text;
  the saved recording is complete.

### 6.7 Recording lifecycle and states

**Start**

1. Read chosen mode and microphone; clear stop event and last transcript /
   segments.
2. `_ui_state = "recording"`; log cleared; live box set to Listening placeholder
   (even in best mode, where the box is hidden); activity card shown; options
   and record setup fold; start log line:
   - live: `[Record] Starting live recording…`
   - best: `[Record] Starting recording (best quality transcript after the stop)…`
3. Record button becomes `Stop recording` and is wired to stop.
4. Worker:
   - live only: load the speech recognizer first; status `Loading Model`
     (`Getting ready`), with hint `Loading the speech model. This can take a minute.`
   - choose destination, ensure `BASE_DIR/recordings`, start microphone capture
     at 16 kHz mono int16, writing crash-safe PCM + JSON sidecar.
   - status `Recording`; start timer and level tick.
   - live only: start VAD and decoder threads. Log
     `[Record] Listening. Select Stop recording when done`.
   - wait indefinitely for the stop event.
5. During recording: mode switch disabled; mic, choice tiles, folder controls,
   file controls disabled; record button remains enabled as Stop; activity
   status `Recording`, indeterminate coral progress, no percent; the batch line
   stays empty (no file is being processed); no activity Stop button.

**Stop**

- Clicking Stop sets the stop event and immediately changes the record button to
  `Saving…` (U+2026) and disables it. No confirmation, no discard option.
- Live: status `Finishing` (`Finishing the last words`) while the live VAD
  flushes the sentence in progress and the decoder drains its queue (up to 60 s).
- Best: status `Saving Recording` (`Saving the recording`).
- Microphone stops; PCM is converted to WAV in the chosen recordings folder.
- Live only: if at least one line was decoded, write `<session_stem>.txt` next
  to the WAV, one line per segment, UTF-8, trailing newline.
- Sidecars are deleted only after the WAV is safely written.
- Status `Done`; then the UI finish handler runs.

**Finish**

- Reset record button to `Start recording`; clear timer ticker; `_ui_state` back
  to idle; percent cleared; progress motion stopped.
- If the worker reported an error and no audio exists:
  - status `Could not record` crimson; batch line empty; error banner with the
    raw error; progress 0; no result buttons; if no live text, live box returns
    to the idle placeholder.
- If best mode and audio exists: immediately start the best-quality file
  pipeline (below). The "Recording saved" intermediate result is not shown.
- Otherwise show `Recording saved` green:
  - detail `The audio is in {path}.`
  - live with transcript:
    `The audio and the live transcript are in {path}.`
  - live without transcript:
    `No speech was heard live, so there is no live transcript. The audio is in {path}.`
  - error (for example the live transcript could not be written) is shown in the
    error banner even while the status is green.
  - progress green full.
  - result actions: `Open transcript` only when a live transcript exists,
    `Open folder` always; Name box; `Make best quality transcript` row when an
    audio file exists.

### 6.8 Best-quality-after-recording flow

`_on_full_transcript()` is used both automatically after a best-mode recording
and by the `Make best quality transcript` button after a live recording.

Preconditions: idle and `last_recording` set. If the audio file has been moved
or deleted, show error banner `The recording was moved or deleted.`

- Output folder: **the recording's own folder**, never the Files tab output
  choice.
- Output name: `<audio stem>.txt`; if that file already exists, use
  `<audio stem> (best quality).txt`. (No further collision check.)
- Options are forced: `timestamps = true`, `diarize = false`.
- `_batch_recording = audio`, so the result is treated as a recording result:
  no Files-tab list changes; the record tab stays open.
- Start a normal one-file batch with the activity panel. The record summary
  stays folded at the top; the record button is disabled while running.
- On success:
  - status `Transcript ready`, detail `Saved in {path}`, green full bar;
  - `Open transcript`, `Open folder`, Name box;
  - `last_segments` is stored, so the Find speakers card is offered when
    diarization models are available.
- On failure/stop: the normal result state is shown with `Try again` and
  `Make best quality transcript` (if the recording still exists), so the user
  can retry. A fatal missing-dependency failure omits both and is presented as
  `EasyScribe could not continue`.

The live transcript is never overwritten: if it exists at `<stem>.txt`, the
best-quality transcript becomes `<stem> (best quality).txt`.

### 6.9 Record error states

| Failure | UI |
|---|---|
| Microphone/start capture error | status `Could not record` crimson; error text: `Could not record: {exc}\nCheck that a microphone is connected, and that Windows allows apps to use it (Settings, Privacy, Microphone).` |
| Model missing while loading live recognizer | status mapped to failure briefly, then `Could not record`; error text: `A program file is missing. Please download EasyScribe again.\n{exc}` |
| Live transcript write failed but audio saved | status `Recording saved`; error banner `Could not save the transcript: {exc}`; audio still openable |
| Save to chosen folder failed | no block-level error dialog; log warning `[Warning] Could not save to {dest}. The recording is in {wav.parent}`; the WAV is in the work folder and `last_output_folder` points there |
| Best-quality target audio missing | error banner `The recording was moved or deleted.`; no batch starts |

After any recording error the activity card stays visible with the error and the
setup/options folded; `Change` returns to editing.

---

## 7. Activity, progress and result card

This one shared card is used in both modes for work-in-progress and results.
It is hidden at launch and on every mode switch, and is shown by the batch,
record, find-speakers and result flows.

### 7.1 Anatomy and order

Inside the white card, top to bottom:

1. **Top row** (`padx=18`, `pady=(14,0)`):
   - status title, 16 bold, left, feature-coloured;
   - percent label, 13 bold muted, right;
   - `Stop` secondary button 80 × 32 at the far right, 14 px gap; only shown
     during `running` (file batch or Find speakers), never during recording.
2. **Batch line** (`padx=18`, 13 regular muted, left, wrap 780), e.g.
   `File 2 of 3: Interview with Sam.mp3`.
3. **Progress bar** (`padx=18`, `pady=(8,14)`, height 8, radius 4, track
   `#DCE1EC`), full width.
4. **Message banner** (hidden when empty), radius 6, wrap 760.
5. **Result row** (`padx=18`, `pady=(0,14)`, left aligned): buttons
   `Open transcript` (teal 150), `Open folder` (secondary 120),
   `Try again` (secondary 110), each hidden independently; the row itself is
   shown when results are offered.
6. **Name row** (`padx=18`, `pady=(0,16)`, `#EEF1F7`, radius 10, hidden by
   default): `Name` label, entry, `Rename` button, note line.
7. **Best row** (`#EEF1F7`, radius 10, hidden by default):
   `Make best quality transcript` + note.
8. **Find speakers card** (white, 2 px amber border, radius 10, hidden by
   default).
9. **Naming host** (transparent, hidden by default) holding the speaker naming
   panel.

The card is `grid(row=3, padx=24, pady=(0,10), sticky="ew")`; it is not a
floating toast and does not cover content.

### 7.2 Status mapping

Internal worker status → visible title, text colour, progress colour, progress
mode:

| Internal status | Visible title | Text colour | Bar colour | Mode |
|---|---|---|---|---|
| `Ready` | `Ready` | `#55607A` | teal | pause |
| `Loading Model` | `Getting ready` | `#131A2B` | teal | busy |
| `Extracting Audio` | `Reading the audio` | `#131A2B` | teal | busy |
| `Transcribing` | `Transcribing` | `#075E5A` | teal | bar |
| `Identifying Speakers` | `Finding the speakers` | `#8A5300` | amber | busy |
| `Naming Speakers` | `Waiting for speaker names` | `#8A5300` | amber | pause |
| `Writing Transcript` | `Saving the transcript` | `#075E5A` | teal | bar |
| `Recording` | `Recording` | `#A3341C` | coral | busy |
| `Finishing` | `Finishing the last words` | `#A3341C` | coral | busy |
| `Saving Recording` | `Saving the recording` | `#A3341C` | coral | busy |
| `Cancelling…` | `Stopping…` | `#55607A` | faint | busy |
| `Done` | `Done` | `#16613A` | green | pause |
| `Cancelled` | `Stopped` | `#55607A` | faint | pause |
| `Failed` | `Something went wrong` | `#A61F35` | crimson | pause |
| unknown | the raw string | `#131A2B` | teal | pause |

The result handlers override the status title with the more specific texts in
§5.8, §6.7, §6.8 and §8.5. `Something went wrong` is usually only a transient
state while the finish handler runs.

**Status hints** (shown as an info banner only while not idle):

| Status | Hint |
|---|---|
| `Loading Model` | `Loading the speech model. This can take a minute.` |
| `Identifying Speakers` | `This step can take a while for long recordings.` |
| `Naming Speakers` | `Type each speaker's name below.` |

Any other status while busy removes the message banner. While idle, hints do not
appear and the handler leaves the banner alone.

### 7.3 Percent and progress

- Determinate progress is per **current file**, not across the batch. It is
  reset to 0 at the start of each file and at the start of every job.
- Percent text is `int(round(value × 100))` + `%`; only while
  `_ui_state == "running"`.
- `Transcribing`, `Writing Transcript`, and diarization progress are
  determinate. `Writing Transcript` initially shows `0%`.
- `Loading Model`, `Extracting Audio`, `Identifying Speakers` (until the child
  reports numbers), `Recording`, `Finishing`, `Saving Recording` and cancelling
  use the indeterminate animation and clear the percent.
- On every result the percent is cleared.
- Progress bar remains visible in results: green full when anything was saved,
  otherwise empty.
- `_stop_progress_motion` returns the bar to determinate and 0 when a status
  leaves busy mode.

### 7.4 Result actions

- `Open transcript` opens `last_transcript` with the OS default program if it
  still exists; otherwise it falls back to `Open folder` behaviour.
- `Open folder` opens `last_output_folder`, else the Files output folder, else
  the recordings folder. If that path is missing, a native info dialog:
  - title `No folder yet`
  - text `There is no folder to open yet.`
- `Try again` (or `Change`) only hides the result and unfolds the options /
  restores the Files action row. It never starts a job.
- Result buttons are hidden individually per the outcome tables.
- Opening any path uses `os.startfile` on Windows, `open` on macOS, `xdg-open`
  on Linux. Failure shows native error:
  - title `Could not open`
  - text `Could not open:\n{path}\n\n{exc}`

### 7.5 Name row

Appears after a result when there is a rename target (see §9). It is part of the
activity card and has an alt background:

- `Name` 13 bold ink, 14/10 px padding.
- Entry 36 high, radius 6, white, border `#C3CAD9`, 13 regular; prefilled with
  the current stem; placeholder (only visible if cleared):
  `Type a new name, for example Team meeting 21 Oct`.
- `Rename` teal 100 × 36, 13 bold, right of the entry.
- `Return` in the entry commits.
- Note line 12 muted, wrap 620, under the entry: mode-specific text in §9.
- The row does not appear for multi-file Files results, or when no target exists
  on disk.

### 7.6 Best row

Shown for a recording result when `_show_results(..., full=True)` is used, i.e.
after a live recording is saved, or after a non-fatal best-quality batch fails
or is stopped. It is not shown after a successful best-quality transcript, a
fatal missing-dependency failure, or Find speakers.

- `#EEF1F7` row, radius 10.
- `Make best quality transcript` teal 240 × 36, 13 bold, left.
- Note 12 muted, wrap 420, to the right:
  - if live text exists: `Adds times. More accurate. Your live transcript is kept. You can find the speakers after.`
  - otherwise: `Adds times. Most accurate. You can find the speakers after.`
- Clicking starts `_on_full_transcript` (§6.8). Only enabled while idle.

### 7.7 Showing, hiding and folding

- `_show_activity()`:
  - grids the card; hides the message, result row, name row, best row, speakers
    card;
  - clears batch line and percent;
  - folds the Files options and Record setup to summary lines;
  - hides the Files action row. It does not change the record step-3 row.
- `_hide_activity()`:
  - removes the card;
  - unfolds Files options and Record setup;
  - restores the Files action row.
- `Change` is visible on a folded summary only when idle. During work it is
  hidden.
- The folded **Files** summary is exactly one of these patterns:
  - `Timestamps on, speaker names on, saved next to each recording.`
  - `Timestamps off, speaker names on, saved in {folder shortened to 50}.`
  - `Timestamps on, speaker names off, saved next to each recording.`
  - `Timestamps off, speaker names off, saved in {folder shortened to 50}.`
  In general: `"{Timestamps on|Timestamps off}, {speaker names on|speaker names off}, {saved in <folder>|saved next to each recording}."`
  (Both option phrases are lowercase after the first `Timestamps ...`.)
- The folded **Record** summary is exactly:
  `"{microphone shortened to 50}. Show words as I speak."` or
  `"{microphone shortened to 50}. Best quality transcript after recording."`
- `_on_change_options()` only acts when idle and simply calls
  `_hide_activity()`.
- Adding new files in Files mode while idle also hides the activity card.
- Mode switches hide the card.
- `Show details` / `Privacy` are independent and remain available.

### 7.8 Enabled/disabled matrix

| Control / action | idle | running | recording |
|---|---|---|---|
| Mode switch | yes | no | no |
| Choose/Add files | yes | no | no |
| Drag/drop | yes | ignored | ignored |
| Clear all | yes | no | no |
| Remove row | yes | no | no |
| Timestamps checkbox | yes | no | no |
| Name speakers checkbox | yes if models present | no | no |
| Files output folder buttons | yes | no | no |
| Transcribe button | yes if files | `Transcribing…`, disabled | hidden (Files view) |
| Activity `Stop` | hidden | yes until clicked | hidden |
| Mic selector | yes | no | no |
| Record choice radios/tiles | yes | no | no |
| Recordings folder buttons | yes | no | no |
| Record button | `Start recording` | disabled | `Stop recording`; after Stop `Saving…`, disabled |
| `Open transcript` / `Open folder` / `Try again` | yes when result shown | hidden | hidden |
| Name entry / Rename | yes when result shown | hidden | hidden |
| Make best quality | yes when shown | hidden | hidden |
| Find speakers | yes when card shown | hidden | hidden |
| Show details / Privacy | yes | yes | yes |
| Privacy dialog controls | yes | yes | yes |

`_set_ui_state("idle")` re-renders the file list and restores all controls.
`_set_ui_state("running")` shows the activity Stop and sets the Transcribe
button to `Transcribing…`; `_set_ui_state("recording")` leaves the record button
enabled because it is the Stop control, but disables everything else.

---

## 8. Speaker features

EasyScribe has two speaker flows that share the same naming panel and visual
language but differ in when they run and what they write.

### 8.1 Files mode: `Name the speakers` option

- Checkbox in Step 2; amber family; availability checked once at startup from
  the presence of both diarization model files
  (`segmentation.onnx`, `embedding.onnx`).
- When ticked, the file pipeline for each input does:
  1. transcribe with the normal engine;
  2. identify speaker turns;
  3. assign each transcript segment to the speaker with the greatest time
     overlap;
  4. build an initial map `SPEAKER_00 → Speaker 1`, etc. in order of first
     appearance;
  5. compute each speaker's share of talking time;
  6. extract a sample clip for each speaker (the longest turn, truncated to
     5 s) into `temp/spk_clip_<raw_label>.wav`;
  7. show the naming panel and wait for the user;
  8. write the transcript with names into the normal output file
     `<input stem>.txt` (there is no extra copy in Files mode).
- If diarization fails for any reason (except user cancellation), the engine
  logs a warning and falls back to the transcript **without** speaker names;
  the file is still reported Done. The UI does not show a blocking error. This
  is deliberate resilience for a long batch.
- If the user cancels while the naming panel is open, the current file is
  marked `Stopped`, the batch stops, and no transcript is written for it.
- Speaker names are embedded in the file transcript; no `(speakers)` suffix is
  used in this flow.

### 8.2 Speaker naming panel

Shown **inside the main window**, in the activity card's naming host, not as a
popup. This was chosen because a popup could open behind the main window and
make the app look frozen. The panel is white, radius 10, with a 2 px amber
border. It appears at the bottom of the activity card and triggers a Windows
taskbar flash (without stealing focus) when opened.

Layout and copy:

- Title 18 bold ink:
  `We found {count} speaker` / `We found {count} speakers` (add `s` only when
  count ≠ 1).
- Body 13 muted, left, exactly two lines:
  `Play a short sample of each voice, then type the person's name.`
  `Leave a box empty to keep "Speaker 1", "Speaker 2" and so on.`
- Rows, one per speaker, in speaker-map order:
  - Row background `#EEF1F7`, radius 10, 1 px `#DCE1EC` border, 8 px side
    margin, 4 px vertical gap.
  - Left badge: default name (`Speaker 1`, `Speaker 2`, …), 13 bold
    `#8A5300` on `#FFF2D9`, radius 6, 92 × 28, padding `(12, 8)`.
  - `Play sample` secondary 110 × 32; if no clip was extracted, the label is
    `No sample` and the button is disabled.
  - Entry 34 high, radius 6, white, border `#C3CAD9`, placeholder
    `Type a name`; expands to fill the row; `Return` confirms the whole panel.
  - Share label `{round(share × 100)}% of the talking`, 12 muted, width 130,
    left aligned, when a share is known (the normal callback supplies it).
- If there are more than 4 speakers, the rows live in a scrollable area with
  height 4 × 62 = 248 px; otherwise a plain frame. The panel grows downward.
- Actions right aligned, 20 px sides, `(12, 18)` padding:
  - `Skip names` subtle 110 × 36;
  - `Save transcript with names` amber 230 × 36, 13 bold.
- First name entry gets focus after 100 ms.

Interaction:

- Typing a name and pressing `Return` anywhere in the panel saves **all**
  non-empty entries and closes the panel.
- `Save transcript with names`: for each row, a non-empty trimmed value
  overwrites that raw speaker's name; empty entries keep the default
  `Speaker N`. Then close and release the waiting worker.
- `Skip names`: close and release the worker without changing any names. Any
  typed-but-not-saved text is discarded.
- Play sample: Windows only, asynchronous `winsound.PlaySound` of the extracted
  clip. On other platforms the button is enabled but silent in the oracle;
  Rune should implement actual playback on all supported platforms. Clicking it
  repeatedly may overlap; there is no playing indicator. Playback is stopped
  when `Save transcript with names` or `Skip names` closes the panel (Windows);
  the cancellation path destroys the panel directly, so a sample already
  playing may finish.
- Temp clips are deleted immediately after the naming callback returns.

If the panel cannot be created for any reason, the queue is released
immediately so the worker never hangs; the transcript proceeds with the default
names.

### 8.3 Find speakers card

After a **recording** has a successful best-quality transcript, the activity
card can show an amber-bordered white card:

- Title `Who said what?` 15 bold ink.
- Body 13 muted, wrap 640:
  `Find the different voices in this recording, then type each person's name. The names are added to a copy of the transcript.`
- Estimate row: estimate text 13 bold `#8A5300`, e.g. `About 6 minutes`, plus a
  `rough estimate` pill: 12 `#8A5300` on `#FFF2D9`, radius 6, height 22,
  6 px horizontal padding.
- Basis line 12 muted, wrap 520:
  `Based on {audio duration} of audio, with {speech duration} of speech. You can keep using your computer and stop at any time.`
  Durations use `_fmt_minutes` (`under 1 min` or `N min`).
- `Find speakers` amber 170 × 40, 13 bold, right side spanning the estimate and
  basis rows.

Base offer conditions (all must hold on the initial offer after a best-quality
transcript):

- diarization models available;
- `last_recording`, `last_segments` and `last_transcript` all set;
- the recording is still present and has a readable WAV duration;
- the last successful batch was a recording batch (`_batch_recording` set) and
  the result was `Done` with at least one transcript saved.

It is offered after the auto best-quality transcript, and re-offered after a
cancelled/failed Find speakers run (the transcript and segments are still set,
so the Done requirement applies only to the initial offer). It is not offered
after a live transcript alone: the user must first make a best-quality
transcript. It is not offered for Files-mode results.

The estimate is a rough model:
`3.0 + 0.045 × audio_seconds + 0.165 × speech_seconds`, multiplied by a
per-PC speed factor learned from earlier runs (`settings` key
`speaker_speed_factor`, clamped 0.2–5.0, averaged with previous runs). Speech
seconds are the union of transcript segment ranges, capped at audio duration.

### 8.4 Find speakers flow

Preconditions in the click handler: idle, audio + segments + transcript all set.
If the audio is gone, show error banner `The recording was moved or deleted.`

1. Output path: `<transcript stem> (speakers).txt` in the transcript's folder.
   The original transcript is never modified.
2. Clear the log, set `running`, show the activity card, batch line =
   `{audio filename shortened to 80}`.
3. Start `SpeakersWorker` (daemon).
4. Worker calls `add_speakers_to_file(audio, segments, output, cancel_event,
   status, progress, log, speaker_name_callback)`.
   - Status `Identifying Speakers` (`Finding the speakers`, amber, busy;
     progress may become determinate from the child).
   - Status `Naming Speakers` (`Waiting for speaker names`, amber, pause) while
     the naming panel waits.
   - `Writing Transcript` (`Saving the transcript`), progress 1.0.
   - Timestamps are **always** added to this copy, regardless of whether the
     original transcript had timestamps.
5. Finish:
   - **Success**: `last_transcript` = the new copy; `last_output_folder` = its
     folder. Status `Speakers added` green. Detail
     `{output name shortened to 70} · took {duration}`. Info banner
     `The first transcript without names is kept too.` Progress green full.
     Buttons `Open transcript`, `Open folder`; Name box; no best row; speakers
     card hidden.
   - **Cancelled**: status `Stopped` muted; detail
     `No names were added. Your transcript is unchanged.`; no banner; progress
     0; `Open transcript`, `Open folder`; Name box; re-offer the speakers card.
   - **Failure**: status `Could not find the speakers` crimson; detail
     `Your transcript is unchanged.`; error banner `{error}\nSelect "Show details" to see why.`;
     progress 0; `Open transcript`, `Open folder`; Name box; re-offer the
     speakers card.
6. `took` is measured from just before diarization until the worker finishes,
   so it **includes time spent in the naming panel**.

### 8.5 Cancellation and failure in speaker flows

- The activity `Stop` button is visible during Find speakers (and during a
  Files batch that reaches `Naming Speakers`). Clicking it sets the shared
  cancel event and disables itself; status becomes `Stopping…`.
- The naming callback waits on a `done_event` in 200 ms slices. If the cancel
  event becomes set, it schedules closing the panel on the UI thread and raises
  `CancelledError` in the worker.
- File-mode diarization cancellation stops the entire batch (current file
  `Stopped`); the previously saved transcripts count as saved.
- Find-speakers cancellation/kill terminates the diarization child process and
  leaves the original transcript unchanged.
- File-mode diarization errors silently fall back to a plain transcript and
  continue; Find-speakers errors are surfaced as above.
- If no speaker turns are found, the naming panel is not shown (empty map); the
  transcript may fall back to raw `SPEAKER_00` labels. This is an edge case, not
  a designed UI state.

---

## 9. Rename flow (Name box)

### 9.1 Where it appears and what it targets

The Name row is added by `_show_results` whenever `_rename_targets()` returns a
target. It appears after the result buttons in the activity card.

Target selection:

- **Record mode** and `last_recording` exists:
  `recording_set(audio)` = every file in the recording's folder with suffix
  `.wav` or `.txt` whose stem is exactly the recording stem, or starts with
  `"<stem> ("`.
  This is the recording plus live/best/speakers transcript copies, sorted.
  Current stem = audio stem.
- Otherwise, `last_transcript` exists:
  targets = `[last_transcript]`; current stem = transcript stem.
- Otherwise, no Name row.

Note text:

| Situation | Note |
|---|---|
| Files mode / single transcript | `Renames the transcript. Your original file is not changed.` |
| Record mode with more than one target | `Renames the recording and its transcripts together.` |
| Record mode with one target | `Renames the recording.` |
| No target | row hidden |

The entry is prefilled with the current stem so the user can append a word
instead of retyping. The placeholder is only visible after clearing:
`Type a new name, for example Team meeting 21 Oct`.

Edge quirk: in Record mode, if `last_recording` is missing but a transcript
exists, `_rename_targets` falls back to that single transcript while the note
still selects the "one target" text `Renames the recording.`

### 9.2 Committing

- `Rename` teal 100 × 36, or `Return` in the entry.
- Only allowed when idle and a target exists.
- Name is trimmed of leading/trailing whitespace. It is a **stem**, not a full
  filename: for a recording, typing `foo.txt` produces `foo.txt.wav`; for a
  transcript it produces `foo.txt.txt`. The app does not strip a user-typed
  extension.
- Validation via `check_name`:
  - empty → `Type a name first.`
  - contains any of `\ / : * ? " < > |` or control characters →
    `A name cannot have these characters: \ / : * ? " < > |`
  - Windows reserved base name (`CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`,
    `LPT1`–`LPT9`, case-insensitive) or ends with `.` →
    `Windows does not allow this name. Please choose another.`
  - longer than 120 characters →
    `Please use a shorter name (at most 120 characters).`
- On a validation error, the note turns crimson with that text; nothing is
  changed.
- On success, `rename_all`:
  - plans each new path by replacing the leading old stem with the new stem,
    preserving the rest of the filename (` (best quality).txt`, etc.);
  - if any planned destination exists (and is not the same path), raises
    `FileExistsError` with the first taken filename **before** changing anything;
    UI shows `"{filename}" already exists in that folder. Please choose another name.`
  - renames all files; if an OSError occurs, undoes the renames already done and
    raises; UI shows
    `Could not rename. If the file is open in another program, close it and try again.`
  - returns a mapping old → new for every target (including unchanged ones).
- On success:
  - update `last_recording`, `last_transcript`, `_batch_recording` through the
    mapping;
  - note becomes `✓ Renamed to {name}` in green;
  - append log `[Rename] {old stem} -> {name} ({n} files)`.
- The Name row stays visible; the entry keeps the typed name. Renaming to the
  same stem does no physical move and still reports success.

### 9.3 Files mode versus Record mode

- **Files mode** renames only the produced `.txt`; the source audio/video is
  untouched. If the transcript is in a custom output folder, only that file is
  renamed.
- **Record mode** renames the `.wav` and every `.txt` variant in the same folder
  that shares the stem or a ` (<variant>)` suffix. This includes live, best
  quality and speakers copies.
- The rename is all-or-nothing; no partial set is left behind.
- The file list in Files mode is unaffected because it contains input media
  paths, not transcripts.

---

## 10. Privacy dialog

A separate non-resizable Toplevel, transient to the main window, opened by the
bottom-right `Privacy` button. It is not modal (no grab); work can continue
behind it. It is centred over the parent using the parent and dialog sizes,
clamped to non-negative coordinates. Title bar: `Your privacy`; app icon; close
via the OS, the `Close` button, or `Escape`. Focus is taken 50 ms after opening.
The dialog does not change app state.

Layout top to bottom (window background `#F4F6FB`):

1. Title `Your privacy`, 18 bold ink, 24 px sides, `(22, 4)`.
2. Green info box: `#16613A` text on `#E1F4E8`, radius 10, 13 regular, left
   aligned, 12 px internal x / 10 px internal y, 24 px side margins:
   `EasyScribe works fully offline. It never connects to the internet,`
   `and it blocks any attempt to do so. Nothing is uploaded, shared`
   `or tracked. There are no accounts, analytics or update checks.`
   (Three explicit lines in the oracle; let Rune reflow at equivalent width if
   needed, preserving the sentences.)
3. Section heading `Where your data is saved`, 13 bold ink.
4. White card with two-column rows; left label 13 bold ink, right body 13 muted,
   wrap 360. Rows:
   - `Transcripts` — `Next to each recording, or in the folder you choose.`
   - `Recordings` — `{DEFAULT_RECORDINGS_DIR}, or the folder you choose.`
     (e.g. `C:\Users\Alice\Documents\EasyScribe, or the folder you choose.`)
   - `Temporary audio` — `Deleted after each file, and again when EasyScribe closes.`
   - `Technical logs` — `{LOGS_DIR}` then a newline and
     `File names and progress only, never the transcript text. Only the last 10 are kept.`
5. Footer 13 muted, 24 px sides:
   `To remove everything, delete the transcripts you made and the`
   `EasyScribe folder. Nothing is stored anywhere else.`
   (Explicit line break after `the` in the oracle.)
6. Actions right aligned 24 px sides, `(16, 22)`:
   - `Open logs folder` secondary 150 × 36 — opens `LOGS_DIR` with the OS;
     failure shows the standard `Could not open` error.
   - `Close` teal 100 × 36.

**Stale screenshot warning:** `docs/screenshots/privacy.png` shows an older row
label `Live recordings` and an older path under `recordings/`. The current code
uses `Recordings` and `DEFAULT_RECORDINGS_DIR` (`Documents/EasyScribe`). The
code strings above are authoritative.

---

## 11. Details panel and technical log

### 11.1 Toggle

- Bottom utility row, left: `Show details` subtle button 110 × 30, 12 pt.
- Click toggles:
  - open: log card is placed in main-grid row 5 with 24 px side margins and
    bottom 16; row 5 weight 1; button text becomes `Hide details`;
  - closed: card removed; row 5 weight 0; button text returns to `Show details`.
- The details state is independent of mode, activity, privacy and job state. It
  remains open until toggled.
- The toggle is wrapped in the screen-change fade.

### 11.2 Log card

- White card, 1 px `#DCE1EC`, radius 14.
- Textbox: read-only, 11 mono (`Consolas`), muted `#55607A`, white fill, no
  border, word wrap, min height 140, 10 px inner padding; grows with the row.
- Auto-scrolls to the end after every append.
- Starts empty at launch and is **not** persisted.
- This is an in-memory job log. It is separate from the rotating technical log
  files in `logs/`; the Privacy panel describes the latter.

### 11.3 Content and line formats

The log is cleared at the start of a file batch, a recording and a Find speakers
run. Rename appends without clearing. Representative lines (exact prefixes):

- `[File {i}/{n}] {filename}` preceded by a line of 50 `─` characters.
- `[Record] Starting live recording…`
- `[Record] Starting recording (best quality transcript after the stop)…`
- `[Record] Listening. Select Stop recording when done`
- `[Done] Transcript saved: {path}`
- `[Done] Audio saved: {path}`
- `[Warning] Could not save to {dest}. The recording is in {parent}`
- `[Warning] transcription falling behind. Some audio was skipped in the live view; the full recording is still saved`
- `[Fatal] {error}`
- `[Error] {error}` / `[Error] {filename}: {error}`
- `[Error] Could not write transcript: {error}` (live transcript save failure)
- `[Unexpected Error] {filename}: {error}`
- `[Cancelled] Operation stopped by user`
- `[Summary] Completed with {N} error(s) out of {M} file(s)`
- `[Rename] {old stem} -> {new stem} ({n} files)`

Engine/ffmpeg/diarizer lines are appended through the same callback:
`[Transcribe] …`, `[ffmpeg] …`, `[Diarize] …`, `[Output] Writing transcript to: …`,
`[Progress] segment …`, raw decoded segment text, and so on. The log may
therefore contain transcript text even though the technical log files do not.
`Select "Show details" to see why.` in error banners refers to this panel.

---

## 12. Filesystem outputs and naming rules

### 12.1 Paths and constants

| Name | Value |
|---|---|
| `BASE_DIR` | folder containing the exe (frozen) / project root (dev) |
| `RECORDING_WORK_DIR` | `BASE_DIR/recordings` |
| `SETTINGS_FILE` | `BASE_DIR/recordings/settings.json` |
| `DEFAULT_RECORDINGS_DIR` | `Documents/EasyScribe` (known Documents folder) |
| `TEMP_DIR` | `BASE_DIR/temp` |
| `LOGS_DIR` | `BASE_DIR/logs` |
| `MAX_LOG_FILES` | 10 |
| `MIN_FREE_DISK_BYTES` | 512 MB |
| Supported media | `.mp4 .mkv .mov .avi .webm .mp3 .wav .m4a .flac .ogg .opus .aac` |

### 12.2 Output naming

| Action | File(s) created | Name rule | Overwrite behaviour |
|---|---|---|---|
| Files batch, any options | Transcript | `<input stem>.txt` in `output_folder` or next to input | Overwrites an existing file silently |
| Files batch with speaker names | Same transcript | `<input stem>.txt`; names embedded | Same |
| Record best-quality (record only) | WAV + best transcript | `recording_YYYYMMDD_HHMMSS.wav`; `<stem>.txt` or `<stem> (best quality).txt` if base exists | WAV timestamp; best suffix only avoids the base collision |
| Record live | WAV + live transcript | same WAV stem; `<stem>.txt` (one line per segment) | Timestamp collision would append to an existing PCM; normal run has a unique second |
| Manual `Make best quality transcript` after live | New transcript | `<stem>.txt` if no live txt, else `<stem> (best quality).txt` | A pre-existing `(best quality).txt` is overwritten silently |
| Find speakers | Speaker copy | `<transcript stem> (speakers).txt` | Existing copy with that name is overwritten silently; original kept |
| Rename | Renames only | old stem → new stem, preserving ` (variant)` suffixes | Blocked before any change if a destination exists |
| Crash recovery | WAV | `recording_<timestamp>.wav` in chosen/default recordings folder | No existing file is deleted on "No" |
| Settings | `settings.json` | JSON keys `recordings_dir` (user choice) and `speaker_speed_factor` (internal); whole file rewritten | Small direct rewrite, no atomic temp-file swap |
| Logs | `EasyScribe_YYYYMMDD_HHMMSS.log` | in `logs/` | Oldest removed when at 10 |

Important duplicate/collision notes:

- Ordinary Files-mode output has **no overwrite confirmation** and no suffix.
  Two inputs with the same stem sent to one custom output folder overwrite each
  other; both still count as saved, so the result can say `2 transcripts ready`
  while only one file exists, and `last_transcript` is `None`.
- Best-quality collision protection is exactly one suffix level:
  `<stem>.txt` → `<stem> (best quality).txt`; if that also exists, it is
  overwritten.
- Find speakers always overwrites an existing `(speakers)` copy; the first
  transcript is preserved.
- Rename is the only operation that refuses collisions and is all-or-nothing.
- `recording_set()` includes `.wav` and `.txt` only; it does not include
  `.pcm`, `.json`, logs, or unrelated extensions.
- Recording stems use local time `%Y%m%d_%H%M%S`, e.g.
  `recording_20261021_143207`.

### 12.3 Transcript content (for UI expectations and tests)

- Plain file transcript: decoded segments joined with one space; single
  paragraph; UTF-8, LF newlines, one trailing newline.
- `Add timestamps`: blocks split when a segment starts **2.0 s or more** after
  the previous segment's end. Each block:
  `[HH:MM:SS]` newline, then the block text; blocks separated by one blank line
  (`\n\n`); one trailing newline. Hours are zero-padded to two digits.
- Diarized without timestamps: runs of the same speaker become
  `[Name]` newline text; blocks separated by a blank line.
- Diarized with timestamps: a new block starts on speaker change or a ≥2.0 s
  gap; format `[HH:MM:SS] [Name]` newline text.
- Best-quality and Find-speakers transcripts always have timestamps; Find
  speakers always adds names.
- Live transcript: plain decoded segment lines, no timestamps/speakers, one
  line per segment, trailing newline.
- Speaker clips are temporary: `temp/spk_clip_<raw label>.wav`, deleted when the
  naming callback returns (success, skip or cancel).

### 12.4 Temporary files and cleanup

- Each input file is extracted to `temp/<uuid>.wav`, then deleted in the
  per-file `finally` block.
- Speaker clips are deleted immediately after naming.
- `temp/*.wav` and `temp/*.json` are also removed at startup and at exit by
  `main.py`/`atexit`.
- Recording sidecars (`.pcm` + `.json`) live in `BASE_DIR/recordings` until a
  successful WAV conversion; if the app closes/crashes first, they remain for
  recovery.
- No transcript or recording is ever uploaded or copied elsewhere.

---

## 13. State machine, concurrency and lifecycle

### 13.1 State variables

| Variable | Values / meaning |
|---|---|
| `_ui_state` | `"idle"`, `"running"`, `"recording"` |
| `_mode` | `"Transcribe files"` or `"Record"` |
| `_selected_files` | ordered unique list of input media paths |
| `_file_states` | `Path → Waiting / Working / Done / Failed / Stopped` |
| `_output_folder` | custom Files transcript folder, or `None` = next to each input |
| `_rec_folder` | remembered recordings folder, or `None` = `DEFAULT_RECORDINGS_DIR` |
| `_cancel_event` | shared cancellation flag for file batch / find speakers / namer wait |
| `_stop_recording_event` | graceful "stop recording and save" flag |
| `_last_output_folder` | folder of the last saved output |
| `_last_transcript` | last saved transcript path, or `None` |
| `_last_recording` | last saved WAV, or `None` |
| `_batch_recording` | audio being transcribed by the current/last recording batch |
| `_last_segments` | timed transcript segments of the last recording, for Find speakers |
| `_naming_panel` | current embedded speaker panel, if any |
| `_output_overrides` | input path → explicit output path (best-quality naming) |
| `_details_open` | details log visibility |
| `_live_has_text` | whether the live box has received real text |
| `_rec_started_at` | monotonic recording start, or `None` |
| `_visible_rows` | 2 or 3, fixed at launch from screen height |
| `_diarization_available` | models present at startup |
| `_dnd_enabled` | native DnD initialised |
| `_progress_running` | indeterminate animation state |

Defaults at launch: Files mode, empty file list, no custom transcript folder,
recordings folder from settings if valid, timestamps off, speaker names off,
best-quality record mode, `Default microphone`, activity hidden, details
closed, status `Ready`.

No other choices persist between launches. Window size, selected mode,
microphone, record mode, timestamps, diarization, details and file list are all
reset.

### 13.2 Transitions

| Event | Guard | Effects | New state |
|---|---|---|---|
| App launch | dependencies OK | build UI, render empty Files, status Ready, DnD/device/diarization probe | idle |
| Choose/Add files or valid drop | idle, Files mode | add unique supported paths; hide result and prune states; render | idle |
| Remove / Clear all | idle | update list/states; Clear hides result; render | idle |
| Change output folder / Use default | idle | update label; no persistence | idle |
| Change/reset recordings folder | idle | update label; persist/remove setting | idle |
| Click Transcribe | idle, ≥1 file | optional low-disk confirm; snapshot options; clear log; states=Waiting; start worker; show activity; fold options; hide action row | running |
| Click Stop (activity) | running | set cancel; status `Stopping…`; disable Stop | running (cancelling) |
| Batch worker ends | running | set idle; store last transcript/folder; result title/detail/message/buttons; progress; maybe offer rename/speakers | idle (+result) |
| Click record mode tile | idle | set radio, update tile colours, show/hide live card | idle |
| Click Start recording | idle | clear stop event; clear log; set listening placeholder; show activity; fold setup; button=Stop; start RecordingWorker | recording |
| Click Stop recording | recording | set stop event; button `Saving…`, disabled | recording (saving) |
| Recording worker ends | recording | reset button/timer; set idle; show saved/error result; best mode auto-starts transcription | idle (+result) or running |
| Auto best quality starts | idle after recording | `_start_batch([wav])`, timestamps on; batch_recording=wav | running |
| Click Make best quality | idle, recording exists | resolve output, set override, start batch | running |
| Click Find speakers | idle, audio+segments+transcript | resolve `(speakers)` path; clear log; show activity; start worker | running |
| Naming panel Save/Skip | panel open | mutate map or not; close; release worker | running |
| Click Stop while naming | running | cancel event; panel closes; worker raises Cancelled | running (cancelling) |
| Speakers worker ends | running | idle; result; maybe re-offer speakers card | idle (+result) |
| Click Rename / Return | idle, target | validate; rename atomically or show error; update paths | idle |
| Click Change / Try again | idle, result visible | hide activity; unfold options; restore action row | idle |
| Click mode switch | idle | show other view; hide activity; unfold options/setup; selected colour changes | idle |
| Show/Hide details | any | toggle log card and row weight | unchanged |
| Open/Close Privacy | any | create/destroy dialog | unchanged |
| Close window | any | destroy root; daemon workers end; PCM remains for recovery | process exit |

`running` is used for both Files transcription and Find speakers. `recording`
covers both best-quality and live recording; the sub-phase is conveyed by the
record button text, timer, status and log.

### 13.3 Threads

| Worker | Type | Starts when | Job |
|---|---|---|---|
| UI main loop | main | launch | all widget access, dialogs, timers, callbacks |
| `TranscriptionWorker` | daemon thread | Transcribe / best quality | ffmpeg extraction + transcription (+ diarization/naming) per file, sequential |
| `RecordingWorker` | daemon thread | Start recording | optional model load, mic start, wait for stop, live pipeline, WAV/transcript write |
| `PCMWriter` | daemon thread | mic start | append raw int16 PCM, fsync every 50 chunks, update JSON sidecar |
| `LiveVAD` | daemon thread | live recording | Silero VAD, onset padding, segment queue |
| `LiveDecoder` | daemon thread | live recording | decode queued segments, call `on_segment` |
| `SpeakersWorker` | daemon thread | Find speakers | diarize, assign, naming panel, write `(speakers).txt` |
| Diarization child | spawned process | diarization | sherpa-onnx pipeline, progress messages, can be terminated |
| Timer tick | `after(500 ms)` | recording start | update timer until `_rec_started_at` is `None` |
| Level tick | `after(100 ms)` | recording start | read/reset peak, update meter until stopped |

The engine lazily loads the Whisper model under a lock (shared by file and live
paths, but only one job runs at a time in the UI); the first load can take a
minute and shows `Getting ready`.

### 13.4 Thread-safe UI updates and ordering

- Workers must never touch Tk widgets directly. All UI changes go through
  `self.after(0, callback)` wrappers:
  `_safe_update_status`, `_safe_set_progress`, `_safe_append_log`,
  `_safe_set_batch_label`, `_safe_set_file_state`.
- Closures capture the value at call time (default arguments) so late binding
  does not show the wrong status.
- `_safe_set_file_state` updates the state map and re-renders the whole file
  list on the UI thread.
- Finalisation always goes through `after(0, _finish_batch / _finish_recording /
  _finish_speakers)`. The worker schedules the final status update first, then
  the finish callback; both run in order on the UI thread.
- The naming callback is the only synchronous handshake: the worker schedules
  the panel creation and then blocks in `done_event.wait(0.2)` slices, checking
  `_cancel_event` each slice. UI actions (Save/Skip/cancel) close the panel and
  set the event.
- The batch holds a snapshot of the file list; adding/removing is disabled
  while running, so the worker does not see mutations.
- `_output_overrides` is written before the batch starts and only read by the
  worker.
- No mutex protects UI state because only the main thread reads/writes it.
  Cross-thread communication is through `threading.Event` and `after`.

### 13.5 Cancellation semantics

- **Files:** `_cancel_event` is polled by ffmpeg (process terminated), by the
  transcription loop between segments, by whisper.cpp wrapper, and by the
  diarization parent (child process terminated). The current file becomes
  `Stopped`; the loop breaks; earlier `saved` files remain valid. Remaining
  files stay `Waiting`.
- **Find speakers:** same cancel event; the diarization child is terminated; no
  new file is written (the write happens after all processing); original
  transcript stays. The naming panel closes if open.
- **Recording:** Stop is a graceful stop, not a cancel. There is no discard
  control. The captured audio is finalised (or left as recoverable PCM if the
  WAV conversion fails), never discarded. Clicking Stop during model loading
  sets the event, so capture starts and stops immediately.
- The `Stop` activity button disables itself after one click. There is no
  force-quit path in the UI.
- Pressing cancel after the last transcript write may still produce a
  `Stopped` result with saved files; saved count is respected.
- The Name box only commits when idle; there is no background rename. The
  speaker naming panel is the one modal-like handshake and commits while its
  job is paused waiting for it.

### 13.6 Close, recovery and startup errors

**Close:** see §2.6. No confirmation, no save-on-close in the Python oracle.
Temporary extraction files are cleaned at exit/startup; an unfinished PCM+JSON
pair is preserved and recovered next launch.

**Crash-recovery prompt** (before the main GUI is created, one prompt per
orphaned `recording_*.pcm` that has a matching `.json`):

- Native yes/no dialog, title `Unfinished Recording Found`, body:

```text
An unfinished recording was found:

  File: {pcm_file_name}
  Duration: {Xm Ys}

Recover it as a .wav file now?
(Choosing No keeps the raw recording for next time.)
```

- `Yes`: convert the raw int16 PCM (sample rate/channels/samples written from
  the JSON) to `recording_<timestamp>.wav` in the remembered recordings folder,
  or the default if none; if that folder fails, save next to the raw file. On
  success delete the `.pcm` and `.json`, then show an info dialog:
  - title `Recovery Complete`
  - text `Saved to:\n{wav_path}`
- `No`: leave both files untouched so the prompt can appear next launch. This
  is intentional: declining must never delete the only copy.
- Conversion failure: error dialog title `Recovery Failed`, text
  `Could not recover {pcm_file_name}:\n{exc}`; continue with the next orphan.
- Duration uses whole minutes and seconds: `f"{int(sec//60)}m {int(sec%60)}s"`.
- If recovery itself fails to run, the app logs a warning and continues to the
  GUI.

**Startup dependency error** (in `main.py`, before the GUI):

- Check bundled `ffmpeg.exe`, `ffprobe.exe`, and the Whisper/VAD model files.
- On any missing piece, create a withdrawn root and show a native error:
  - title `EasyScribe cannot start`
  - text `Some program files are missing. Please download EasyScribe again.\n\n`
    followed by the collected errors separated by blank lines.
- Then exit with status 1; no main window is created.
- The collected error blocks are:
  - `ffmpeg.exe not found.\nExpected at: {FFMPEG_BIN}\n\nThe bundled ffmpeg folder may be missing.`
  - `ffprobe.exe not found.\nExpected at: {FFPROBE_BIN}`
  - `Model files missing or incomplete:\n` followed by one `  • {error}` line per missing model file.
- This is distinct from a missing model discovered later: that becomes a fatal
  file-batch result or a recording error.
- Recovery runs before this dependency check.

---

## 14. Copy deck and formatting helpers

This section is the master checklist of visible strings. Strings already shown
in earlier sections are repeated here only where useful for coverage.

### 14.1 Window and chrome

| Context | Exact text |
|---|---|
| Window title | `EasyScribe 3.2.3` (`f"{APP_NAME} {APP_VERSION}"`) |
| Header | `EasyScribe` |
| Mode 1 | `Transcribe files` |
| Mode 2 | `Record` |
| Bottom left | `Show details` / `Hide details` |
| Bottom right | `Privacy` |
| Privacy title bar | `Your privacy` |

### 14.2 Transcribe files

| Context | Exact text |
|---|---|
| Step 1 title | `Add recordings` |
| Count | `1 file`, `2 files`, …; empty string at 0 |
| Clear | `Clear all` |
| Drop title, empty, DnD | `Drag audio or video files here` |
| Drop subtitle, empty, DnD | `Or choose them from your computer. MP3, WAV, M4A, MP4 and more.` |
| Drop title, empty, no DnD | `Choose the recordings to transcribe` |
| Drop subtitle, empty, no DnD | `MP3, WAV, M4A, MP4 and other audio and video files.` |
| Drop title, files, DnD | `Drag more files here` |
| Drop subtitle (set but hidden when files exist), DnD | `Or add more from your computer.` |
| Drop title, files, no DnD | `Choose the recordings to transcribe` |
| Drop button, empty | `Choose files` |
| Drop button, files | `Add files` |
| Row action | `Remove` |
| Row size fallback | `Not found` |
| Step 2 title | `Choose options` |
| Folded link | `Change` |
| Output label | `Save transcripts to` |
| Output default | `The same folder as each recording` |
| Output reset | `Use default` |
| Output chooser | `Change folder` |
| Timestamp option | `Add timestamps` |
| Timestamp body | `Shows the time, like [00:01:23], at the start of each paragraph.` |
| Speaker option | `Name the speakers` |
| Speaker body | `Finds who is talking. You hear a short sample of each voice and type their name. Takes longer.` |
| Speaker unavailable | `Not available: the speaker files are missing from this copy of EasyScribe.` |
| Start, empty | `Transcribe` |
| Start, files | `Transcribe 1 file` / `Transcribe {N} files` |
| Start, running | `Transcribing…` |
| Hint, empty | `Add a recording to start.` |
| Hint, defensive | `Add at least one recording first.` |
| File statuses | `Waiting`, `Working`, `Done`, `Failed`, `Stopped` |
| File dialog title | `Choose recordings to transcribe` |
| File dialog filters | `Audio and video files`; `All files` |
| Transcript folder title | `Choose where to save transcripts` |
| Skipped note | `Skipped {names}{more}. EasyScribe reads audio and video files: {extensions}.` |

### 14.3 Record

| Context | Exact text |
|---|---|
| Step 1 title | `Choose a microphone` |
| Default device | `Default microphone` |
| Step 2 title | `What do you want?` |
| Best title | `Best quality transcript after recording` |
| Best body | `Saves the recording.` / `Adds times. Most accurate. Find speakers after.` |
| Live title | `Show words as I speak` |
| Live body | `Saves the recording.` / `Quick text while you talk. Less accurate.` |
| Folder label | `Save recordings to` |
| Folder reset | `Use default` |
| Folder chooser | `Change folder` |
| Folder dialog title | `Choose where to save recordings` |
| Step 3 title | `Record` |
| Record idle | `Start recording` |
| Record active | `Stop recording` |
| Record saving | `Saving…` |
| Timer idle | `00:00` |
| Live card title | `Live transcript` |
| Live idle placeholder | `Select Start recording. The words appear here as people speak.` |
| Live listening placeholder | `Listening. The words appear here as people speak.` |
| Folded summary, best | `{mic, shortened}. Best quality transcript after recording.` |
| Folded summary, live | `{mic, shortened}. Show words as I speak.` |
| Record log start, live | `[Record] Starting live recording…` |
| Record log start, best | `[Record] Starting recording (best quality transcript after the stop)…` |
| Listening log | `[Record] Listening. Select Stop recording when done` |

### 14.4 Activity, progress and results

| Context | Exact text |
|---|---|
| Status Ready | `Ready` |
| Status Loading | `Getting ready` |
| Status extraction | `Reading the audio` |
| Status transcribing | `Transcribing` |
| Status diarizing | `Finding the speakers` |
| Status naming | `Waiting for speaker names` |
| Status writing | `Saving the transcript` |
| Status recording | `Recording` |
| Status finishing | `Finishing the last words` |
| Status saving recording | `Saving the recording` |
| Status stopping | `Stopping…` |
| Status done | `Done` |
| Status cancelled | `Stopped` |
| Status failed | `Something went wrong` |
| Hint loading | `Loading the speech model. This can take a minute.` |
| Hint diarizing | `This step can take a while for long recordings.` |
| Hint naming | `Type each speaker's name below.` |
| Stop button | `Stop` |
| Result buttons | `Open transcript`, `Open folder`, `Try again` |
| Name label | `Name` |
| Name placeholder | `Type a new name, for example Team meeting 21 Oct` |
| Rename button | `Rename` |
| Best button | `Make best quality transcript` |
| Best note, live kept | `Adds times. More accurate. Your live transcript is kept. You can find the speakers after.` |
| Best note, otherwise | `Adds times. Most accurate. You can find the speakers after.` |
| Success, one | `Transcript ready` |
| Success, many | `{N} transcripts ready` |
| Partial | `{S} of {T} transcripts ready` |
| Failure title | `Transcription failed` |
| Fatal title | `EasyScribe could not continue` |
| Fatal detail | `A program file is missing. Please download EasyScribe again.` |
| Saved detail | `Saved in {path}` |
| Partial message | `Could not transcribe {names}{more}. The file may be damaged or have no sound. Select "Show details" to see why.` |
| Cancelled, saved | `1 transcript saved before you stopped.` / `{N} transcripts saved before you stopped.` |
| Cancelled, none | `No transcript was saved.` |
| Record saved | `Recording saved` |
| Record saved, audio only | `The audio is in {path}.` |
| Record saved, live transcript | `The audio and the live transcript are in {path}.` |
| Record saved, no speech | `No speech was heard live, so there is no live transcript. The audio is in {path}.` |
| Record failure | `Could not record` |
| No folder info | `No folder yet` / `There is no folder to open yet.` |
| Open failure | `Could not open` / `Could not open:\n{path}\n\n{exc}` |

### 14.5 Speakers

| Context | Exact text |
|---|---|
| Naming title | `We found {N} speaker` / `We found {N} speakers` |
| Naming body line 1 | `Play a short sample of each voice, then type the person's name.` |
| Naming body line 2 | `Leave a box empty to keep "Speaker 1", "Speaker 2" and so on.` |
| Play | `Play sample` |
| No clip | `No sample` |
| Name placeholder | `Type a name` |
| Share | `{N}% of the talking` |
| Skip | `Skip names` |
| Save | `Save transcript with names` |
| Find card title | `Who said what?` |
| Find card body | `Find the different voices in this recording, then type each person's name. The names are added to a copy of the transcript.` |
| Estimate badge | `rough estimate` |
| Find button | `Find speakers` |
| Find success | `Speakers added` |
| Find detail | `{file, shortened} · took {duration}` |
| Find success banner | `The first transcript without names is kept too.` |
| Find cancelled | `No names were added. Your transcript is unchanged.` |
| Find failure | `Could not find the speakers` |
| Find failure detail | `Your transcript is unchanged.` |
| Find failure banner | `{error}\nSelect "Show details" to see why.` |
| Estimate basis | `Based on {audio} of audio, with {speech} of speech. You can keep using your computer and stop at any time.` |
| Estimate values | `less than a minute`, `about 1 minute`, `about N minutes`, `about N hours`, `about N hours M minutes` |

### 14.6 Rename

| Context | Exact text |
|---|---|
| Files note | `Renames the transcript. Your original file is not changed.` |
| Recording set note | `Renames the recording and its transcripts together.` |
| Recording note | `Renames the recording.` |
| Empty | `Type a name first.` |
| Bad characters | see exact string below |
| Reserved/trailing dot | `Windows does not allow this name. Please choose another.` |
| Too long | `Please use a shorter name (at most 120 characters).` |
| Taken | `"{filename}" already exists in that folder. Please choose another name.` |
| OS error | `Could not rename. If the file is open in another program, close it and try again.` |
| Success | `✓ Renamed to {name}` |
| Log | `[Rename] {old} -> {new} ({n} files)` |

Exact bad-character message (outside the table so the pipe is unambiguous):

```text
A name cannot have these characters: \ / : * ? " < > |
```

### 14.7 Privacy and details

| Context | Exact text |
|---|---|
| Privacy heading | `Your privacy` |
| Offline box | `EasyScribe works fully offline. It never connects to the internet,` / `and it blocks any attempt to do so. Nothing is uploaded, shared` / `or tracked. There are no accounts, analytics or update checks.` |
| Section | `Where your data is saved` |
| Transcripts | `Next to each recording, or in the folder you choose.` |
| Recordings | `{DEFAULT_RECORDINGS_DIR}, or the folder you choose.` |
| Temporary audio | `Deleted after each file, and again when EasyScribe closes.` |
| Technical logs | `{LOGS_DIR}` + newline + `File names and progress only, never the transcript text. Only the last 10 are kept.` |
| Footer | `To remove everything, delete the transcripts you made and the` / `EasyScribe folder. Nothing is stored anywhere else.` |
| Open logs | `Open logs folder` |
| Close | `Close` |
| Details closed (button label) | `Show details` |
| Details open (button label) | `Hide details` |

### 14.8 Dialogs, errors, recovery

| Context | Exact text |
|---|---|
| Low disk title | `Low disk space` |
| Low disk body | `Only {N} MB of disk space is free.\nTranscription may fail. Continue anyway?` |
| Open error | `Could not open` / `Could not open:\n{path}\n\n{exc}` |
| No folder | `No folder yet` / `There is no folder to open yet.` |
| Recovery title | `Unfinished Recording Found` |
| Recovery body | `An unfinished recording was found:\n\n  File: {name}\n  Duration: {Xm Ys}\n\nRecover it as a .wav file now?\n(Choosing No keeps the raw recording for next time.)` |
| Recovery complete | `Recovery Complete` / `Saved to:\n{path}` |
| Recovery failed | `Recovery Failed` / `Could not recover {name}:\n{exc}` |
| Startup title | `EasyScribe cannot start` |
| Startup body prefix | `Some program files are missing. Please download EasyScribe again.\n\n` |
| Mic failure | `Could not record: {exc}\nCheck that a microphone is connected, and that Windows allows apps to use it (Settings, Privacy, Microphone).` |
| Model failure | `A program file is missing. Please download EasyScribe again.\n{exc}` |
| Moved audio | `The recording was moved or deleted.` |
| Live save failure | `Could not save the transcript: {exc}` |
| No names fallback | (silent; see §8.1) |

### 14.9 Dynamic templates and pluralisation

| Function / template | Rule |
|---|---|
| `_plural(n, word)` | `f"{n} {word}"` if `n == 1`, else `f"{n} {word}s"` |
| File count | `_plural(count, "file")` |
| Start button | `f"Transcribe {_plural(count, 'file')}"` |
| Transcript count | one saved → `Transcript ready`; otherwise `f"{len(saved)} transcripts ready"` |
| Saved before stop | `f"{_plural(len(saved), 'transcript')} saved before you stopped."` |
| Speaker count | `f"We found {count} speaker{'s' if count != 1 else ''}"` |
| File batch label | `f"File {i} of {n}: {shortened}"` or `shortened` when n=1 |
| File log header | `f"[File {i}/{n}] {name}"` |
| Percent | `f"{int(round(value * 100))}%"` |
| Timer | `f"{h}:{m:02d}:{s:02d}"` if h else `f"{m:02d}:{s:02d}"` |
| Share | `f"{round(share * 100)}% of the talking"` |
| Size | `bytes` no decimals; `KB`/`MB`/`GB` one decimal; divide by 1024 |
| Minutes | `<60 s` → `under 1 min`; else `f"{round(sec/60)} min"` |
| Took | `f"{m} min {s} s"` if m else `f"{s} s"` |
| Estimate | exact algorithm in §8.3; first letter uppercased for display |
| Path shortening | `_shorten(text, limit)` middle ellipsis (see below) |
| Failure names | up to 3 shortened to 40, joined `, `, plus ` and N more` if >3 |
| Skipped names | same as failure names |
| Saved path | shortened to 90 |
| Batch filename | shortened to 70 (multi) / 80 (single) |
| Current recording label | shortened to 80 |
| File row | filename 60 / parent 60 |
| Output labels | 70 |
| Options summary folder | 50 |
| Mic summary | 50 |
| Recorded-where path | 60 |
| Find-speakers output name | 70 |
| Speaker share width | 130 px |

`_shorten(text, limit)`:

```text
if len(text) <= limit: return text
keep = limit - 1
head = keep * 2 // 3
return text[:head] + "…" + text[-(keep - head):]
```

It keeps the start (including the drive/first words) and the tail (including
the extension) visible.

`_fmt_size(bytes)`:

```text
for unit in ("bytes", "KB", "MB", "GB"):
    if size < 1024 or unit == "GB":
        return f"{size:.0f} {unit}" if unit == "bytes" else f"{size:.1f} {unit}"
    size /= 1024
```

`_fmt_minutes(seconds)`:

```text
return "under 1 min" if seconds < 60 else f"{round(seconds / 60)} min"
```

`_fmt_took(seconds)`:

```text
minutes, secs = divmod(int(round(seconds)), 60)
return f"{minutes} min {secs} s" if minutes else f"{secs} s"
```

`fmt_estimate(seconds)` (then capitalise first character for the Find card):

```text
if seconds < 50: return "less than a minute"
minutes = round(seconds / 60)
if minutes <= 1: return "about 1 minute"
if minutes < 60: return f"about {minutes} minutes"
hours, rest = divmod(minutes, 60)
rest = 5 * round(rest / 5)
if rest == 60: hours, rest = hours + 1, 0
hours_text = "1 hour" if hours == 1 else f"{hours} hours"
return f"about {hours_text}" + (f" {rest} minutes" if rest else "")
```

### 14.10 Type and punctuation rules

- U+2026 ellipsis in `Transcribing…`, `Stopping…`, `Saving…`,
  `Cancelling…`, `[Record] Starting live recording…`, and shortened paths.
- U+00B7 middot in `{size}  ·  {parent}` and `{file} · took {time}`.
- U+2713 in `✓ Renamed to {name}`.
- U+2500 in the 50-character log separator.
- U+2022 in startup dependency error bullets (`  • Missing: …`).
- Straight ASCII apostrophes and quotes; no curly quotes.
- No exclamation marks, em dashes or en dashes in designed UI copy; engine log
  lines may contain their own punctuation.
- Sentence case for headings and buttons, except product names and
  `EasyScribe`.
- Full stops at the end of sentences and result details; button labels have no
  trailing punctuation.

---

## 15. Edge cases and quirks to preserve (or consciously improve)

These are deliberate behaviours or low-level consequences of the oracle. A
Rune implementation should either preserve them or document an approved polish
change.

1. **No result history.** Switching tabs or pressing Change/Try again hides the
   result permanently; there is no way to reopen it except producing a new one.
2. **Multi-file results hide Open transcript and Rename.** They show only Open
   folder (and Try again on failure), because `last_transcript` is `None` when
   more than one file is saved.
3. **Try again is not a retry.** It only returns to the options/action row; the
   user must press Transcribe again.
4. **Adding a file clears result pills.** In idle Files mode, all file states
   except `Waiting` are dropped and the result card is hidden.
5. **Clear all has no confirmation** and keeps options/output settings.
6. **Remove has no confirmation**; removing a Done row does not clear the
   visible result.
7. **No per-row percentage.** Every row gets a status pill, but only the
   current file says `Working`; numeric progress appears only in the global
   activity card.
8. **Progress resets per file**, not cumulative across a batch.
9. **Output collisions overwrite silently** for ordinary transcripts and the
   `(best quality)`/`(speakers)` suffix targets; only Rename refuses collisions.
10. **Best quality gives exactly one suffix level.** `<stem> (best quality).txt`
    is overwritten if it already exists.
11. **Find speakers always writes a copy** and always adds timestamps, even if
    the original transcript had none.
12. **Find speakers cannot run after live-only recording** until a best-quality
    transcript exists.
13. **Find speakers timing includes user naming time**, because the stopwatch
    spans the panel wait.
14. **Files-mode diarization failure is silent** in the UI: the transcript is
    written without names and the batch still finishes Done. The details log has
    a warning, not a banner.
15. **Naming panel has no Cancel button**; Skip names is the only "leave names
    as they are" action. Stop in the activity card cancels the whole job.
16. **Empty naming boxes keep defaults**; typed values are applied only on Save
    transcript with names (or Return).
17. **Speaker sample playback is Windows-only in the oracle**; Rune should play
    on all supported OSes.
18. **The timer keeps the last duration** after stop, in faint text; it resets
    only when a new recording starts.
19. **The live box retains its last content** after a live recording (transcript
    or the `Listening…` placeholder). A successful live recording with no
    speech leaves the listening placeholder visible rather than returning to the
    idle placeholder; the next Start resets it.
20. **The level meter resets silently** after stop and is not a recording
    indicator by itself.
21. **Best-mode recording has a large blank spacer** below the record card when
    no live transcript card is shown, because the Record view claims all
    remaining height. This is visible in `transcript-ready.png`. Rune may
    tighten it only if the result card placement stays recognisably the same.
22. **Record mode output folder is independent** of the Files transcript folder
    and is the only persisted choice.
23. **The microphone list is built once** at launch; plugging in a device later
    requires a restart.
24. **Speaker models missing** disables the option and shows the unavailable
    copy; there is no install action.
25. **Startup model/ffmpeg check aborts the app** before the GUI, with a native
    error; missing models discovered later become fatal batch/recording errors.
26. **A fatal batch leaves later files as Waiting**, not Failed, because the
    worker breaks. The result title is "EasyScribe could not continue".
27. **Fatal with no saved file** may leave an empty result row area; Rune should
    avoid the blank strip while preserving the buttons/copy.
28. **Cancelled with saved files** shows a green full progress bar, even though
    the status title is "Stopped".
29. **Rename treats the field as a stem**, so for a recording, typing
    `name.txt` produces `name.txt.wav` (for a transcript, `name.txt.txt`); do
    not silently strip an extension without documenting it.
30. **Rename records every target**, including unchanged ones, and logs the
    count; renaming to the same name reports success.
31. **Rename set is prefix-based**: any `.wav`/`.txt` whose stem is the exact
    stem or starts with `"<stem> ("` is included, even unrelated variants.
32. **Details and Privacy stay enabled during work**; Privacy is non-modal, so
    users can read it while a job runs.
33. **Details log includes transcript text**, while the technical log files do
    not; the Privacy copy refers to the files, not the on-screen details panel.
34. **Results are not persisted**; closing the window loses the in-memory
    result, though output files remain.
35. **No dark mode, no settings dialog, no onboarding, no tray icon.**
36. **The privacy screenshot in docs is stale** (`Live recordings` vs current
    `Recordings`); use current code strings.
37. **The file picker allows All files** but unsupported choices are filtered and
    reported in the red skipped note.
38. **Drop note names are limited to three**, with `and N more`; extension list
    is alphabetical uppercase.
39. **No file is opened by clicking its row**; only the result buttons open
    outputs.
40. **Changing mode hides results but keeps the file list and settings.**
41. **Closing during a recording has no warning**; recovery is the safety net.

---

## 16. Rune visual translation and allowed polish

### 16.1 Must remain identical

- All labels, sentences, punctuation, capitalisation and ordering in §14.
- Two-mode IA and every step/order within cards.
- Feature colour meanings and status colour mapping.
- Exact window minimum, 24 px gutter, 18 px card padding, 12 px card gap.
- Palette hex values, radii 6/10/14, control heights 36/44, 8 px progress.
- All condition/state logic, enabled/disabled rules, cancel and recovery
  semantics.
- Which files are created and their naming rules.
- No network, no telemetry, no accounts, light theme only.

### 16.2 Recommended Rune polish

These are visual-only and may be applied without changing IA:

- **Typography:** Segoe UI Variable Display for 22/18/16 titles if it matches
  Segoe UI metrics; Segoe UI for body. Keep sizes and weights.
- **Cards:** retain white fill and 1 px border; add a subtle ambient shadow
  behind top-level cards. Do not turn cards into heavy neumorphic surfaces.
- **Segmented control:** animate the selected pill with a 120–160 ms spring;
  keep the exact teal/coral selected colours and white label.
- **Buttons:** 36/44 px heights, 10 px radius; 120 ms hover/press transitions;
  visible 2 px focus ring; no layout shift on press.
- **Progress:** smooth width interpolation for determinate changes; a subtle
  shimmer for indeterminate; do not use striped bars. Keep the 8 px height and
  rounded ends.
- **Drag-over:** softly animate the border/background to teal-soft; no scale of
  the whole card. A small drop icon may be added beside the title if it does not
  alter text or layout.
- **Screen changes:** 120–160 ms fade/slide for mode switch, activity/result
  appearance, details, naming panel; honour reduced motion.
- **Lists:** 46 px rows, 3 px row padding (6 px between rows), auto-hide
  scrollbar, gentle hover on Remove; keep status pill and metadata positions.
- **Speaker panel:** fade/slide in from the activity card; focus first entry;
  animate the share label subtly; keep the amber 2 px border.
- **Privacy dialog:** native modal-looking window with the same content; no
  changes to wording or row order. A soft scrim behind it is allowed only if it
  remains non-blocking to the work behind (or make it modal and document the
  change; the oracle is transient, not modal).
- **Icons:** use the EasyScribe waveform mark from `assets/`; no generic icons
  for the brand. Small functional icons (folder/open) may be added only if they
  do not replace the text labels.
- **Focus/keyboard:** add focus rings, tab order matching visual order, and
  `Alt`/accelerator access if Rune supports it; do not add global shortcuts that
  conflict with text fields.

### 16.3 Suggested Rune component mapping

| Oracle component | Rune treatment |
|---|---|
| `CTk` root | Window with fixed light theme and DPI-aware logical layout |
| Header + logo | Flex row; vector/PNG brand mark 40 × 40 |
| `CTkSegmentedButton` | Two-segment control, animated selected pill |
| `CTkFrame` card | Rounded surface + 1 px border + optional shadow |
| Step badge | 26 px circular badge with soft feature fill |
| `CTkButton` variants | Button styles: primary teal/coral/amber, secondary, subtle, link |
| `CTkScrollableFrame` file list | Virtualized list or scroll area, auto-hide scrollbar, 46 px rows |
| `CTkOptionMenu` | Native-looking select/dropdown; same fills and fonts |
| `CTkCheckBox` / `CTkRadioButton` | Custom-painted 22/20 px controls with feature colours |
| `CTkProgressBar` | 8 px rounded track/fill, determinate + indeterminate |
| `CTkTextbox` read-only | Read-only scrollable text with auto-scroll |
| `CTkToplevel` Privacy | Dialog window with exact copy and centring |
| `tkinterdnd2` | Rune native drag/drop; same filtering and skipped note |
| `filedialog` | Native file/folder pickers; same titles/filters |
| `messagebox` | Native confirm/error/info dialogs; same strings |
| `winsound` speaker samples | Rune audio playback (rodio or platform API), async |
| `after` timers | Rune scheduler/animation clock; 500 ms timer, 100 ms meter |

### 16.4 Things not to add

- Dark mode or a theme switch.
- A start/onboarding wizard, account, telemetry, update check, network call.
- A new settings screen, recent-files list, transcript preview, editor, or
  search.
- A modal confirmation for Clear all, Remove, Rename overwrite, or window close.
- Toasts that replace the inline status/message/result card.
- A drag-and-drop overlay that changes the file list layout.
- Automatic retry loops.
- A progress bar that aggregates the whole batch (the oracle is per file).
- Renaming the product, modes, steps or buttons.

---

## 17. Acceptance checklist

Use this as the parity test list for the Rune UI.

### Window and chrome

- [ ] Title exactly `EasyScribe 3.2.3`.
- [ ] 900 logical px wide; height clamp 600–820 from screen height; min 760×600.
- [ ] 2 file rows on 1366×768, 3 on taller screens; resizable.
- [ ] Light theme only; `#F4F6FB` background; logo 40 px, teal tile, white bars.
- [ ] Segmented switch exactly `Transcribe files` / `Record`; teal selected in
  Files, coral in Record; white selected text, ink unselected.
- [ ] Mode switch blocked while running/recording; activity hidden on switch.
- [ ] No close confirmation; close leaves unfinished PCM recoverable.

### Design tokens

- [ ] Every hex in §3.1 matches; no stray theme blue from the base theme.
- [ ] Type sizes 22/18/16/15/14/13/12/11; Segoe UI/Consolas.
- [ ] Radii 6/10/14, 26 px step badge, 8 px progress, 36/44 px buttons.
- [ ] 24 px gutter, 18 px card padding, 12 px card gaps, 2 px special borders.

### Files

- [ ] Empty copy and disabled Transcribe/hint exactly.
- [ ] Choose files native dialog title/filter; supported extension filtering.
- [ ] Add files changes copy, count, Clear all, list height.
- [ ] File row name/meta/status/Remove; metadata size formatting; `Not found`.
- [ ] Drag/drop hover colours, valid filtering, skipped note wording/extensions.
- [ ] Remove/Clear all; no confirmation; busy disabling.
- [ ] Options expanded and folded summaries exact; Change only idle.
- [ ] Output default/custom labels, Use default visibility, folder picker.
- [ ] Add timestamps and Name the speakers visuals; unavailable copy/disabled.
- [ ] Transcribe N files; low-disk confirmation; options snapshot.
- [ ] Running: per-file statuses, current file line, percent, Stop, disabled
  controls, folded options.
- [ ] All result variants in §5.8, including button visibility, progress,
  message wording and multi-file suppression.
- [ ] Cancel leaves remaining files Waiting; current Stopped; saved files
  counted.
- [ ] Fatal behavior and details hint.

### Record

- [ ] Mic selector default first, devices de-duplicated/48-char shortened.
- [ ] Both choice tiles exact copy, default best, whole-tile click, selected
  coral tint/border.
- [ ] Save recordings to default/custom, persistence, Use default, native picker.
- [ ] Start/Stop/Saving labels; timer format/retained faint; level meter coral
  and idle reset.
- [ ] Best mode auto best-quality transcript after Stop; no intermediate saved
  card.
- [ ] Live card show/hide; idle/listening placeholders; segments append/scroll.
- [ ] Live saved card copy for transcript/no transcript; Name and best rows.
- [ ] Make best quality availability and note variants; output naming.
- [ ] Record error card and copy; no result buttons on total failure.

### Activity and results

- [ ] Status mapping table, colours, bar modes, percent rules.
- [ ] Hints appear only when non-idle and clear on other statuses.
- [ ] Activity Stop only during running; record Stop in record row.
- [ ] Result buttons per outcome; Open transcript fallback; No folder info.
- [ ] Name row prefilled, Return, note copy; best row copy.
- [ ] Folding/unfolding, action row hide/restore, Change/Try again behavior.

### Speakers

- [ ] Files option works end-to-end; naming panel embedded, not popup.
- [ ] Naming panel count/plural/body/badges/play/share/entry/actions exact.
- [ ] Empty = default, Return saves all, Skip discards, first focus.
- [ ] >4 speaker scroll height; sample playback/deletion.
- [ ] Find speakers card conditions, estimate/basis, button.
- [ ] Find speakers progress, Stop, naming panel, `(speakers).txt` output.
- [ ] Success/cancel/failure result copy and re-offer behavior.
- [ ] File diarization failure falls back silently to plain.
- [ ] Total timing includes naming wait.

### Rename, privacy, details, outputs

- [ ] Target rules for Files vs Record; notes exact.
- [ ] Validation strings; collision refusal; atomic rollback; success checkmark
  and log.
- [ ] Privacy dialog content, paths, buttons, Escape, centring, stale screenshot
  ignored.
- [ ] Details toggle labels, log content, auto-scroll, monospace.
- [ ] Every output path/name in §12; no unintended overwrites beyond documented
  ones.
- [ ] Temp cleanup; settings persistence; log rotation limit.

### Concurrency and recovery

- [ ] No widget access off the UI thread.
- [ ] Cancel event stops ffmpeg/transcription/diarization, keeps saved files.
- [ ] Data races absent; file states update on the UI thread only.
- [ ] UI enabled/disabled matrix exact in all states.
- [ ] Close/worker shutdown; recovery prompt exact; No preserves files; Yes
  writes WAV and deletes sidecars.
- [ ] Startup dependency failure dialog exact and no main window.

### Visual QA

- [ ] Side-by-side with `files.png`, `done.png`, `transcribing.png`,
  `recording.png`, `recording-saved.png`, `transcript-ready.png`,
  `name-speakers.png` at the same logical size.
- [ ] No clipped labels at 760×600, 900×678, 900×820, and 200% scaling.
- [ ] All long paths/names ellipsize in the middle, retaining extensions.
- [ ] No accidental dark surfaces, pure-black text, or base-theme blue.
- [ ] Focus rings visible; keyboard operation of all buttons/entries.
- [ ] Reduced-motion mode has no fades/slides but identical state changes.

---

## Appendix A — End-to-end walkthroughs

### A.1 Files, three files, success

1. Launch: Files empty, `Transcribe` disabled, hint `Add a recording to start.`
2. Choose three files (or drop): count `3 files`, drop becomes one line
   `Drag more files here`, button `Add files`, `Clear all` appears.
3. Tick `Add timestamps` and `Name the speakers` (if available).
4. Press `Transcribe 3 files`.
5. Activity appears; options fold to
   `Timestamps on, speaker names on, saved next to each recording.`; action row
   disappears; file states Waiting.
6. `Reading the audio`, then `Transcribing`, `Finding the speakers`, name panel
   if speakers found, `Saving the transcript`; percent updates per file; Stop
   available.
7. Each row becomes Done; on completion activity says `3 transcripts ready`,
   detail `Saved in /path`, green full bar, only `Open folder`; no Name box
   because multiple.
8. `Show details` reveals the per-file log; `Privacy` remains available.

### A.2 Files, partial failure, cancel

1. Start three files.
2. File 1 Done, file 2 Failed; activity continues to file 3.
3. Finish: `2 of 3 transcripts ready` amber, message naming file 2, buttons
   `Open folder`, `Try again`; no Open transcript (two saved).
4. Press `Try again`: options expand, action row returns; result hidden.
5. Start again and press Stop during file 2: file 1 Done, current Stopped,
   file 3 Waiting; result `Stopped`, `1 transcript saved before you stopped.`,
   `Open transcript`, `Open folder`, `Try again`.

### A.3 Record, best quality, then speakers

1. Record tab defaults to Best; `Start recording`.
2. Timer starts, level moves; activity shows coral Recording.
3. Stop: button `Saving…`; status `Saving the recording`; WAV saved.
4. Best transcript starts automatically: record summary folded; activity
   `Reading the audio` → `Transcribing` → `Saving the transcript`.
5. Success: `Transcript ready`, `Saved in ...`, `Open transcript`,
   `Open folder`, Name box, `Who said what?` card with estimate and
   `Find speakers`.
6. Press `Find speakers`: activity `Finding the speakers`, Stop visible,
   naming panel opens with samples/shares.
7. Type names, press `Save transcript with names`: status `Speakers added`,
   detail `... (speakers).txt · took 6 min 12 s`, info banner, Open transcript/
   folder, Name box. Original transcript remains.

### A.4 Record, live, then best quality

1. Choose `Show words as I speak`; `Live transcript` card appears; Start.
2. Words appear line by line while speaking; timer and level run.
3. Stop: `Finishing the last words` (flush), then `Saving the recording`;
   WAV + live `.txt` saved.
4. Result `Recording saved`; detail says audio and live transcript; buttons
   `Open transcript`, `Open folder`; Name; `Make best quality transcript`.
5. Press best: new file `<stem> (best quality).txt` (live `.txt` kept);
   activity runs; success `Transcript ready`; Find speakers card appears.
6. Find speakers writes `<stem> (speakers).txt` based on the best transcript.

### A.5 Rename

1. After a result, Name box shows the current stem.
2. Edit and press Return/Rename.
3. In Record mode all `.wav`/`.txt` variants sharing the stem are renamed
   together; in Files mode only the transcript.
4. Invalid name → note turns crimson, nothing moves. Taken name → quoted
   collision message. Success → green `✓ Renamed to ...` and log line.

### A.6 Close during recording / recovery

1. Start recording, then close the window with the title-bar X.
2. No prompt; the app exits; PCM + JSON remain in `BASE_DIR/recordings`.
3. Relaunch: before the main window, `Unfinished Recording Found` asks to
   recover, showing file and duration.
4. Yes → WAV in chosen/default recordings folder, delete sidecars, show
   `Recovery Complete`. No → files remain and the prompt returns next launch.

---

## Appendix B — Screenshot map

All images are in `docs/screenshots/` relative to the repository root (this
document lives in `docs/rewrite/`, so links use `../screenshots/…`).

| Image | State captured | Notes |
|---|---|---|
| `../screenshots/files.png` | Files populated, 3 files, both options ticked, empty result | 900×820 reference |
| `../screenshots/transcribing.png` | Files running, 42%, file 2 of 3, one Done/Working/Waiting | 900×820 |
| `../screenshots/done.png` | Files success, 3 transcripts, options folded, result | 900×820 |
| `../screenshots/recording.png` | Record default best, mic, both choices, save folder, idle timer | 900×820 |
| `../screenshots/recording-saved.png` | Live recording saved, live transcript, Name, best row | 900×820 |
| `../screenshots/transcript-ready.png` | Best-quality transcript ready, Name, Who said what? | 900×820; shows the Record flex spacer |
| `../screenshots/name-speakers.png` | Speaker naming panel close-up (3 speakers) | Crop 828×356 |
| `../screenshots/privacy.png` | Privacy dialog | 542×494; **older copy/path**, see §10 |

Use the screenshots for spacing, colour and hierarchy. Use `gui.py` for all
strings, conditions and edge cases.

---

## Appendix C — Source map for implementers

| Concern | Python source |
|---|---|
| Palette, fonts, dimensions, layout | `src/gui.py` (class `C`, `_build_*`) |
| Window sizing/DPI | `TranscriberApp.__init__` |
| Files flow | `_build_files_view`, `_on_*`, `_transcription_worker`, `_finish_batch` |
| Record flow | `_build_record_view`, `_on_record`, `_on_stop_recording`, `_recording_worker`, `_finish_recording`, `_on_full_transcript` |
| Activity/status/progress | `_build_activity_panel`, `_update_status`, `_set_progress`, `_show_results` |
| Speakers | `SpeakerNamingPanel`, `_offer_speakers`, `_on_find_speakers`, `_speakers_worker`, `_finish_speakers` |
| Rename | `src/rename.py`, `_offer_rename`, `_on_rename` |
| Privacy | `PrivacyDialog` |
| Details/log | `_build_details`, `_toggle_details`, `_append_log` |
| Output naming | `_resolve_output_path`, `_on_full_transcript`, `_on_find_speakers`, `src/mic_recorder.py` |
| Settings | `src/settings.py` |
| Recovery/startup | `src/main.py`, `src/recovery.py` |
| Motion | `src/transitions.py` |
| Transcript formatting | `src/transcriber.py` (`_build_plain_transcript`, `_build_diarized_transcript`, `_write_transcript`) |
| Estimate math/copy | `src/diarizer.py` |

**End of specification.**



