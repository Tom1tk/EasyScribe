#!/usr/bin/env python3
"""
Generate assets/splash.png, the "Loading" window of the EasyScribe .exe.

The PyInstaller bootloader shows this image (see Splash() in
launcher/launcher.spec) while it unpacks the app, before any Python code
runs. That takes up to a minute on the first start, so the image says so.

Colours match the C palette in src/gui.py. The font is Roboto from the
customtkinter package, so the result is the same on every machine.
Needs Pillow and customtkinter (dev machine only):
    python assets/make_splash.py
"""
import importlib.util
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from make_icon import render

WIDTH, HEIGHT = 440, 150
SCALE = 4  # supersample, then downscale for smooth text and edges

SURFACE = (0xFF, 0xFF, 0xFF)
BORDER = (0xDC, 0xE1, 0xEC)
TEAL = (0x0B, 0x7A, 0x75)
INK = (0x13, 0x1A, 0x2B)
MUTED = (0x55, 0x60, 0x7A)
OUT = Path(__file__).parent / "splash.png"


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    # find_spec only locates the package; importing it needs tkinter
    spec = importlib.util.find_spec("customtkinter")
    path = Path(spec.origin).parent / "assets" / "fonts" / "Roboto" / name
    return ImageFont.truetype(str(path), size * SCALE)


def main() -> None:
    w, h = WIDTH * SCALE, HEIGHT * SCALE
    img = Image.new("RGB", (w, h), SURFACE)
    draw = ImageDraw.Draw(img)

    # Border and a teal accent line at the top
    draw.rectangle((0, 0, w - 1, h - 1), outline=BORDER, width=SCALE)
    draw.rectangle((0, 0, w - 1, 4 * SCALE), fill=TEAL)

    logo_size = 64
    logo = render(logo_size * SCALE)
    img.paste(logo, (28 * SCALE, 36 * SCALE), logo)

    x = (28 + logo_size + 20) * SCALE
    draw.text((x, 36 * SCALE), "EasyScribe", font=_font("Roboto-Medium.ttf", 26), fill=INK)
    draw.text((x, 74 * SCALE), "Getting ready…", font=_font("Roboto-Regular.ttf", 16), fill=MUTED)
    draw.text(
        (28 * SCALE, 116 * SCALE),
        "The first start can take a minute. Please wait.",
        font=_font("Roboto-Regular.ttf", 13),
        fill=MUTED,
    )

    img.resize((WIDTH, HEIGHT), Image.LANCZOS).save(OUT)
    print(f"Wrote {OUT.name}")


if __name__ == "__main__":
    main()
