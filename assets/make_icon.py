#!/usr/bin/env python3
"""
Generate assets/EasyScribe.ico and assets/EasyScribe.png from the logo design.

The geometry matches _logo() in src/gui.py: a teal rounded tile with five
white sound-wave bars. Needs Pillow (dev machine only, not bundled):
    python assets/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw

TEAL = (0x0B, 0x7A, 0x75, 255)
WHITE = (255, 255, 255, 255)
HEIGHTS = (0.22, 0.46, 0.62, 0.38, 0.18)
ICO_SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
OUT = Path(__file__).parent


def render(size: int) -> Image.Image:
    big = size * 8  # supersample, then downscale for smooth edges
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((0, 0, big - 1, big - 1), radius=big * 0.22, fill=TEAL)
    bar_w = big * 0.09
    gap = big * 0.07
    total = len(HEIGHTS) * bar_w + (len(HEIGHTS) - 1) * gap
    x = (big - total) / 2
    mid = big / 2
    for h in HEIGHTS:
        half = max(big * h / 2, bar_w / 2)
        draw.rounded_rectangle(
            (x, mid - half, x + bar_w, mid + half), radius=bar_w / 2, fill=WHITE
        )
        x += bar_w + gap
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    frames = [render(s) for s in ICO_SIZES]
    frames[-1].save(
        OUT / "EasyScribe.ico", format="ICO",
        sizes=[(s, s) for s in ICO_SIZES], append_images=frames[:-1],
    )
    frames[-1].save(OUT / "EasyScribe.png")
    print("Wrote EasyScribe.ico and EasyScribe.png")


if __name__ == "__main__":
    main()
