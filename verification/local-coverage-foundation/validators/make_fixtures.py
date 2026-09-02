#!/usr/bin/env python3
"""Generate deterministic local fixtures for the OCR and screen-grounding gates.

Everything is drawn from code, so the ground truth is exact and the gates need no
downloaded corpus, no network and no third-party sample data.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path("/home/typhoon/git/frankenstein-llm/verification/local-coverage-foundation/fixtures")

DOC_LINES = [
    "INCIDENT REPORT 4172",
    "Pool zroot completed a recovery scrub on 2026-08-29.",
    "Result: repaired 0B with 0 errors; no known data errors.",
]
TABLE_HEADER = ["Device", "Read", "Write", "Cksum"]
TABLE_ROWS = [
    ["nvme0n1", "0", "0", "0"],
    ["ROCm1", "0", "0", "0"],
]

BUTTONS = [
    {"label": "Restart Router", "box": [72, 132, 292, 188]},
    {"label": "Stop Downloads", "box": [332, 132, 552, 188]},
    {"label": "Open Evidence", "box": [72, 232, 292, 288]},
    {"label": "Clear Cache", "box": [332, 232, 552, 288]},
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    raise SystemExit("no scalable font found; install dejavu or noto before generating fixtures")


def build_document() -> dict:
    image = Image.new("RGB", (1000, 620), "white")
    draw = ImageDraw.Draw(image)
    title = load_font(30)
    body = load_font(21)
    mono = load_font(19)

    draw.text((48, 40), DOC_LINES[0], font=title, fill="black")
    draw.text((48, 96), DOC_LINES[1], font=body, fill="black")
    draw.text((48, 132), DOC_LINES[2], font=body, fill="black")

    top, left, row_height = 210, 48, 42
    widths = [220, 130, 130, 130]
    for row_index, row in enumerate([TABLE_HEADER, *TABLE_ROWS]):
        x = left
        y = top + row_index * row_height
        for column_index, cell in enumerate(row):
            draw.rectangle([x, y, x + widths[column_index], y + row_height], outline="black", width=2)
            draw.text((x + 12, y + 10), cell, font=mono, fill="black")
            x += widths[column_index]

    path = OUT / "ocr-document.png"
    image.save(path)
    return {
        "path": str(path),
        "lines": DOC_LINES,
        "table_header": TABLE_HEADER,
        "table_rows": TABLE_ROWS,
    }


def build_screen() -> dict:
    image = Image.new("RGB", (900, 420), "#1d2027")
    draw = ImageDraw.Draw(image)
    heading = load_font(26)
    label = load_font(20)

    draw.rectangle([0, 0, 899, 72], fill="#2b303b")
    draw.text((36, 24), "Local AI Control Panel", font=heading, fill="#f4f4f5")
    for button in BUTTONS:
        x0, y0, x1, y1 = button["box"]
        draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill="#3b82f6")
        text_box = draw.textbbox((0, 0), button["label"], font=label)
        draw.text(
            (x0 + (x1 - x0 - text_box[2]) / 2, y0 + (y1 - y0 - text_box[3]) / 2),
            button["label"],
            font=label,
            fill="#ffffff",
        )
    draw.text((72, 330), "Router: idle    Queue: running", font=label, fill="#a1a1aa")

    path = OUT / "screen-grounding.png"
    image.save(path)
    return {"path": str(path), "size": list(image.size), "buttons": BUTTONS}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = {"document": build_document(), "screen": build_screen()}
    (OUT / "ground-truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    print(json.dumps({k: v["path"] for k, v in truth.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
