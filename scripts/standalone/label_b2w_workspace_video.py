"""Overlay Stage-20b target labels and build a compact review contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


SEGMENTS = (
    (0.0, 2.0, "Settling / 初始化落稳"),
    (2.0, 7.5, "HOME / 初始位姿"),
    (7.5, 13.0, "FRONT +1.00 m / 前向 1.00 米"),
    (13.0, 18.5, "REAR -0.80 m / 后退 0.80 米"),
    (18.5, 24.0, "HIGH +0.25 m + PITCH / 高位与俯仰"),
    (24.0, 29.5, "LOW -0.25 m + PITCH / 低位与俯仰"),
    (29.5, 35.0, "DIAGONAL RIGHT ~0.99 m / 右前对角约 1 米"),
    (35.0, 40.5, "HOME / 回到中性位姿"),
    (40.5, 46.0, "LARGE 6D POSE / 大幅度六维位姿"),
    (46.0, 51.5, "HOME / 回到中性位姿"),
    (51.5, 57.0, "DIAGONAL LEFT ~0.99 m / 左前对角约 1 米"),
    (57.0, 62.5, "LEFT +0.75 m / 左侧 0.75 米"),
)


def segment_at(time_s: float) -> tuple[float, float, str]:
    for start, end, label in SEGMENTS:
        if start <= time_s < end:
            return start, end, label
    return SEGMENTS[-1]


def draw_label(frame, time_s: float, font: ImageFont.FreeTypeFont):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    start, end, label = segment_at(time_s)
    line = f"{label}    t={time_s:05.1f}s"
    box = draw.textbbox((0, 0), line, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x, y = 28, 25
    draw.rounded_rectangle(
        (x - 14, y - 10, x + width + 14, y + height + 14),
        radius=10,
        fill=(0, 0, 0, 175),
    )
    draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
    progress = max(0.0, min(1.0, (time_s - start) / max(end - start, 1.0e-6)))
    bar_y = y + height + 5
    draw.rectangle((x, bar_y, x + width, bar_y + 4), fill=(100, 100, 100, 220))
    draw.rectangle((x, bar_y, x + int(width * progress), bar_y + 4), fill=(62, 205, 255, 255))
    return cv2.cvtColor(__import__("numpy").asarray(image), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to create {args.output}")
    font = ImageFont.truetype(str(args.font), 27)
    sheet_times = [2.0, 10.0, 15.5, 21.0, 26.5, 32.0, 43.0, 48.5, 54.0, 59.2]
    sheet_frames: list[Image.Image] = []
    sheet_indices = {min(frame_count - 1, round(value * fps)) for value in sheet_times}

    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        labeled = draw_label(frame, index / fps, font)
        writer.write(labeled)
        if index in sheet_indices:
            preview = Image.fromarray(cv2.cvtColor(labeled, cv2.COLOR_BGR2RGB))
            preview.thumbnail((480, 270))
            sheet_frames.append(preview.copy())
        index += 1
    capture.release()
    writer.release()

    if args.contact_sheet and sheet_frames:
        columns = 2
        rows = (len(sheet_frames) + columns - 1) // columns
        cell_width = max(image.width for image in sheet_frames)
        cell_height = max(image.height for image in sheet_frames)
        sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "black")
        for item, image in enumerate(sheet_frames):
            sheet.paste(image, ((item % columns) * cell_width, (item // columns) * cell_height))
        args.contact_sheet.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(args.contact_sheet, quality=92)

    print(f"frames={index} fps={fps:.3f} output={args.output}")


if __name__ == "__main__":
    main()
