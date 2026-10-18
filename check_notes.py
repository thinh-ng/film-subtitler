"""Render each note's time on screen with libass and report whether the note ever moves.

Usage: python check_notes.py work/<slug> [work/<slug> ...]   (needs ffmpeg with libass on PATH)
Reads <slug>.subs.json and <slug>.bible.json, as written by translate.py.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import translate

WIDTH, HEIGHT, FPS = 1280, 720, 10


def note_tops(srt_path, start, end, workdir):
    """Top row of the highest text on screen, per frame from start to end."""
    raw = os.path.join(workdir, "frames.raw")
    # The subtitles filter only sees absolute timestamps, so render from 0 and trim after it.
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"color=black:s={WIDTH}x{HEIGHT}:r={FPS}:d={end + 1}",
                    "-vf", f"subtitles={os.path.basename(srt_path)},trim=start={start}:end={end}",
                    "-pix_fmt", "gray", "-f", "rawvideo", raw], cwd=workdir, check=True)
    frames = np.fromfile(raw, np.uint8).reshape(-1, HEIGHT, WIDTH)
    tops = []
    for img in frames:
        rows = np.flatnonzero((img > 100).sum(axis=1))
        tops.append(int(rows[0]) if len(rows) else None)
    return tops


def main():
    moved = 0
    with tempfile.TemporaryDirectory() as workdir:
        for prefix in sys.argv[1:]:
            with open(f"{prefix}.subs.json", encoding="utf-8") as f:
                lines = json.load(f)
            with open(f"{prefix}.bible.json", encoding="utf-8") as f:
                notes = {n["line"]: n for n in json.load(f)["notes"]}
            srt = os.path.join(workdir, "check.srt")
            translate.write_srt(srt, lines, "vi", notes)
            for note in translate.build_note_cues(lines, "vi", notes):
                # A frame's edge from the note's start and end is skipped: either cue may own it.
                tops = note_tops(srt, note["start"] + 0.1, note["end"] - 0.1, workdir)
                steady = len(set(tops)) == 1 and tops[0] is not None
                moved += not steady
                print(f"{os.path.basename(prefix)} {translate.srt_time(note['start'])[:8]} "
                      f"note top rows {sorted(set(map(str, tops)))}: {'steady' if steady else 'MOVES'}")
    sys.exit(1 if moved else 0)


if __name__ == "__main__":
    main()
