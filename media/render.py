#!/usr/bin/env python3
"""Turn a `script(1)` capture of a real bobbin run into an animated GIF.

Not a mockup and not a re-enactment: the frames are the bytes the agent actually
wrote to the terminal, replayed with the timings `script -T` recorded. The only
liberties taken are speed (long model pauses are dull) and a cap on idle gaps.

The capture is append-only with seven SGR codes and no cursor addressing, so a
line renderer reproduces it exactly rather than approximating it.
"""
import html, re, subprocess, sys
from pathlib import Path

DEMO = Path(sys.argv[1])
COLS, ROWS = 100, 26
CELL_W, CELL_H = 8.4, 18       # DejaVu Sans Mono at 14px
FONT_SIZE = 14
PAD = 14
SPEED = 2.6                    # replay multiplier
IDLE_CAP = 0.9                 # seconds any single pause may occupy, after speedup
FPS = 10

BG = "#12131a"
FG = "#c9d1d9"
COLOURS = {31: "#f47067", 32: "#6bc46d", 33: "#e3b341", 36: "#56b6c2",
           2: "#7d8590", 1: "#e6edf3"}


def read_capture():
    """(delay, text) pairs, from script's typescript + timing file."""
    raw = (DEMO / "session.log").read_bytes()
    raw = raw.split(b"\n", 1)[1]          # drop the "Script started" banner
    events, pos = [], 0
    for line in (DEMO / "timing.log").read_text().splitlines():
        delay, count = line.split()
        count = int(count)
        events.append((float(delay), raw[pos:pos + count]))
        pos += count
    return events


SGR = re.compile(rb"\x1b\[([0-9;]*)m")


class Screen:
    """A wrapping, scrolling grid of (char, colour) — no cursor addressing needed."""

    def __init__(self):
        self.rows = [[]]
        self.colour = None

    def write(self, chunk: bytes):
        i = 0
        while i < len(chunk):
            m = SGR.match(chunk, i)
            if m:
                code = m.group(1).decode() or "0"
                last = code.split(";")[-1]
                self.colour = None if last in ("0", "") else COLOURS.get(int(last))
                i = m.end()
                continue
            byte = chunk[i:i + 1]
            if byte == b"\x1b":                      # any other escape: skip it
                j = i + 1
                while j < len(chunk) and not (0x40 <= chunk[j] <= 0x7E):
                    j += 1
                i = j + 1
                continue
            if byte == b"\r":
                i += 1
                continue
            if byte == b"\n":
                self.rows.append([])
                i += 1
                continue
            # decode one UTF-8 character
            length = 1
            if chunk[i] >= 0xF0: length = 4
            elif chunk[i] >= 0xE0: length = 3
            elif chunk[i] >= 0xC0: length = 2
            char = chunk[i:i + length].decode("utf-8", "replace")
            i += length
            if char == "\t":
                char = " "
            if len(self.rows[-1]) >= COLS:
                self.rows.append([])
            self.rows[-1].append((char, self.colour))
        self.rows = self.rows[-400:]

    def visible(self):
        return self.rows[-ROWS:] if len(self.rows) >= ROWS else \
            self.rows + [[]] * (ROWS - len(self.rows))


def svg(rows, seconds):
    w = int(COLS * CELL_W + PAD * 2)
    h = int(ROWS * CELL_H + PAD * 2 + 26)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
           f'<rect width="{w}" height="{h}" fill="{BG}"/>',
           f'<rect width="{w}" height="26" fill="#1c1f28"/>']
    for i, c in enumerate(("#f47067", "#e3b341", "#6bc46d")):
        out.append(f'<circle cx="{16 + i * 16}" cy="13" r="5" fill="{c}"/>')
    out.append(f'<text x="{w/2}" y="18" fill="#7d8590" font-size="11" '
               f'font-family="DejaVu Sans Mono, monospace" text-anchor="middle">'
               f'bobbin — cross-file rename — {seconds:.0f}s elapsed</text>')
    for r, row in enumerate(rows):
        y = 26 + PAD + (r + 1) * CELL_H - 5
        # one <text> per colour run, so the monospace grid stays exact
        run, colour, start = [], None, 0
        def flush(end):
            if not run:
                return
            x = PAD + start * CELL_W
            fill = colour or FG
            out.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" '
                       f'font-size="{FONT_SIZE}" font-family="DejaVu Sans Mono, monospace" '
                       f'xml:space="preserve">{html.escape("".join(run))}</text>')
        for c, (char, col) in enumerate(row):
            if col != colour:
                flush(c); run, colour, start = [], col, c
            run.append(char)
        flush(len(row))
    out.append("</svg>")
    return "\n".join(out)


def main():
    events = read_capture()
    frames_dir = DEMO / "frames"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("*"):
        old.unlink()

    screen = Screen()
    clock = 0.0          # replay clock, after speedup
    real = 0.0           # true elapsed, for the caption
    next_frame = 0.0
    n = 0
    for delay, data in events:
        real += delay
        clock += min(delay / SPEED, IDLE_CAP)
        screen.write(data)
        while next_frame <= clock:
            (frames_dir / f"f{n:05d}.svg").write_text(svg(screen.visible(), real))
            n += 1
            next_frame += 1.0 / FPS
    for _ in range(FPS * 2):        # hold the final screen
        (frames_dir / f"f{n:05d}.svg").write_text(svg(screen.visible(), real))
        n += 1
    print(f"{n} frames, {clock:.1f}s of animation from {real:.0f}s of real time")
    return n


if __name__ == "__main__":
    main()
