"""Preflight the poster build before spending 20 minutes on a print run.

Catches the failures that have actually happened here: a copy function deleted
while refactoring, a takeaway grown long enough that the renderer silently
shrinks the whole block below body size, a headline number that came back NaN
because a sweep did not write the metric, and figures missing from the slots.

    .venv312/bin/python3.12 scripts/check_poster.py

Exits non-zero if anything is wrong, so it can gate a build.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poster_layout as L

REQUIRED_COPY = (
    "background_paragraphs", "approach_bullets", "methods_note",
    "conclusion_takeaways", "conclusion_note", "figure_caption",
)

# Numbers the copy quotes unconditionally; NaN here means a broken sweep.
REQUIRED_FINITE = (
    "model", "explicit", "majority", "delta", "attn_r2", "dosage_r2",
    "plain", "n_sites", "n_individuals", "mask_rate",
)


def main() -> int:
    problems: list[str] = []
    notes: list[str] = []

    missing = [name for name in REQUIRED_COPY if not hasattr(L, name)]
    if missing:
        problems.append(f"poster_layout is missing copy functions: {missing}")
        print("\n".join(f"FAIL  {p}" for p in problems))
        return 1

    try:
        h = L.load_headline()
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        print(f"FAIL  could not load headline numbers from {L.RESULTS}: {exc}")
        return 1

    for field in REQUIRED_FINITE:
        value = getattr(h, field)
        if value != value:
            problems.append(f"headline.{field} is NaN — the sweep did not write it")

    if not h.has_strong:
        notes.append(
            "no strong_baseline.csv — the poster will quote the margin over the sparse "
            "top-8 control, which overstates it. Run scripts/strong_baseline_pass.py."
        )
    if h.crossover_n is None:
        notes.append("no crossover in the scaling curve; panel B is framed as ahead at every size")

    # Every copy function must actually run against these numbers.
    rendered: dict[str, object] = {}
    for name in REQUIRED_COPY:
        fn = getattr(L, name)
        try:
            rendered[name] = fn(h)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}() raised {type(exc).__name__}: {exc}")

    # The conclusion takeaways must each fit one line at body size, or
    # draw_bullets shrinks them below the surrounding copy and the section stops
    # reading as a summary.
    fig = plt.figure(figsize=(L.PAGE_W, L.PAGE_H), dpi=150)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def width_in(text: str, size: float, weight: str = "normal") -> float:
        artist = fig.text(0, 0, text, fontsize=size, fontfamily=L.PDF_FONT_BODY,
                          fontweight=weight)
        w = artist.get_window_extent(renderer=renderer).width / fig.dpi
        artist.remove()
        return w

    usable = L.CO_BODY.w - 0.75
    for label, body, _ in rendered.get("conclusion_takeaways", []):
        total = width_in(f"{label}: ", L.SZ_BODY, "bold") + width_in(body, L.SZ_BODY)
        if total > usable:
            problems.append(
                f"conclusion takeaway wraps at {L.SZ_BODY}pt "
                f"({total:.2f} in > {usable:.2f} in): {label}: {body}"
            )

    def n_lines(text: str, size: float, width: float) -> int:
        words = text.split()
        if not words:
            return 0
        lines = 1
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if width_in(trial, size) <= width:
                current = trial
            else:
                lines += 1
                current = word
        return lines

    caption = rendered.get("figure_caption")
    if isinstance(caption, str):
        lines = n_lines(caption, L.SZ_CAPTION, L.RE_CAPTION.w)
        line_h = L.SZ_CAPTION / 72.0 * 1.22
        if lines * line_h > L.RE_CAPTION.h:
            problems.append(
                f"figure caption does not fit at {L.SZ_CAPTION}pt "
                f"({lines} lines, {lines * line_h:.2f} in > {L.RE_CAPTION.h:.2f} in)"
            )

    note = rendered.get("conclusion_note")
    if isinstance(note, str):
        note_size = 28  # matches render_poster_pdf.py
        lines = n_lines(note, note_size, L.CO_NOTE.w)
        line_h = note_size / 72.0 * 1.22
        if lines * line_h > L.CO_NOTE.h:
            problems.append(
                f"conclusion note does not fit at {note_size}pt "
                f"({lines} lines, {lines * line_h:.2f} in > {L.CO_NOTE.h:.2f} in)"
            )

    methods = rendered.get("methods_note")
    if isinstance(methods, str):
        methods_size = 30  # matches render_poster_pdf.py
        lines = n_lines(methods, methods_size, L.ME_NOTE.w)
        line_h = methods_size / 72.0 * 1.22
        if lines * line_h > L.ME_NOTE.h:
            problems.append(
                f"methods note does not fit at {methods_size}pt "
                f"({lines} lines, {lines * line_h:.2f} in > {L.ME_NOTE.h:.2f} in)"
            )
    plt.close(fig)

    for name, (filename, _) in L.FIGURES.items():
        if not (L.POSTER / filename).exists():
            problems.append(f"figure for slot {name} not built: {filename}")
    if not (L.POSTER / "fig_qr.png").exists():
        notes.append("fig_qr.png not built; the template's QR (wrong project) would be used")

    for note in notes:
        print(f"WARN  {note}")
    for problem in problems:
        print(f"FAIL  {problem}")
    if not problems:
        print(f"OK    {L.RESULTS}: model {h.pct(h.model)}, {100 * h.delta:+.1f} pts vs usual "
              f"explicit-LD"
              + (f", {100 * h.strong_delta:+.1f} vs tuned all-partner" if h.has_strong else "")
              + f", {h.n_seeds} seeds")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
