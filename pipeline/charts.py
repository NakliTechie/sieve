#!/usr/bin/env python3
"""Charts for the README and the thread, drawn from the recorded results (data/bench/*/curve.jsonl, distill.jsonl).

  .venv/bin/python pipeline/charts.py        # -> marketing/{curves,review,order,social}.png

Palette: the dataviz reference categorical slots 1-3 (blue, orange, aqua; validated all-pairs, light surface) for
trained models; neutral gray for zero-shot reference lines. Every line is direct-labelled (aqua is under 3:1 contrast).
The option-order figures are copied from results/order-averaging-2026-09-26.md (djev on fast-decisions, 2,600
questions), which reads them from data/ab/perm.
"""
import json
import os
import statistics as S
import sys
from pathlib import Path

import matplotlib
import matplotlib.lines

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import djcore as P  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "marketing"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STAGES = ["5", "10", "20", "50", "all"]
ARMS = [("tfidf", "TF-IDF (keywords)", BLUE), ("mmbert", "mmBERT-small", ORANGE), ("tiny", "Ettin-17M", AQUA)]

_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Inter", "Helvetica Neue", "Arial") if f in _fonts), "DejaVu Sans")
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11})


def curve(name):
    cells = {}
    for line in (P.DATA / "bench" / name / "curve.jsonl").read_text().splitlines():
        r = json.loads(line)
        cells.setdefault((r["arm"], str(r["stage"])), {})[r.get("seed", 0)] = 100 * r["acc"]
    return cells


def best_zero_shot(cells):
    return max((S.mean(v.values()), a) for (a, s), v in cells.items() if s == "0")


def draw_curves(ax, name, title, label_lines=True):
    cells = curve(name)
    zs, zs_arm = best_zero_shot(cells)
    x = range(len(STAGES))
    ax.axhline(zs, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.text(len(STAGES) - 0.85 if label_lines else len(STAGES) - 1.05, zs - 1.5, f"best zero-shot {zs:.0f} %",
            color=INK2, fontsize=9, ha="left" if label_lines else "right", va="top")
    ends = []
    for arm, label, color in ARMS:
        pts = [(i, cells[(arm, s)]) for i, s in enumerate(STAGES) if (arm, s) in cells]
        if not pts:
            continue
        xs = [i for i, _ in pts]
        ys = [S.mean(v.values()) for _, v in pts]
        err = [S.stdev(v.values()) if len(v) > 1 else 0 for _, v in pts]
        ax.errorbar(xs, ys, yerr=err, color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2, capsize=0,
                    elinewidth=1)
        ends.append([ys[-1], xs[-1], label])
    if label_lines:  # end labels, pushed apart so none touch (min 5 pts)
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 5)
        for y, _, label in ends:
            ax.text(len(STAGES) - 0.85, y, label, va="center", fontsize=9, color=INK2)
    ax.set_xticks(list(x), [f"{s}" for s in STAGES])
    ax.set_xlim(-0.3, len(STAGES) - 0.4 + (1.6 if label_lines else 0))
    ax.set_ylim(20, 100)
    ax.set_title(title, loc="left", fontsize=11.5, color=INK, fontweight="semibold")


def legend(fig, items, y):
    """One legend row under the subtitle: (label, color, style) with style '-o', 'D', '--' or ':'."""
    import matplotlib.lines as L
    h = []
    for label, color, style in items:
        if style == "-o":
            h.append(L.Line2D([], [], color=color, lw=2, marker="o", ms=5, label=label))
        elif style == "D":
            h.append(L.Line2D([], [], color=color, lw=0, marker="D", ms=6, label=label))
        else:
            h.append(L.Line2D([], [], color=color, lw=1.5, ls=(0, (4, 3)) if style == "--" else ":", label=label))
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.055, y), ncol=len(h), frameon=False, fontsize=10,
               handlelength=2.2, columnspacing=1.8)


def fig_curves():
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharey=True)
    for ax, (name, title) in zip(axes.flat, [("banking77-curve", "banking77 · 77 intents"),
                                             ("clinc", "CLINC150 · 151 intents"),
                                             ("massive-en", "MASSIVE English · 60 intents"),
                                             ("hinglish-top", "Hinglish-TOP · 57 intents, code-mixed")]):
        draw_curves(ax, name, title)
    for ax in axes[1]:
        ax.set_xlabel("labelled examples per intent")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy on 1,000 held-out rows (%)")
    fig.suptitle("Train on your own labels: a keyword model passes zero-shot first", x=0.06, ha="left", y=0.99,
                 fontsize=15, fontweight="semibold")
    fig.text(0.06, 0.935, "Mean of 3 seeds (bars: ± 1 SD). Dashed: best zero-shot model, no labels "
             "(DiffusionGemma-Jev with 3 shuffled reads, or GLiNER2.5-Decide).", color=INK2, fontsize=10)
    legend(fig, [(label, color, "-o") for _, label, color in ARMS] + [("best zero-shot", MUTED, "--")], 0.915)
    fig.tight_layout(rect=(0.03, 0, 1, 0.885))
    fig.savefig(OUT / "curves.png", dpi=160)


def fig_review():
    sets = [("massive-en", "MASSIVE English"), ("massive-hi", "MASSIVE Hindi"), ("clinc", "CLINC150")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, (name, title) in zip(axes, sets):
        rec = {}
        for line in (P.DATA / "bench" / name / "distill.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["model"] == "mmbert":
                rec[r["set"]] = 100 * r["acc"]
        teacher = best_zero_shot(curve(name))[0]
        gold = rec["gold"]
        ks = [0, 10, 25, 50]
        ys = [rec[f"review{k}-least-djev@3"] for k in ks]
        ax.axhline(teacher, color=MUTED, lw=1.5, ls=(0, (4, 3)))
        ax.text(29, teacher - 0.8, f"zero-shot teacher {teacher:.0f} %", color=INK2, fontsize=9, ha="left", va="top")
        ax.axhline(gold, color=MUTED, lw=1, ls=":")
        ax.text(37.5, gold + 0.7, f"all human labels {gold:.0f} %", color=INK2, fontsize=9, ha="center")
        ax.plot(ks, ys, color=BLUE, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2)
        ax.text(48.5, ys[-1] - 0.4, f"{ys[-1]:.0f} %", ha="right", va="top", fontsize=10, color=INK)
        rnd = rec["review25-random-djev@3"]
        ax.plot([25], [rnd], color=ORANGE, marker="D", ms=6, lw=0, mec=SURFACE, mew=1.2)
        ax.set_xticks(ks, [f"{k} %" for k in ks])
        ax.set_ylim(55, 90)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="semibold")
        ax.set_xlabel("share of zero-shot labels a person checks")
    axes[0].set_ylabel("mmBERT accuracy (%)")
    fig.suptitle("No labels? Check the half the zero-shot model is least sure of", x=0.06, ha="left", y=0.99,
                 fontsize=15, fontweight="semibold")
    fig.text(0.06, 0.91, "Teacher: DiffusionGemma-Jev (3 shuffled reads) labels the training pool; mmBERT-small "
             "trains on the result. Checks simulated with gold labels.", color=INK2, fontsize=10)
    legend(fig, [("least-confident rows checked first", BLUE, "-o"), ("random 25 % checked", ORANGE, "D"),
                 ("zero-shot teacher", MUTED, "--"), ("all human labels", MUTED, ":")], 0.855)
    fig.tight_layout(rect=(0.03, 0, 1, 0.83))
    fig.savefig(OUT / "review.png", dpi=160)


def fig_order():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    # results/order-averaging-2026-09-26.md: flips when options are reversed, fast-decisions, 2,600 questions
    labels = ["djev\n1 read", "djev\n2 reads", "djev\n3 reads", "GLiNER\n1 read"]
    vals = [23.6, 14.3, 11.8, 3.0]
    bars = a1.bar(range(4), vals, width=0.6, color=[BLUE, BLUE, BLUE, MUTED], edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        a1.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f} %", ha="center", fontsize=10, color=INK)
    a1.set_xticks(range(4), labels)
    a1.set_ylim(0, 28)
    a1.set_ylabel("answers that change (%)")
    a1.set_title("Reverse the options: how many answers flip", loc="left", fontsize=11.5, fontweight="semibold")
    a1.grid(axis="x", visible=False)
    sets = [("banking77-curve", "banking77 (77)"), ("massive-en", "MASSIVE English (60)"),
            ("massive-hi", "MASSIVE Hindi (60)"), ("clinc", "CLINC150 (151)")]
    for i, (name, lab) in enumerate(sets):
        c = curve(name)
        one, three = S.mean(c[("djev", "0")].values()), S.mean(c[("djev@3", "0")].values())
        a2.plot([one, three], [i, i], color=GRID, lw=3, zorder=1)
        a2.plot([one], [i], "o", color=MUTED, ms=8, mec=SURFACE, mew=1.5, zorder=2)
        a2.plot([three], [i], "o", color=BLUE, ms=8, mec=SURFACE, mew=1.5, zorder=2)
        a2.text(three + 1.2, i, f"+{three - one:.0f} pts", va="center", fontsize=10, color=INK)
    a2.set_yticks(range(len(sets)), [s[1] for s in sets])
    a2.set_xlim(30, 80)
    a2.set_xlabel("zero-shot accuracy (%)  ·  gray: 1 read  ·  blue: 3 shuffled reads")
    a2.set_title("Many options: averaging 3 shuffled reads", loc="left", fontsize=11.5, fontweight="semibold")
    a2.grid(axis="y", visible=False)
    fig.suptitle("DiffusionGemma-Jev is sensitive to option order", x=0.06, ha="left", y=0.99, fontsize=15,
                 fontweight="semibold")
    fig.tight_layout(rect=(0.03, 0, 1, 0.95))
    fig.savefig(OUT / "order.png", dpi=160)


def fig_social():
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.text(0.06, 0.78, "sieve", fontsize=64, fontweight="bold", color=INK)
    fig.text(0.06, 0.60, "Start zero-shot. Train on your own labels.\nSwitch only when it wins.", fontsize=26,
             color=INK, va="top", linespacing=1.35)
    fig.text(0.06, 0.16, "Per-customer text classifiers · English, Hindi, Hinglish · MIT", fontsize=15, color=INK2)
    ax = fig.add_axes((0.62, 0.2, 0.34, 0.58))
    draw_curves(ax, "banking77-curve", "", label_lines=False)
    ax.set_ylim(20, 100)
    ax.set_xlabel("labelled examples per intent (banking77)", fontsize=11)
    ax.legend(handles=[matplotlib.lines.Line2D([], [], color=c, lw=2, marker="o", ms=4, label=lab)
                       for _, lab, c in ARMS], loc="lower right", frameon=False, fontsize=10)
    ax.tick_params(labelsize=10)
    fig.savefig(OUT / "social.png", dpi=100, facecolor=SURFACE)


def main():
    OUT.mkdir(exist_ok=True)
    fig_curves()
    fig_review()
    fig_order()
    fig_social()
    print(f"verdict=DRAWN dir={OUT} files=curves.png,review.png,order.png,social.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
