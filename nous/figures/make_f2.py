"""Regenerate f2_architecture.png (the writeup's architecture diagram).

Usage: python make_f2.py   (writes f2_architecture.png next to itself)

Hardware-neutral by design: the diagram describes the data flow, not any
particular rig. Run with any matplotlib >= 3.5.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

BG = "#fbfbf8"
BOX_FILL = "#eaf0fb"
BOX_EDGE = "#3b78d8"
TEXT = "#26282b"
ARROW = "#6e6e6e"

BOXES = {
    # name: (x, y, w, h, label)
    "agent": (0.02, 0.42, 0.20, 0.26, "Hermes Agent\n(tools, planning,\ntask loop)"),
    "model": (0.31, 0.42, 0.24, 0.26,
              "Local model\n(your weights, your\nhardware: CPU-only\nto multi-GPU)"),
    "output": (0.64, 0.42, 0.30, 0.26,
               "Agent output\n(replies, tool calls,\nfinal answers)"),
    "probes": (0.31, 0.05, 0.24, 0.22, "SAE probes\n(forward hooks on\nchosen layers)"),
    "trace": (0.64, 0.05, 0.30, 0.22,
              "Live feature trace\n(per-token top-k JSONL,\nsame inference pass)"),
}

ARROWS = [
    # (from-box right/bottom, to-box, label, orientation)
    ("agent", "model", "prompt", "h"),
    ("model", "output", "tokens", "h"),
    ("model", "probes", "residual stream", "v"),
    ("probes", "trace", "features", "h"),
]


def center(name):
    x, y, w, h, _ = BOXES[name]
    return x + w / 2, y + h / 2


fig, ax = plt.subplots(figsize=(12.0, 4.3), dpi=120)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

fig.text(0.02, 0.955,
         "One forward pass, two outputs: what the agent said, and what it was thinking",
         fontsize=15, fontweight="bold", color=TEXT, ha="left", va="top",
         family="DejaVu Sans")

for x, y, w, h, label in BOXES.values():
    ax.add_patch(Rectangle((x, y), w, h, facecolor=BOX_FILL, edgecolor=BOX_EDGE,
                           linewidth=1.8))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=11.5, color=TEXT, family="DejaVu Sans")

for src, dst, label, orient in ARROWS:
    sx, sy, sw, sh, _ = BOXES[src]
    dx, dy, dw, dh, _ = BOXES[dst]
    if orient == "h":
        start = (sx + sw, sy + sh / 2)
        end = (dx, dy + dh / 2)
        lx, ly = (start[0] + end[0]) / 2, start[1] + 0.055
    else:
        start = (sx + sw / 2, sy)
        end = (dx + dw / 2, dy + dh)
        lx, ly = start[0] + 0.005, (start[1] + end[1]) / 2
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 linewidth=1.6, color=ARROW, shrinkA=2, shrinkB=2))
    ax.text(lx, ly, label, ha="center" if orient == "h" else "left",
            va="center", fontsize=10.5, color=ARROW, family="DejaVu Sans")

for name in ("f2_architecture.png", "f2_architecture.svg"):
    out = Path(__file__).parent / name
    fig.savefig(out, bbox_inches="tight", facecolor=BG,
                dpi=300 if name.endswith(".png") else "figure")
    print(f"wrote {out}")
