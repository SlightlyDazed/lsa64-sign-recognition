"""
make_readme_figures.py — Genera las figuras del README en docs/images/.

  banner.png              título con el esqueleto de MediaPipe de ambas manos
  architecture.png        de la cámara a la voz: landmarks → BiLSTM → atención → 64 señas
  what_changed.png        51.2 % → 81.1 %: el bug de normalización y la posición de las muñecas
  per_class_accuracy.png  precisión top-1 de las 64 señas en los signantes de test

Lee results/lsa64_evaluation_summary.json. Uso: python docs/make_readme_figures.py
"""
import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "images")
os.makedirs(OUT, exist_ok=True)

# Paleta de la presentación (docs/LSA64_presentacion.pdf)
BG = "#0f172a"
PANEL = "#1e293b"
EDGE = "#334155"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
ORANGE = "#f59e0b"
GREEN = "#22c55e"
RED = "#ef4444"
PURPLE = "#a855f7"
TEAL = "#14b8a6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})

# 21 landmarks de MediaPipe Hands (mano derecha, palma hacia la cámara), coordenadas aproximadas
HAND = [
    (0.0, 0.0),
    (-0.55, 0.25), (-0.85, 0.6), (-1.05, 0.95), (-1.2, 1.25),
    (-0.35, 1.2), (-0.42, 1.75), (-0.47, 2.1), (-0.52, 2.45),
    (0.0, 1.3), (0.0, 1.9), (0.0, 2.3), (0.0, 2.65),
    (0.3, 1.2), (0.38, 1.75), (0.43, 2.1), (0.47, 2.42),
    (0.6, 1.05), (0.75, 1.45), (0.85, 1.75), (0.95, 2.0),
]
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def draw_hand(ax, cx, cy, scale, color, mirror=False, labels=False):
    pts = [((-x if mirror else x) * scale + cx, y * scale + cy) for x, y in HAND]
    for i, j in CONNECTIONS:
        (x1, y1), (x2, y2) = pts[i], pts[j]
        ax.plot([x1, x2], [y1, y2], color=color, lw=6, alpha=0.15, solid_capstyle="round")
        ax.plot([x1, x2], [y1, y2], color=color, lw=2.2, solid_capstyle="round")
    for k, (x, y) in enumerate(pts):
        ax.scatter([x], [y], s=70, color=BG, edgecolors=color, linewidths=2, zorder=3)
        if labels:
            ax.text(x + 0.09 * scale, y, str(k), fontsize=8, color=MUTED, va="center")
    return pts


def banner():
    fig = plt.figure(figsize=(16, 4.6), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    for x in range(17):
        ax.axvline(x, color="#141d33", lw=1)
    for y in range(5):
        ax.axhline(y, color="#141d33", lw=1)

    ax.text(0.9, 2.95, "LSA64", fontsize=78, weight="bold", color=TEXT)
    ax.text(5.15, 2.95, "Sign Recognition", fontsize=38, weight="bold", color=ORANGE)
    ax.text(0.95, 2.2, "Real-time Argentinian Sign Language recognition from webcam video",
            fontsize=17, color="#cbd5e1")
    ax.text(0.95, 1.75, "BiLSTM + temporal attention on MediaPipe hand landmarks",
            fontsize=17, color="#cbd5e1")
    ax.add_patch(plt.Rectangle((0.95, 1.25), 0.8, 0.05, color=ORANGE, zorder=3))
    ax.text(0.95, 0.72, "64 SIGNS  ·  81.1 % TOP-1 ON UNSEEN SIGNERS  ·  SPOKEN IN SPANISH",
            fontsize=13, color=ORANGE, family="DejaVu Sans Mono")

    draw_hand(ax, 12.0, 0.8, 1.05, RED, mirror=True)
    draw_hand(ax, 14.85, 0.8, 1.05, TEAL)
    fig.savefig(os.path.join(OUT, "banner.png"), facecolor=BG)
    plt.close(fig)


def box(ax, x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=PANEL, ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=12.5, weight="bold",
            color=color)
    ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top", fontsize=10, color=TEXT,
            linespacing=1.5)


def arrow(ax, x1, x2, y):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6, mutation_scale=16))


def architecture():
    fig = plt.figure(figsize=(16, 3.6), dpi=110, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    w, h, y = 2.35, 2.1, 0.45
    xs = [0.3, 2.95, 5.6, 8.25, 10.9, 13.55]
    stages = [
        ("Webcam", "video frames\nup to 64 per sign", MUTED),
        ("MediaPipe Hands", "2 × 21 landmarks\n× (x, y, depth) = 126\n+ 2 wrist positions = 4", GREEN),
        ("Projection", "Linear 130 → 512\nLayerNorm · ReLU\nDropout", TEAL),
        ("BiLSTM", "3 layers\nhidden 512\nbidirectional", PURPLE),
        ("Attention", "pooling over time\nweights the frames\nthat matter", ORANGE),
        ("64 signs", "Spanish label\n+ gTTS speech\nwhen conf. > 50 %", RED),
    ]
    for x, (t, b, c) in zip(xs, stages):
        box(ax, x, y, w, h, t, b, c)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, a + w + 0.04, b - 0.04, y + h / 2)

    ax.text(0.3, 3.05, "130 features per frame  →  17.4 M parameters  →  one sign per sequence",
            fontsize=13, color=MUTED, family="DejaVu Sans Mono")
    fig.savefig(os.path.join(OUT, "architecture.png"), facecolor=BG)
    plt.close(fig)


def what_changed():
    fig = plt.figure(figsize=(16, 5.2), dpi=110, facecolor=BG)

    # Izquierda: la mano, qué se perdía y qué se agregó
    ax = fig.add_axes([0.0, 0.0, 0.42, 1.0])
    ax.set_xlim(-2.6, 3.4)
    ax.set_ylim(-0.9, 3.4)
    ax.axis("off")
    pts = draw_hand(ax, 0.0, 0.0, 1.0, TEAL, labels=True)
    wx, wy = pts[0]
    ax.scatter([wx], [wy], s=420, facecolors="none", edgecolors=ORANGE, linewidths=2.5, zorder=4)
    ax.text(wx + 0.35, wy - 0.55, "wrist x, y in the frame\n(+4 features: where the sign is made)",
            fontsize=10.5, color=ORANGE)
    ax.text(-2.4, 3.05, "21 landmarks per hand", fontsize=14, weight="bold", color=TEXT)
    ax.text(-2.4, 2.7, "x, y in pixels + depth", fontsize=11, color=MUTED)

    # Derecha: barras antes / después
    ax2 = fig.add_axes([0.5, 0.25, 0.46, 0.6], facecolor=BG)
    labels = ["First version", "Final model"]
    vals = [51.2, 81.1]
    bars = ax2.barh(labels, vals, color=[EDGE, GREEN], height=0.55)
    for bar, v in zip(bars, vals):
        ax2.text(v + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f} %", va="center",
                 fontsize=18, weight="bold", color=TEXT)
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", labelsize=13, colors=TEXT, length=0)
    ax2.tick_params(axis="x", labelsize=10)
    ax2.set_xlabel("Top-1 accuracy on test signers 9–10 (%)", fontsize=11)
    for s in ["top", "right", "left"]:
        ax2.spines[s].set_visible(False)
    ax2.spines["bottom"].set_color(EDGE)
    ax2.grid(axis="x", color="#1e293b", lw=1)
    ax2.set_axisbelow(True)

    fig.text(0.5, 0.9, "Two fixes, +29.9 points", fontsize=17, weight="bold", color=TEXT)
    fig.text(0.5, 0.025,
             "① depth was read as a confidence score and erased 20 of 21 points per hand\n"
             "② the two wrist positions restore where the sign is made (forehead vs. chest)",
             fontsize=10.5, color=MUTED, linespacing=1.6)
    fig.savefig(os.path.join(OUT, "what_changed.png"), facecolor=BG)
    plt.close(fig)


def per_class_accuracy():
    with open(os.path.join(ROOT, "results", "lsa64_evaluation_summary.json")) as f:
        data = json.load(f)
    pc = sorted(data["per_class_accuracy"], key=lambda p: (-p["accuracy"], p["class_name"]))
    names = [p["class_name"] for p in pc]
    acc = [p["accuracy"] * 100 for p in pc]
    colors = [GREEN if a == 100 else TEAL if a >= 80 else ORANGE if a >= 50 else RED for a in acc]

    fig, ax = plt.subplots(figsize=(16, 5.4), dpi=110, facecolor=BG)
    ax.set_facecolor(BG)
    ax.bar(range(len(acc)), acc, color=colors, width=0.78)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8.5)
    ax.set_xlim(-0.7, len(acc) - 0.3)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=11)
    top1 = data["test_accuracy_top1"] * 100
    ax.axhline(top1, color=TEXT, lw=1, ls="--", alpha=0.6)
    ax.text(len(acc) - 0.5, top1 + 2, f"mean {top1:.1f} %", ha="right", fontsize=10, color=TEXT)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(EDGE)
    ax.grid(axis="y", color="#1e293b", lw=1)
    ax.set_axisbelow(True)
    n_perfect = sum(a == 100 for a in acc)
    ax.set_title(f"Per-sign accuracy on unseen signers — {n_perfect} of 64 signs recognised perfectly",
                 fontsize=14, weight="bold", color=TEXT, loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "per_class_accuracy.png"), facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    banner()
    architecture()
    what_changed()
    per_class_accuracy()
    print(f"Figures written to {OUT}")
