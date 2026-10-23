"""
Graphe : meilleure précision de sélection en fonction du taux d'apprentissage,
avec et sans masque tiré à chaque pas, un panneau par modèle ; en bas, le
gain du masque en fonction du taux rapporté au meilleur taux sans masque.

Usage : python lambada/plot_mask_lr.py
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent / "outputs" / "openai" / "convergence"
MODELS = [("28M.json", "TinyStories-28M"), ("pythia70m.json", "Pythia-70M"), ("pythia160m.json", "Pythia-160M")]
COLORS = {0.0: "#2a78d6", 0.125: "#eb6834"}  # palette de référence, emplacements 1 et 2
LABELS = {0.0: "Sans masque", 0.125: "Masque 12,5 % tiré à chaque pas"}
MODEL_COLORS = ["#2a78d6", "#1baf7a", "#6250d6"]  # emplacements 1, 3, 7 : une couleur par modèle en bas
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax, ticks=None, labels=None):
    ax.set_xscale("log")
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    fig = plt.figure(figsize=(12, 8.2), facecolor="#fcfcfb")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.45, wspace=0.25)
    bottom = fig.add_subplot(grid[1, :])
    summary = {}

    for col, (file, title) in enumerate(MODELS):
        runs = json.loads((ROOT / file).read_text(encoding="utf-8"))["runs"]
        lrs = sorted({r["lr"] for r in runs})
        ax = fig.add_subplot(grid[0, col])
        means = {}
        for mask in (0.0, 0.125):
            vals = [[r["best_acc"] * 100 for r in runs if r["lr"] == lr and r["mask"] == mask] for lr in lrs]
            m = [st.mean(v) for v in vals]
            means[mask] = m
            low = [mi - min(v) for mi, v in zip(m, vals)]
            high = [max(v) - mi for mi, v in zip(m, vals)]
            ax.errorbar(lrs, m, yerr=[low, high], color=COLORS[mask], linewidth=2, marker="o", markersize=7,
                        capsize=3, elinewidth=1.2, label=LABELS[mask],
                        markeredgecolor="#fcfcfb", markeredgewidth=1.5)
        best = max(range(len(lrs)), key=lambda i: means[0.0][i])
        ax.axvline(lrs[best], color=MUTED, linewidth=0.8, linestyle=":")
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=8)
        ax.set_xlabel("Taux d'apprentissage", color=MUTED, fontsize=9)
        if col == 0:
            ax.set_ylabel("Meilleure précision, sélection (%)", color=MUTED, fontsize=9)
        style(ax, lrs, [f"{lr:.0e}".replace("e-0", "e-") for lr in lrs])
        gains = [b - a for a, b in zip(means[0.0], means[0.125])]
        summary[title] = (lrs, gains, lrs[best])

    handles, labels = fig.axes[1].get_legend_handles_labels()  # axes[0] = panneau du bas
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.93))

    for color, (title, (lrs, gains, best_lr)) in zip(MODEL_COLORS, summary.items()):
        ratio = [lr / best_lr for lr in lrs]
        bottom.plot(ratio, gains, color=color, linewidth=2, marker="o", markersize=7,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=title)
    bottom.axhline(0, color=MUTED, linewidth=1)
    bottom.axvline(1, color=MUTED, linewidth=0.8, linestyle=":")
    bottom.set_title("Gain du masque selon l'écart au meilleur taux sans masque", color=INK, fontsize=12,
                     loc="left", pad=8)
    bottom.set_xlabel("Taux d'apprentissage ÷ meilleur taux sans masque (pointillés = meilleur taux)",
                      color=MUTED, fontsize=9)
    bottom.set_ylabel("Masque − sans masque (points)", color=MUTED, fontsize=9)
    bottom.legend(frameon=False, fontsize=9, loc="upper left")
    ticks = [0.5, 1, 2, 4, 5, 10]
    style(bottom, ticks, ["÷2", "meilleur", "×2", "×4", "×5", "×10"])

    fig.text(0.01, 0.005, "Modèle seul fine-tuné sur toutes les données ; moyenne de 3 graines, barres = min et max ; "
             "lambada_openai, 500 passages de sélection.", fontsize=8, color=MUTED)
    out = ROOT / "masque_taux_apprentissage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(out)


if __name__ == "__main__":
    main()
