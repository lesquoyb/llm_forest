"""
Test : neutraliser des dimensions de l'embedding raccourcit-il le fine-tuning
sur LAMBADA ?

Pour chaque proportion de dimensions neutralisées et chaque graine, on
fine-tune le modèle de base avec le même budget d'epochs en évaluant la loss
de dev plusieurs fois par epoch. On mesure :

- le coût d'un pas d'entraînement (ms/pas). Neutraliser = multiplier par 0,
  donc le calcul est le même : on s'attend à un coût identique ;
- le nombre d'epochs pour atteindre 90 % de sa propre amélioration de loss
  (vitesse de convergence vers son propre plateau) ;
- le nombre d'epochs pour atteindre une cible commune : la loss que le modèle
  complet (0 % neutralisé) atteint à 90 % de son amélioration ;
- l'epoch du meilleur checkpoint (là où s'arrêterait un early stopping) ;
- la meilleure loss de dev et l'accuracy au meilleur checkpoint.

Plan apparié : pour une même graine, l'ordre des données est le même et les
masques sont emboîtés (les dimensions neutralisées à 25 % font partie de
celles neutralisées à 50 %, etc.). Seule la proportion change.

Exemples :
    python lambada/convergence.py
    python lambada/convergence.py --quick
    python lambada/convergence.py --fractions 0 0.25 0.5 --seeds 5 --epochs 20
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_lambada, prepare_corpus
from finetune import MaskedLM, evaluate, finetune, random_masked_dims


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roneneldan/TinyStories-1M")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--evals-per-epoch", type=int, default=4)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent / "outputs" / "convergence")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="petit test de bout en bout (peu d'exemples, 2 epochs, 1 graine)",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 2
        args.seeds = 1
        args.fractions = [0.0, 0.5]
        args.max_train_samples = args.max_train_samples or 512
        args.max_eval_samples = args.max_eval_samples or 128

    return args


# ============================================================
# Mesures sur une courbe de loss
# ============================================================

def first_crossing(steps, losses, target):
    """Premier pas où la loss passe sous `target` (interpolation linéaire), sinon None."""

    for i, loss in enumerate(losses):
        if loss <= target:
            if i == 0:
                return steps[0]
            prev = losses[i - 1]
            return steps[i - 1] + (steps[i] - steps[i - 1]) * (prev - target) / (prev - loss)

    return None


def own_target(run, q):
    """Loss à q (ex. 90 %) du chemin entre la loss initiale et la meilleure loss."""

    return run["loss"][0] - q * (run["loss"][0] - min(run["loss"]))


def mean_std(values):
    values = [v for v in values if v is not None]

    if not values:
        return None, None

    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.eval()

    hidden_size = base_model.config.hidden_size

    train_raw, dev_raw, _ = load_lambada(
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
    )
    train = prepare_corpus(train_raw, tokenizer, args.max_length)
    dev = prepare_corpus(dev_raw, tokenizer, args.max_length)

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    print(
        f"Train : {len(train)} | Dev : {len(dev)} | {steps_per_epoch} pas/epoch | "
        f"évaluation tous les {eval_every} pas"
    )

    # --------------------------------------------------------
    # Entraînements
    # --------------------------------------------------------

    # Chauffe GPU (noyaux CUDA, allocateur) : sans elle, le premier
    # entraînement, toujours à 0 %, paraîtrait plus lent par pas.
    finetune(
        base_model, tokenizer, train[:args.batch_size * 20],
        epochs=1, batch_size=args.batch_size, max_length=args.max_length, verbose=False,
    )

    runs = []
    n_runs = args.seeds * len(args.fractions)

    for seed in range(args.seeds):
        for fraction in args.fractions:
            # Même générateur pour toutes les proportions d'une graine :
            # masques emboîtés.
            masked_dims = random_masked_dims(
                hidden_size, fraction, torch.Generator().manual_seed(seed)
            )

            initial = evaluate(
                [MaskedLM(base_model, tokenizer, masked_dims)], dev, max_length=args.max_length
            )["soft"]

            tuned = finetune(
                base_model,
                tokenizer,
                train,
                dev,
                masked_dims=masked_dims,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                max_length=args.max_length,
                seed=seed,
                eval_every=eval_every,
                verbose=False,
            )
            h = tuned.history

            run = {
                "fraction": fraction,
                "seed": seed,
                "masked_dims": list(masked_dims),
                "steps": [0] + h["val_step"],
                "loss": [initial["loss"]] + h["val_loss"],
                "acc": [initial["accuracy"]] + h["val_acc"],
                "compute_s": [0.0] + h["val_compute_s"],
                "ms_per_step": h["val_compute_s"][-1] / h["val_step"][-1] * 1000,
            }
            runs.append(run)

            best = min(range(len(run["loss"])), key=run["loss"].__getitem__)
            print(
                f"[{len(runs)}/{n_runs}] graine {seed} | {fraction:>6.1%} neutralisé | "
                f"loss {run['loss'][0]:.3f} -> {run['loss'][best]:.3f} "
                f"(epoch {run['steps'][best] / steps_per_epoch:.2f}) | "
                f"{run['ms_per_step']:.2f} ms/pas",
                flush=True,
            )

    # --------------------------------------------------------
    # Mesures
    # --------------------------------------------------------

    baseline_runs = [r for r in runs if r["fraction"] == 0.0]
    common_target = (
        statistics.mean(own_target(r, args.target) for r in baseline_runs)
        if baseline_runs
        else None
    )

    def to_epochs(step):
        return None if step is None else step / steps_per_epoch

    for run in runs:
        best = min(range(len(run["loss"])), key=run["loss"].__getitem__)
        run["best_loss"] = run["loss"][best]
        run["best_acc"] = run["acc"][best]
        run["best_epoch"] = to_epochs(run["steps"][best])
        run["epochs_to_own_target"] = to_epochs(
            first_crossing(run["steps"], run["loss"], own_target(run, args.target))
        )
        run["epochs_to_common_target"] = (
            to_epochs(first_crossing(run["steps"], run["loss"], common_target))
            if common_target is not None
            else None
        )

    metrics = [
        "ms_per_step",
        "epochs_to_own_target",
        "epochs_to_common_target",
        "best_epoch",
        "best_loss",
        "best_acc",
    ]

    summary = []
    for fraction in args.fractions:
        group = [r for r in runs if r["fraction"] == fraction]
        row = {
            "fraction": fraction,
            "kept_dims": hidden_size - round(fraction * hidden_size),
            "initial_loss": statistics.mean(r["loss"][0] for r in group),
            "reached_common_target": sum(r["epochs_to_common_target"] is not None for r in group),
            "n_seeds": len(group),
        }
        for metric in metrics:
            row[metric], row[f"{metric}_std"] = mean_std([r[metric] for r in group])
        summary.append(row)

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    def fmt(value, std, spec):
        if value is None:
            return "jamais"
        return f"{value:{spec}}±{std:{spec}}"

    print("\n" + "=" * 118)
    print(
        f"{'Neutralisé':>10}{'Dims':>6}{'ms/pas':>13}{'Loss init':>11}"
        f"{'Meilleure loss':>17}{'Acc dev':>15}{'Epoch meilleur':>17}"
        f"{f'Ep. {args.target:.0%} propre':>15}{'Ep. cible commune':>22}"
    )
    print("-" * 118)
    for row in summary:
        acc = (
            f"{row['best_acc'] * 100:.1f}±{row['best_acc_std'] * 100:.1f}%"
            if row["best_acc"] is not None
            else "-"
        )
        common = fmt(row["epochs_to_common_target"], row["epochs_to_common_target_std"], ".2f")
        common += f" ({row['reached_common_target']}/{row['n_seeds']})"
        print(
            f"{row['fraction']:>10.1%}{row['kept_dims']:>6}"
            f"{fmt(row['ms_per_step'], row['ms_per_step_std'], '.2f'):>13}"
            f"{row['initial_loss']:>11.3f}"
            f"{fmt(row['best_loss'], row['best_loss_std'], '.3f'):>17}"
            f"{acc:>15}"
            f"{fmt(row['best_epoch'], row['best_epoch_std'], '.2f'):>17}"
            f"{fmt(row['epochs_to_own_target'], row['epochs_to_own_target_std'], '.2f'):>15}"
            f"{common:>22}"
        )
    print("-" * 118)
    if common_target is not None:
        print(
            f"Cible commune : loss de dev {common_target:.3f}, atteinte par le modèle "
            f"complet à {args.target:.0%} de son amélioration."
        )
    print(f"Budget : {args.epochs} epochs, {args.seeds} graine(s) par proportion.")

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "steps_per_epoch": steps_per_epoch,
                "common_target_loss": common_target,
                "summary": summary,
                "runs": runs,
            },
            f,
            indent=2,
        )

    plot_curves(runs, args.fractions, steps_per_epoch, common_target, output_dir / "curves.png")
    plot_summary(summary, args, output_dir / "summary.png")

    print(f"\nRésultats et graphiques enregistrés dans {output_dir}/")


# ============================================================
# Graphiques
# ============================================================

# Rampe bleue ordinale (claire -> foncée), validée : 5 niveaux au maximum.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SERIES = "#2a78d6"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _curve_fractions(fractions):
    """Au plus 5 proportions réparties sur la liste, pour rester lisible."""

    if len(fractions) <= len(RAMP):
        return list(fractions)

    idx = sorted({round(i * (len(fractions) - 1) / (len(RAMP) - 1)) for i in range(len(RAMP))})

    return [fractions[i] for i in idx]


def plot_curves(runs, fractions, steps_per_epoch, common_target, path):
    shown = _curve_fractions(sorted(fractions))
    colors = RAMP[-len(shown):] if len(shown) < len(RAMP) else RAMP

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE)

    for fraction, color in zip(shown, colors):
        group = [r for r in runs if r["fraction"] == fraction]
        epochs = [s / steps_per_epoch for s in group[0]["steps"]]
        loss = [statistics.mean(v) for v in zip(*(r["loss"] for r in group))]
        acc = [statistics.mean(v) * 100 for v in zip(*(r["acc"] for r in group))]
        label = f"{fraction * 100:g} % neutralisé"

        for ax, values in ((ax_loss, loss), (ax_acc, acc)):
            ax.plot(epochs, values, color=color, linewidth=2, label=label)
            ax.annotate(
                f"{fraction * 100:g} %",
                (epochs[-1], values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=TEXT_SECONDARY,
            )

    if common_target is not None:
        ax_loss.axhline(common_target, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
        ax_loss.annotate(
            "cible commune",
            (0, common_target),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=9,
            color=TEXT_SECONDARY,
        )

    _style(ax_loss, "Loss de dev (moyenne sur les graines)", "Epoch", "Loss par token")
    _style(ax_acc, "Accuracy de dev sur le dernier mot", "Epoch", "Accuracy (%)")
    ax_loss.legend(frameon=False, fontsize=9, labelcolor=TEXT)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_summary(summary, args, path):
    x = [row["fraction"] * 100 for row in summary]

    panels = [
        ("ms_per_step", "Coût d'un pas d'entraînement", "ms / pas", 1),
        ("epochs_to_own_target", f"Epochs pour {args.target:.0%} de sa propre amélioration", "Epochs", 1),
        ("epochs_to_common_target", "Epochs pour atteindre la cible commune", "Epochs", 1),
        ("best_epoch", "Epoch du meilleur checkpoint", "Epoch", 1),
        ("best_loss", "Meilleure loss de dev", "Loss par token", 1),
        ("best_acc", "Accuracy de dev au meilleur checkpoint", "Accuracy (%)", 100),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)

    for ax, (metric, title, ylabel, scale) in zip(axes.flat, panels):
        points = [
            (xi, row[metric] * scale, row[f"{metric}_std"] * scale)
            for xi, row in zip(x, summary)
            if row[metric] is not None
        ]

        if points:
            px, py, pe = zip(*points)
            ax.errorbar(
                px, py, yerr=pe,
                color=SERIES, linewidth=2, marker="o", markersize=6,
                capsize=3, elinewidth=1,
            )

        missing = [xi for xi, row in zip(x, summary) if row[metric] is None]
        for xi in missing:
            ax.annotate(
                "jamais\natteinte", (xi, 0.5), xycoords=("data", "axes fraction"),
                ha="center", fontsize=8, color=TEXT_SECONDARY,
            )

        _style(ax, title, "Dimensions neutralisées (%)", ylabel)
        ax.set_xlim(min(x) - 5, max(x) + 5)

    fig.suptitle(
        f"LAMBADA : effet de la proportion de dimensions neutralisées "
        f"(moyenne ± écart-type sur {args.seeds} graine(s), budget {args.epochs} epochs)",
        color=TEXT,
        fontsize=12,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
