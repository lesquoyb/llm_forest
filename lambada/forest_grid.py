"""
LAMBADA : Random LLM Forest selon la taille de la forêt et la taille du masque.

Pour chaque proportion de dimensions neutralisées, on entraîne une seule fois
une forêt de N membres (bootstrap + masque). Une forêt de k < N membres est
un sous-ensemble de ces N membres : pour chaque k, on moyenne les scores sur
plusieurs sous-ensembles tirés au hasard (les mêmes indices pour toutes les
proportions). Chaque membre ne tourne qu'une fois sur le test.

- 0 % neutralisé = bagging sans masque : c'est la référence pour isoler
  l'effet du masque.
- Référence supplémentaire : un modèle fine-tuné seul sur tout le corpus.
- Temps d'entraînement d'une forêt = somme des temps de ses membres.
- Temps d'inférence d'une forêt de k membres = k x temps d'un membre (les
  membres tournent l'un après l'autre).

Exemples :
    python lambada/forest_grid.py
    python lambada/forest_grid.py --quick
    python lambada/forest_grid.py --max-estimators 15 --fractions 0 0.25 0.5
"""

import argparse
import itertools
import json
import math
import random
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_lambada, prepare_corpus
from finetune import evaluate, finetune
from forest import RandomLLMForest
from main import timed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roneneldan/TinyStories-1M")
    parser.add_argument("--max-estimators", type=int, default=10)
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.0, 0.125, 0.25, 0.5, 0.75]
    )
    parser.add_argument("--subsets-per-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--format", choices=["cimec", "openai"], default="cimec",
        help="openai = texte propre, entraînement sur la moitié a du test (voir data.py)",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="tous les membres font exactement --epochs epochs (meilleur pour la forêt)",
    )
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--offload",
        action="store_true",
        help="garde les membres entraînés en mémoire vive jusqu'à l'évaluation (gros modèles)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reprend un run interrompu : garde la référence et les proportions "
        "déjà présentes dans results.json",
    )
    parser.add_argument(
        "--test-half", choices=["a", "b"], default=None,
        help="évaluer sur une moitié du test (b = moitié jamais utilisée pour choisir)",
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent / "outputs" / "forest_grid")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="petit test de bout en bout (peu d'exemples, 1 epoch, 3 membres)",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.max_estimators = 3
        args.fractions = [0.0, 0.25]
        args.max_train_samples = args.max_train_samples or 512
        args.max_eval_samples = args.max_eval_samples or 256

    return args


def draw_subsets(n_members, per_size, rng):
    """Pour chaque taille k : toutes les combinaisons si elles sont peu nombreuses, sinon un tirage."""

    subsets = {}

    for k in range(1, n_members + 1):
        if math.comb(n_members, k) <= per_size:
            subsets[k] = [list(c) for c in itertools.combinations(range(n_members), k)]
        else:
            drawn = set()
            while len(drawn) < per_size:
                drawn.add(tuple(sorted(rng.sample(range(n_members), k))))
            subsets[k] = [list(c) for c in sorted(drawn)]

    return subsets


def mean_std(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

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

    train_raw, dev_raw, test_raw = load_lambada(
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
        seed=args.seed,
        test_half=args.test_half,
        fmt=args.format,
    )
    train = prepare_corpus(train_raw, tokenizer, args.max_length)
    dev = prepare_corpus(dev_raw, tokenizer, args.max_length)
    test = prepare_corpus(test_raw, tokenizer, args.max_length)

    print(f"Train : {len(train)} | Dev : {len(dev)} | Test : {len(test)}")

    finetune_kwargs = dict(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # --------------------------------------------------------
    # Référence : fine-tuné seul sur tout le corpus
    # --------------------------------------------------------

    previous = None
    if args.resume and (output_dir / "results.json").exists():
        with open(output_dir / "results.json", encoding="utf-8") as f:
            previous = json.load(f)

    if previous is not None:
        reference_row = previous["reference"]
        grid = previous["grid"]
        members_info = {float(k): v for k, v in previous["members"].items()}
        print(
            f"\nReprise : référence et proportions "
            f"{sorted(members_info)} relues dans results.json",
            flush=True,
        )
    else:
        print("\n=== Référence : fine-tuné seul (corpus complet, sans masque) ===", flush=True)

        reference = finetune(
            base_model, tokenizer, train, dev, seed=args.seed, verbose=False, **finetune_kwargs
        )
        evaluate([reference], test[:64])  # préchauffage GPU
        ref_scores, ref_inference = timed(lambda: evaluate([reference], test), device)
        reference_row = {
            "params": sum(p.numel() for p in base_model.parameters()),
            "accuracy": ref_scores["soft"]["accuracy"],
            "perplexity": ref_scores["soft"]["perplexity"],
            "train_time_s": reference.train_time_s,
            "inference_time_s": ref_inference,
        }
        print(
            f"acc {reference_row['accuracy'] * 100:.2f}% | "
            f"perplexité {reference_row['perplexity']:.1f} | "
            f"entraînement {reference.train_time_s:.1f}s",
            flush=True,
        )
        del reference

        grid = []
        members_info = {}

    # --------------------------------------------------------
    # Forêts
    # --------------------------------------------------------

    subsets = draw_subsets(args.max_estimators, args.subsets_per_size, random.Random(args.seed))
    flat_subsets = [idx for k in subsets for idx in subsets[k]]

    for fraction in args.fractions:
        if fraction in members_info:
            print(f"\n{fraction * 100:g} % neutralisé : déjà fait, ignoré", flush=True)
            continue

        print(
            f"\n=== {fraction * 100:g} % neutralisé : {args.max_estimators} membres ===",
            flush=True,
        )

        forest = RandomLLMForest(
            base_model,
            tokenizer,
            n_estimators=args.max_estimators,
            mask_fraction=fraction,
            early_stopping=not args.no_early_stopping,
            offload=args.offload,
            seed=args.seed,
            verbose=True,
            **finetune_kwargs,
        ).fit(train)
        forest.to(device)

        # Coût d'inférence : tous les membres, sans calcul des sous-ensembles.
        _, t_all = timed(
            lambda: evaluate(forest.members, test, batch_size=args.eval_batch_size), device
        )
        inference_per_member = t_all / args.max_estimators

        scores = evaluate(
            forest.members, test, batch_size=args.eval_batch_size, subsets=flat_subsets
        )
        by_subset = iter(scores["subsets"])
        train_times = [m.train_time_s for m in forest.members]

        members_info[fraction] = [
            {"masked_dims": list(m.masked_dims), "train_time_s": m.train_time_s, **s}
            for m, s in zip(forest.members, scores["members"])
        ]

        for k, idx_list in subsets.items():
            results = [next(by_subset) for _ in idx_list]
            acc_soft = mean_std([r["soft"]["accuracy"] for r in results])
            acc_hard = mean_std([r["hard"]["accuracy"] for r in results])
            log_ppl = mean_std([math.log(r["soft"]["perplexity"]) for r in results])
            train_time = mean_std([sum(train_times[i] for i in idx) for idx in idx_list])

            grid.append({
                "fraction": fraction,
                "size": k,
                "n_subsets": len(idx_list),
                "accuracy": acc_soft[0],
                "accuracy_std": acc_soft[1],
                "hard_accuracy": acc_hard[0],
                "hard_accuracy_std": acc_hard[1],
                # Moyenne géométrique : les perplexités varient sur des ordres de grandeur.
                "perplexity": math.exp(log_ppl[0]),
                "train_time_s": train_time[0],
                "inference_time_s": k * inference_per_member,
            })

        best = max((g for g in grid if g["fraction"] == fraction), key=lambda g: g["accuracy"])
        solo = next(g for g in grid if g["fraction"] == fraction and g["size"] == 1)
        print(
            f"membre seul : acc {solo['accuracy'] * 100:.2f}% | "
            f"forêt de {args.max_estimators} : "
            f"{grid[-1]['accuracy'] * 100:.2f}% | "
            f"meilleure taille : {best['size']} ({best['accuracy'] * 100:.2f}%) | "
            f"entraînement {sum(train_times):.0f}s",
            flush=True,
        )

        # Sauvegarde après chaque proportion : un run de plusieurs heures
        # interrompu ne perd pas les forêts déjà évaluées (voir --resume).
        save_results(output_dir, args, test, reference_row, grid, members_info)

        # Libère la forêt avant d'entraîner la suivante. Sinon les deux
        # forêts cohabitent sur le GPU pendant l'entraînement : à 100 membres,
        # la mémoire déborde sur la RAM et tout devient 10x plus lent.
        del forest, scores, by_subset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    # Au-delà de 12 tailles, le tableau n'affiche qu'une sélection de colonnes.
    sizes = list(subsets)
    if len(sizes) > 12:
        shown = {1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100}
        sizes = [k for k in sizes if k in shown or k == sizes[-1]]

    def pivot(title, key, fmt):
        print(f"\n{title}")
        print(f"{'Neutralisé':>11} | " + "".join(f"{k:>10}" for k in sizes))
        print("-" * (14 + 10 * len(sizes)))
        for fraction in args.fractions:
            cells = {g["size"]: g[key] for g in grid if g["fraction"] == fraction}
            print(f"{fraction * 100:>10g}% | " + "".join(f"{fmt(cells[k]):>10}" for k in sizes))

    print("\n" + "=" * (14 + 10 * len(sizes)))
    print("Lignes : proportion de dimensions neutralisées ; colonnes : nombre de membres.")
    pivot("Accuracy dernier mot, vote soft (%)", "accuracy", lambda v: f"{v * 100:.2f}")
    pivot("Accuracy dernier mot, vote hard (%)", "hard_accuracy", lambda v: f"{v * 100:.2f}")
    pivot("Perplexité, vote soft", "perplexity", lambda v: f"{v:,.0f}".replace(",", " "))
    pivot("Temps d'entraînement (s)", "train_time_s", lambda v: f"{v:.0f}")
    print(
        f"\nRéférence (fine-tuné seul, corpus complet) : "
        f"acc {reference_row['accuracy'] * 100:.2f}% | "
        f"perplexité {reference_row['perplexity']:.0f} | "
        f"entraînement {reference_row['train_time_s']:.0f}s | "
        f"inférence {reference_row['inference_time_s']:.2f}s"
    )
    print(
        f"Moyenne sur jusqu'à {args.subsets_per_size} sous-ensembles de membres par taille "
        f"(un seul pour {args.max_estimators} membres)."
    )

    save_results(output_dir, args, test, reference_row, grid, members_info)
    plot_grid(grid, reference_row, args, output_dir / "grid.png")

    print(f"\nRésultats et graphique enregistrés dans {output_dir}/")


def save_results(output_dir, args, test, reference_row, grid, members_info):
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "n_test": len(test),
                "reference": reference_row,
                "grid": grid,
                "members": {str(k): v for k, v in members_info.items()},
            },
            f,
            indent=2,
        )


# ============================================================
# Graphique
# ============================================================

# Rampe bleue ordinale (claire -> foncée), validée : 5 niveaux au maximum.
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
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


def _shown_fractions(fractions):
    """Au plus 5 proportions réparties sur la liste, pour rester lisible."""

    fractions = sorted(fractions)
    if len(fractions) <= len(RAMP):
        return fractions

    idx = sorted({round(i * (len(fractions) - 1) / (len(RAMP) - 1)) for i in range(len(RAMP))})

    return [fractions[i] for i in idx]


def plot_grid(grid, reference, args, path):
    shown = _shown_fractions(args.fractions)
    # Niveaux de la rampe espacés au maximum (ex. 2 courbes : clair et foncé).
    if len(shown) == 1:
        colors = [RAMP[2]]
    else:
        start = 1 if len(shown) <= 4 else 0
        colors = [
            RAMP[start + round(i * (len(RAMP) - 1 - start) / (len(shown) - 1))]
            for i in range(len(shown))
        ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor=SURFACE)
    (ax_soft, ax_hard), (ax_ppl, ax_time) = axes

    panels = [
        (ax_soft, "size", "accuracy", 100),
        (ax_hard, "size", "hard_accuracy", 100),
        (ax_ppl, "size", "perplexity", 1),
        (ax_time, "train_time_s", "accuracy", 100),
    ]

    for fraction, color in zip(shown, colors):
        rows = sorted((g for g in grid if g["fraction"] == fraction), key=lambda g: g["size"])
        label = f"{fraction * 100:g} % neutralisé" + (" (bagging)" if fraction == 0 else "")

        for ax, x_key, y_key, scale in panels:
            x = [g[x_key] for g in rows]
            y = [g[y_key] * scale for g in rows]
            ax.plot(x, y, color=color, linewidth=2, label=label,
                    **({"marker": "o", "markersize": 5} if len(rows) <= 20 else {}))

    ref_style = dict(color=TEXT_SECONDARY, linestyle=":", linewidth=1)
    ax_soft.axhline(reference["accuracy"] * 100, label="Fine-tuné seul (corpus complet)", **ref_style)
    ax_hard.axhline(reference["accuracy"] * 100, **ref_style)
    ax_ppl.axhline(reference["perplexity"], **ref_style)
    ax_time.axhline(reference["accuracy"] * 100, **ref_style)
    ax_time.scatter(
        [reference["train_time_s"]], [reference["accuracy"] * 100],
        color=TEXT_SECONDARY, s=40, zorder=3,
    )

    _style(ax_soft, "Accuracy dernier mot, vote soft", "Nombre de membres", "Accuracy (%)")
    _style(ax_hard, "Accuracy dernier mot, vote hard", "Nombre de membres", "Accuracy (%)")
    _style(ax_ppl, "Perplexité du mot cible, vote soft", "Nombre de membres", "Perplexité (échelle log)")
    _style(ax_time, "Qualité en fonction du temps d'entraînement", "Temps d'entraînement (s)", "Accuracy (%)")
    ax_ppl.set_yscale("log")

    sizes = sorted({g["size"] for g in grid})
    if len(sizes) <= 20:
        for ax in (ax_soft, ax_hard, ax_ppl):
            ax.set_xticks(sizes)

    handles, labels = ax_soft.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=TEXT,
               loc="lower center", ncol=len(handles))
    fig.suptitle(
        f"LAMBADA : Random LLM Forest selon le nombre de membres et la taille du masque "
        f"({args.epochs} epochs par membre"
        f"{', sans early stopping' if getattr(args, 'no_early_stopping', False) else ''}"
        f"{f', moitié {args.test_half} du test' if getattr(args, 'test_half', None) else ''})",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
