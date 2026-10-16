"""
LAMBADA : benchmark de l'apprentissage en contexte (voir icl.py).

Pour chaque modèle, nombre de démonstrations, stratégie de choix et taille
de masque, on évalue sur la moitié "b" du test (celle des forêts et des
modèles fine-tunés) :
- un membre seul (moyenne sur les membres) ;
- des ensembles de 5, 10... membres qui diffèrent par leurs démonstrations
  et/ou leur masque (moyenne sur plusieurs sous-ensembles).

Les résultats sont enregistrés après chaque configuration et un run
interrompu reprend là où il s'était arrêté (--resume).

Exemples :
    python lambada/icl_bench.py
    python lambada/icl_bench.py --quick
    python lambada/icl_bench.py --models roneneldan/TinyStories-28M --shots 0 4 8 --max-length 1024
    python lambada/icl_bench.py --strategies similar --mask-fractions 0 0.125 0.25 --resume
"""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_lambada, prepare_corpus
from finetune import evaluate
from forest_grid import draw_subsets
from icl import STRATEGIES, IclConfig, log_mean, make_icl_members

OUTPUTS = Path(__file__).parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "roneneldan/TinyStories-1M",
            "roneneldan/TinyStories-28M",
            "roneneldan/TinyStories-33M",
        ],
    )
    parser.add_argument("--shots", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    parser.add_argument("--mask-fractions", type=float, nargs="+", default=[0.0, 0.125])
    parser.add_argument("--n-members", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--subsets-per-size", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-half", choices=["a", "b"], default="b")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(OUTPUTS / "icl"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="petit test de bout en bout (1M, 64 questions, 3 membres)",
    )
    args = parser.parse_args()

    if args.quick:
        args.models = ["roneneldan/TinyStories-1M"]
        args.shots = [0, 2]
        args.n_members = 3
        args.sizes = [1, 3]
        args.max_eval_samples = args.max_eval_samples or 64

    args.sizes = sorted({k for k in args.sizes if k <= args.n_members} | {1})

    return args


def config_key(model, config):
    strategy = config.strategy if config.shots > 0 else "-"
    return f"{model}|{config.shots}|{strategy}|{config.mask_fraction:g}|{config.max_length}"


def configs_for(args):
    """Toutes les configurations ; à 0 démonstration, la stratégie n'a pas de sens."""

    seen = set()
    for shots in args.shots:
        for strategy in (args.strategies if shots > 0 else args.strategies[:1]):
            for mask in args.mask_fractions:
                config = IclConfig(shots, strategy, mask, args.max_length, args.separator)
                key = (shots, strategy if shots > 0 else "-", mask)
                if key not in seen:
                    seen.add(key)
                    yield config


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    rows = []
    if args.resume and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        print(f"Reprise : {len(rows)} configurations relues", flush=True)
    done = {row["key"] for row in rows}

    train_raw, _, test_raw = load_lambada(
        max_eval=args.max_eval_samples, seed=args.seed, test_half=args.test_half
    )
    subsets = draw_subsets(args.n_members, args.subsets_per_size, random.Random(args.seed))

    for model_name in args.models:
        configs = [c for c in configs_for(args) if config_key(model_name, c) not in done]
        if not configs:
            continue

        short = model_name.split("/")[-1]
        print(f"\n=== {short} : {len(configs)} configurations ===", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        model.config.pad_token_id = tokenizer.pad_token_id

        pool = prepare_corpus(train_raw, tokenizer)
        queries = prepare_corpus(test_raw, tokenizer)

        for config in configs:
            n_members = args.n_members if config.diverse else 1
            members, corpora, stats = make_icl_members(
                model, tokenizer, pool, queries, config, n_members, seed=args.seed
            )
            sizes = [k for k in args.sizes if k <= n_members]
            flat = [idx for k in sizes for idx in subsets[k]] if n_members > 1 else None

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            scores = evaluate(
                members, None, batch_size=args.batch_size, subsets=flat, member_corpora=corpora
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start

            by_size = {}
            if n_members == 1:
                by_size[1] = {
                    "accuracy": scores["soft"]["accuracy"],
                    "perplexity": scores["soft"]["perplexity"],
                    "hard_accuracy": scores["hard"]["accuracy"],
                }
            else:
                results = iter(scores["subsets"])
                for k in sizes:
                    group = [next(results) for _ in subsets[k]]
                    by_size[k] = {
                        "accuracy": statistics.mean(r["soft"]["accuracy"] for r in group),
                        "perplexity": log_mean([r["soft"]["perplexity"] for r in group]),
                        "hard_accuracy": statistics.mean(r["hard"]["accuracy"] for r in group),
                    }

            row = {
                "key": config_key(model_name, config),
                "model": model_name,
                **config.as_dict(),
                "strategy": config.strategy if config.shots > 0 else "-",
                "n_members": n_members,
                "n_test": len(queries),
                **stats,
                # Temps d'évaluation, calcul des sous-ensembles compris (négligeable).
                "inference_time_per_member_s": elapsed / n_members,
                "by_size": {str(k): v for k, v in by_size.items()},
            }
            rows.append(row)

            best = by_size[max(by_size)]
            print(
                f"{config.shots} démo(s) | {row['strategy']:<7} | masque "
                f"{config.mask_fraction * 100:>4g} % | membre seul "
                f"{by_size[1]['accuracy'] * 100:5.2f}% | ensemble de {max(by_size)} "
                f"{best['accuracy'] * 100:5.2f}% (ppl {best['perplexity']:,.0f}) | "
                f"{stats['shots_used_mean']:.1f} démos, prompt {stats['prompt_length_mean']:.0f} "
                f"tokens | {elapsed / n_members:.1f}s/membre",
                flush=True,
            )

            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"config": vars(args), "rows": rows}, f, indent=2)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_table(rows)
    plot_icl(rows, finetuned_references(), output_dir / "icl.png")
    print(f"\nRésultats et graphique enregistrés dans {output_dir}/")


def print_table(rows):
    print("\n" + "=" * 118)
    print(
        f"{'Modèle':<20}{'Démos':>6}{'Stratégie':>10}{'Masque':>8}{'Seul':>9}"
        f"{'Ensemble':>18}{'Perplexité ens.':>17}{'Prompt':>9}{'s/membre':>10}"
    )
    print("-" * 118)
    for r in sorted(rows, key=lambda r: (r["model"], r["shots"], r["strategy"], r["mask_fraction"])):
        sizes = sorted(int(k) for k in r["by_size"])
        solo, best = r["by_size"]["1"], r["by_size"][str(sizes[-1])]
        ens = f"{best['accuracy'] * 100:.2f}% (x{sizes[-1]})" if sizes[-1] > 1 else "-"
        print(
            f"{r['model'].split('/')[-1]:<20}{r['shots']:>6}{r['strategy']:>10}"
            f"{r['mask_fraction'] * 100:>7g}%{solo['accuracy'] * 100:>8.2f}%{ens:>18}"
            f"{best['perplexity']:>17,.0f}{r['prompt_length_mean']:>9.0f}"
            f"{r['inference_time_per_member_s']:>10.1f}"
        )


def finetuned_references():
    """Accuracy des modèles fine-tunés (même moitié b du test), si disponible."""

    refs = {}
    single = OUTPUTS / "single_models" / "results.json"
    if single.exists():
        for r in json.load(open(single, encoding="utf-8"))["rows"]:
            if r.get("finetuned") and r.get("model"):
                refs[r["model"]] = r["accuracy"]
    grid = OUTPUTS / "forest_grid_100m" / "results.json"
    if grid.exists():
        results = json.load(open(grid, encoding="utf-8"))
        if results["config"].get("test_half") == "b":
            refs.setdefault(results["config"]["model"], results["reference"]["accuracy"])
    return refs


# ============================================================
# Graphique
# ============================================================

STRATEGY_COLORS = {"fixed": "#2a78d6", "random": "#eb6834", "similar": "#1baf7a"}
STRATEGY_LABELS = {
    "fixed": "démos fixes par membre",
    "random": "démos au hasard par question",
    "similar": "démos les plus proches",
}
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_icl(rows, references, path):
    models = sorted({r["model"] for r in rows}, key=lambda m: [r["model"] for r in rows].index(m))
    masks = sorted({r["mask_fraction"] for r in rows})
    styles = ["-", "--", ":", "-."]

    fig, axes = plt.subplots(
        2, len(models), figsize=(5.5 * len(models), 8.5), facecolor=SURFACE,
        squeeze=False, sharey="row",
    )

    for col, model in enumerate(models):
        model_rows = [r for r in rows if r["model"] == model]

        for row_i, which in enumerate(("solo", "ensemble")):
            ax = axes[row_i][col]

            for strategy, color in STRATEGY_COLORS.items():
                for mask, style in zip(masks, styles):
                    points = []
                    for r in model_rows:
                        if r["mask_fraction"] != mask or r["strategy"] not in (strategy, "-"):
                            continue
                        sizes = sorted(int(k) for k in r["by_size"])
                        k = 1 if which == "solo" else sizes[-1]
                        points.append((r["shots"], r["by_size"][str(k)]["accuracy"] * 100))
                    if len(points) > 1:
                        points.sort()
                        ax.plot(*zip(*points), color=color, linestyle=style, linewidth=2,
                                marker="o", markersize=5)

            if model in references:
                ax.axhline(references[model] * 100, color=TEXT_SECONDARY, linestyle=":",
                           linewidth=1)
                ax.annotate("fine-tuné", (0, references[model] * 100), xytext=(2, 3),
                            textcoords="offset points", fontsize=8, color=TEXT_SECONDARY)

            title = model.split("/")[-1] + (" : membre seul" if which == "solo" else " : ensemble")
            ax.set_facecolor(SURFACE)
            ax.set_title(title, color=TEXT, fontsize=11, loc="left")
            ax.set_xlabel("Démonstrations dans le prompt", color=TEXT_SECONDARY)
            ax.set_ylabel("Accuracy dernier mot (%)", color=TEXT_SECONDARY)
            ax.set_xticks(sorted({r["shots"] for r in model_rows}))
            ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)

    handles = [
        plt.Line2D([], [], color=c, linewidth=2, label=STRATEGY_LABELS[s])
        for s, c in STRATEGY_COLORS.items()
    ] + [
        plt.Line2D([], [], color=TEXT_SECONDARY, linestyle=st, linewidth=2,
                   label=f"masque {m * 100:g} %")
        for m, st in zip(masks, styles)
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=TEXT,
               loc="lower center", ncol=len(handles))
    fig.suptitle(
        "LAMBADA (moitié b du test) : apprentissage en contexte, sans fine-tuning",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
