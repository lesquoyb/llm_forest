"""
LAMBADA : modèles TinyStories plus gros, seuls (zero-shot) puis fine-tunés,
comparés aux Random LLM Forests de TinyStories-1M.

Même protocole que forest_grid.py --test-half b :
- entraînement sur le même train (4 382 passages) ;
- choix de la meilleure epoch sur la moitié "a" du test ;
- évaluation sur la moitié "b" du test, celle des forêts.

Les lignes "forêt" sont relues dans les results.json de forest_grid.py
(--forest-results). --resume réutilise les modèles seuls déjà évalués.

Exemples :
    python lambada/single_models.py
    python lambada/single_models.py --models roneneldan/TinyStories-33M --epochs 5
    python lambada/single_models.py --quick
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_lambada, prepare_corpus
from finetune import MaskedLM, evaluate, finetune
from main import timed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["roneneldan/TinyStories-28M", "roneneldan/TinyStories-33M"],
    )
    parser.add_argument("--epochs", type=int, default=10)
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
        "--forest-results",
        nargs="+",
        default=[
            str(Path(__file__).parent / "outputs" / "forest_grid_100m" / "results.json"),
            str(Path(__file__).parent / "outputs" / "forest_grid_28M" / "results.json"),
        ],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="réutilise les modèles seuls déjà présents dans results.json",
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent / "outputs" / "single_models")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="petit test de bout en bout (peu d'exemples, 1 epoch)",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.max_train_samples = args.max_train_samples or 256
        args.max_eval_samples = args.max_eval_samples or 128

    return args


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.format == "openai":
        # Moitié a = entraînement + passages de sélection, moitié b = test.
        train_raw, selection_raw, test_raw = load_lambada(
            max_train=args.max_train_samples,
            max_eval=args.max_eval_samples,
            seed=args.seed,
            fmt="openai",
        )
    else:
        train_raw, _, selection_raw = load_lambada(
            max_train=args.max_train_samples,
            max_eval=args.max_eval_samples,
            seed=args.seed,
            test_half="a",
        )
        _, _, test_raw = load_lambada(
            max_eval=args.max_eval_samples, seed=args.seed, test_half="b"
        )

    rows = []
    results_path = output_dir / "results.json"
    if args.resume and results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            rows = [r for r in json.load(f)["rows"] if r.get("model")]
    done = {r["model"] for r in rows}

    for name in args.models:
        if name in done:
            print(f"\n{name} : déjà évalué, réutilisé", flush=True)
            continue

        short = name.split("/")[-1]
        print(f"\n=== {short} ===", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(name).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        n_embedding = model.get_input_embeddings().weight.numel()
        print(
            f"hidden {model.config.hidden_size} | couches {model.config.num_layers} | "
            f"{n_params / 1e6:.1f} M paramètres dont {n_embedding / 1e6:.1f} M d'embeddings",
            flush=True,
        )

        train = prepare_corpus(train_raw, tokenizer, args.max_length)
        selection = prepare_corpus(selection_raw, tokenizer, args.max_length)
        test = prepare_corpus(test_raw, tokenizer, args.max_length)

        base = MaskedLM(model, tokenizer)
        evaluate([base], test[:64])  # préchauffage GPU
        scores, t_base = timed(lambda: evaluate([base], test), device)
        rows.append({
            "name": f"{short} seul",
            "model": name,
            "finetuned": False,
            "params": n_params,
            "accuracy": scores["soft"]["accuracy"],
            "perplexity": scores["soft"]["perplexity"],
            "train_time_s": 0.0,
            "inference_time_s": t_base,
        })
        print(
            f"seul : acc {rows[-1]['accuracy'] * 100:.2f}% | "
            f"perplexité {rows[-1]['perplexity']:.0f} | inférence {t_base:.2f}s",
            flush=True,
        )

        tuned = finetune(
            model,
            tokenizer,
            train,
            selection,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=args.seed,
            desc=short,
        )
        scores, t_tuned = timed(lambda: evaluate([tuned], test), device)
        best_epoch = 1 + min(
            range(len(tuned.history["val_loss"])), key=tuned.history["val_loss"].__getitem__
        )
        rows.append({
            "name": f"{short} fine-tuné",
            "model": name,
            "finetuned": True,
            "params": n_params,
            "best_epoch": best_epoch,
            "accuracy": scores["soft"]["accuracy"],
            "perplexity": scores["soft"]["perplexity"],
            "train_time_s": tuned.train_time_s,
            "inference_time_s": t_tuned,
            "history": tuned.history,
        })
        print(
            f"fine-tuné : acc {rows[-1]['accuracy'] * 100:.2f}% | "
            f"perplexité {rows[-1]['perplexity']:.0f} | meilleure epoch {best_epoch} | "
            f"entraînement {tuned.train_time_s:.0f}s | inférence {t_tuned:.2f}s",
            flush=True,
        )

        del model, base, tuned
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Comparaison avec les forêts de TinyStories-1M
    # --------------------------------------------------------

    forest_rows = [r for path in args.forest_results for r in load_forest_rows(path, args.format)]
    # Un modèle fine-tuné évalué ici remplace la référence du même modèle dans forest_grid.
    names = {r["name"] for r in rows}
    all_rows = [r for r in forest_rows if r["name"] not in names] + rows

    print("\n" + "=" * 100)
    print("LAMBADA, moitié b du test : modèles seuls vs forêts")
    print(
        f"{'Modèle':<34}{'Params':>10}{'Acc':>9}{'Perplexité':>12}"
        f"{'Entraîn. (s)':>14}{'Inférence (s)':>15}"
    )
    print("-" * 100)
    for r in all_rows:
        print(
            f"{r['name']:<34}{r['params'] / 1e6:>8.1f} M{r['accuracy'] * 100:>8.2f}%"
            f"{r['perplexity']:>12.0f}{r['train_time_s']:>14.0f}{r['inference_time_s']:>15.2f}"
        )
    print("-" * 100)
    print(
        "Params d'une forêt = somme de ses membres. Temps d'inférence d'une forêt = "
        "membres l'un après l'autre."
    )

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "rows": all_rows}, f, indent=2)

    plot_comparison(all_rows, output_dir / "comparison.png")

    print(f"\nRésultats et graphique enregistrés dans {output_dir}/")


# Pour les résultats de forest_grid enregistrés avant que la référence ne
# contienne son nombre de paramètres.
KNOWN_PARAMS = {"roneneldan/TinyStories-1M": 3_745_984}


def load_forest_rows(path, fmt="cimec"):
    """Référence fine-tunée et forêts de 10, 20, 100 membres, relues dans forest_grid."""

    path = Path(path)
    if not path.exists():
        print(f"(pas de résultats de forêt dans {path})")
        return []

    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    config = results["config"]
    if config.get("format", "cimec") != fmt:
        print(f"(résultats de forêt ignorés : format {config.get('format', 'cimec')}, pas {fmt})")
        return []
    if fmt == "cimec" and config.get("test_half") != "b":
        print("(résultats de forêt ignorés : ils ne sont pas sur la moitié b du test)")
        return []

    model = results["config"]["model"]
    short = model.split("/")[-1]
    ref = results["reference"]
    params = ref.get("params", KNOWN_PARAMS.get(model, 0))
    rows = [{
        "name": f"{short} fine-tuné",
        "params": params,
        "accuracy": ref["accuracy"],
        "perplexity": ref["perplexity"],
        "train_time_s": ref["train_time_s"],
        "inference_time_s": ref["inference_time_s"],
    }]

    grid = {(g["fraction"], g["size"]): g for g in results["grid"]}
    fractions = sorted({g["fraction"] for g in results["grid"]})

    for size in (10, 20, 100):
        for fraction in fractions:
            g = grid.get((fraction, size))
            if g is None:
                continue
            label = "bagging" if fraction == 0 else f"masque {fraction * 100:g} %"
            rows.append({
                "name": f"Forêt {short.replace('TinyStories-', '')} x{size} ({label})",
                "params": params * size,
                "accuracy": g["accuracy"],
                "perplexity": g["perplexity"],
                "train_time_s": g["train_time_s"],
                "inference_time_s": g["inference_time_s"],
            })

    return rows


# ============================================================
# Graphique
# ============================================================

FOREST = "#2a78d6"
SINGLE = "#eb6834"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_comparison(rows, path):
    rows = sorted(rows, key=lambda r: r["accuracy"])
    names = [r["name"] for r in rows]
    colors = [FOREST if r["name"].startswith("Forêt") else SINGLE for r in rows]
    y = range(len(rows))

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 0.42 * len(rows) + 2), facecolor=SURFACE, sharey=True
    )

    panels = [
        ("Accuracy dernier mot (%)", [r["accuracy"] * 100 for r in rows], "{:.1f}", False),
        ("Perplexité (échelle log)", [r["perplexity"] for r in rows], "{:,.0f}", True),
        ("Temps d'entraînement (s)", [r["train_time_s"] for r in rows], "{:,.0f}", False),
    ]

    for ax, (title, values, fmt, log) in zip(axes, panels):
        bars = ax.barh(y, values, color=colors, height=0.6)
        ax.bar_label(
            bars, labels=[fmt.format(v).replace(",", " ") for v in values],
            padding=4, fontsize=8, color=TEXT_SECONDARY,
        )
        if log:
            ax.set_xscale("log")
        else:
            ax.set_xlim(0, max(values) * 1.2)
        ax.set_facecolor(SURFACE)
        ax.set_title(title, color=TEXT, fontsize=11, loc="left")
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    axes[0].set_yticks(list(y), names, color=TEXT)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FOREST, label="Forêt de TinyStories-1M"),
        plt.Rectangle((0, 0), 1, 1, color=SINGLE, label="Modèle seul"),
    ]
    fig.legend(handles=handles, frameon=False, fontsize=9, labelcolor=TEXT,
               loc="lower center", ncol=2)
    fig.suptitle(
        "LAMBADA (moitié b du test) : modèles seuls vs Random LLM Forests",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
