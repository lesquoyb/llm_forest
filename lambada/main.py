"""
Comparaison sur LAMBADA de trois approches à partir de TinyStories-1M :

1. le modèle de base (zero-shot) ;
2. le modèle fine-tuné sur toutes les dimensions de l'embedding ;
3. une Random LLM Forest : N modèles fine-tunés, chacun sur un bootstrap du
   corpus et en ignorant une partie aléatoire des dimensions de l'embedding.

Mesures : accuracy sur le dernier mot, perplexité du mot cible, temps
d'entraînement et temps d'inférence sur le split test officiel de LAMBADA.

Install :
    pip install torch transformers datasets matplotlib tqdm

Exemples :
    python lambada/main.py
    python lambada/main.py --quick
    python lambada/main.py --epochs 5 --n-estimators 7 --mask-fraction 0.3
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_lambada, prepare_corpus
from finetune import MaskedLM, evaluate, finetune
from forest import RandomLLMForest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roneneldan/TinyStories-1M")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=5)
    parser.add_argument("--mask-fraction", type=float, default=0.25)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "outputs"))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="petit test de bout en bout (peu d'exemples, 1 epoch, 3 membres)",
    )
    args = parser.parse_args()

    if args.quick:
        args.epochs = 1
        args.n_estimators = 3
        args.max_train_samples = args.max_train_samples or 512
        args.max_eval_samples = args.max_eval_samples or 256

    return args


def timed(fn, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    result = fn()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return result, time.perf_counter() - start


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

    # --------------------------------------------------------
    # Modèle et données
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.eval()

    print(f"Taille des embeddings : {base_model.config.hidden_size}")
    print(f"Paramètres : {sum(p.numel() for p in base_model.parameters()):,}")

    train_raw, dev_raw, test_raw = load_lambada(
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
        seed=args.seed,
    )

    train = prepare_corpus(train_raw, tokenizer, args.max_length)
    dev = prepare_corpus(dev_raw, tokenizer, args.max_length)
    test = prepare_corpus(test_raw, tokenizer, args.max_length)

    print(f"Train : {len(train)} | Dev : {len(dev)} | Test : {len(test)}")
    print(
        "Mots cibles en plusieurs tokens (test) : "
        f"{sum(len(ex.target_ids) > 1 for ex in test) / len(test) * 100:.1f}%"
    )

    finetune_kwargs = dict(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Préchauffage GPU, pour que la première mesure d'inférence ne paie pas
    # l'initialisation CUDA.
    base = MaskedLM(base_model, tokenizer)
    evaluate([base], test[:64])

    results = {}

    def record(key, scores, train_time_s, inference_time_s):
        results[key] = {
            "accuracy": scores["accuracy"],
            "perplexity": scores.get("perplexity"),
            "train_time_s": train_time_s,
            "inference_time_s": inference_time_s,
        }

    # --------------------------------------------------------
    # 1. Modèle de base
    # --------------------------------------------------------

    print("\n=== Modèle de base (zero-shot) ===")

    scores, t_base = timed(lambda: evaluate([base], test), device)
    record("base", scores["soft"], 0.0, t_base)

    # --------------------------------------------------------
    # 2. Fine-tuning complet
    # --------------------------------------------------------

    print("\n=== Fine-tuning (toutes les dimensions) ===")

    tuned = finetune(
        base_model,
        tokenizer,
        train,
        dev,
        seed=args.seed,
        desc="Fine-tuning",
        **finetune_kwargs,
    )

    scores, t_tuned = timed(lambda: evaluate([tuned], test), device)
    record("finetuned", scores["soft"], tuned.train_time_s, t_tuned)

    tuned.model.save_pretrained(output_dir / "finetuned")
    tokenizer.save_pretrained(output_dir / "finetuned")

    # --------------------------------------------------------
    # 3. Random LLM Forest
    # --------------------------------------------------------

    print(
        f"\n=== Random LLM Forest ({args.n_estimators} membres, "
        f"{args.mask_fraction:.0%} des dimensions neutralisées) ==="
    )

    forest = RandomLLMForest(
        base_model,
        tokenizer,
        n_estimators=args.n_estimators,
        mask_fraction=args.mask_fraction,
        seed=args.seed,
        **finetune_kwargs,
    ).fit(train)

    scores, t_forest = timed(lambda: forest.evaluate(test), device)
    record("forest_soft", scores["soft"], forest.train_time_s, t_forest)
    record("forest_hard", scores["hard"], forest.train_time_s, t_forest)

    members = [
        {
            "masked_dims": list(member.masked_dims),
            "train_time_s": member.train_time_s,
            **member_scores,
        }
        for member, member_scores in zip(forest.members, scores["members"])
    ]

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    names = {
        "base": "Base (zero-shot)",
        "finetuned": "Fine-tuné",
        "forest_soft": "Forest (vote soft)",
        "forest_hard": "Forest (vote hard)",
    }

    print("\n" + "=" * 82)
    print(
        f"{'Modèle':<22}{'Acc':>8}{'Perplexité':>13}"
        f"{'Entraîn. (s)':>14}{'Inférence (s)':>15}{'ms/ex':>10}"
    )
    print("-" * 82)
    for key, r in results.items():
        perplexity = "-" if r["perplexity"] is None else f"{r['perplexity']:.1f}"
        print(
            f"{names[key]:<22}{r['accuracy'] * 100:>7.2f}%{perplexity:>13}"
            f"{r['train_time_s']:>14.1f}{r['inference_time_s']:>15.2f}"
            f"{r['inference_time_s'] / len(test) * 1000:>10.3f}"
        )
    print("-" * 82)
    print("\nMembres de la forêt (seuls) :")
    for i, m in enumerate(members):
        print(
            f"  Membre {i + 1}: acc {m['accuracy'] * 100:.2f}% | "
            f"perplexité {m['perplexity']:.1f} | "
            f"entraînement {m['train_time_s']:.1f}s"
        )

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "n_test": len(test),
                "results": results,
                "forest_members": members,
                "finetuned_history": tuned.history,
                "forest_histories": [m.history for m in forest.members],
            },
            f,
            indent=2,
        )

    plot_training(tuned, forest, output_dir / "training_curves.png")
    plot_comparison(results, names, output_dir / "comparison.png")

    print(f"\nRésultats, courbes et modèle fine-tuné enregistrés dans {output_dir}/")


# ============================================================
# Graphiques
# ============================================================

def plot_training(tuned, forest, path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    def curve(ax, values, **kwargs):
        if values:
            ax.plot(range(1, len(values) + 1), values, marker="o", **kwargs)

    curve(ax_loss, tuned.history["train_loss"], label="Fine-tuné - train", color="C0")
    curve(ax_loss, tuned.history["val_loss"], label="Fine-tuné - dev", color="C0", linestyle="--")
    curve(ax_acc, [a * 100 for a in tuned.history["val_acc"]], label="Fine-tuné - dev", color="C0")

    for i, member in enumerate(forest.members):
        label = "Membres - OOB" if i == 0 else None
        curve(ax_loss, member.history["val_loss"], color="C1", alpha=0.5, linestyle="--", label=label)
        curve(ax_acc, [a * 100 for a in member.history["val_acc"]], color="C1", alpha=0.5, label=label)

    ax_loss.set(xlabel="Epoch", ylabel="Loss (par token)", title="Loss")
    ax_acc.set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy de validation (dernier mot)")

    for ax in (ax_loss, ax_acc):
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(results, names, path):
    keys = list(results)
    labels = [names[k] for k in keys]
    colors = ["C7", "C0", "C1", "C2"][: len(keys)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    panels = [
        ("Accuracy dernier mot (%)", [results[k]["accuracy"] * 100 for k in keys]),
        ("Temps d'entraînement (s)", [results[k]["train_time_s"] for k in keys]),
        ("Temps d'inférence sur le test (s)", [results[k]["inference_time_s"] for k in keys]),
    ]

    for ax, (title, values) in zip(axes, panels):
        bars = ax.bar(labels, values, color=colors)
        ax.bar_label(bars, fmt="%.2f")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
