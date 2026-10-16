"""
BoolQ : la Random LLM Forest face à d'autres ensembles de même taille.

Toutes les configurations ont N membres et sont appariées : même graine de
départ, donc mêmes masques et mêmes échantillons bootstrap quand ils servent.

1. Graines             : N modèles fine-tunés sur tout le corpus, sans masque,
                         avec des graines différentes (ordre des données).
                         Le membre 1 est exactement le "fine-tuné seul".
2. Bagging             : N modèles fine-tunés sur des bootstraps, sans masque.
3. Random LLM Forest   : bootstrap + masque aléatoire.

Lire 1 -> 2 donne l'effet du bootstrap ; 2 -> 3 l'effet des masques.

Exemples :
    python boolq/ensembles.py
    python boolq/ensembles.py --quick
    python boolq/ensembles.py --n-estimators 7 --epochs 8
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from boolq_data import load_boolq, prepare_corpus
from finetune import MaskedLM
from forest import RandomLLMForest
from main import parse_args, quality, timed


def main():
    args = parse_args()
    if args.output_dir == str(Path(__file__).parent / "outputs"):
        args.output_dir = str(Path(__file__).parent / "outputs" / "ensembles")

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

    train_raw, dev_raw, test_raw = load_boolq(
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
        seed=args.seed,
    )
    train = prepare_corpus(train_raw, tokenizer, args.max_length)
    dev = prepare_corpus(dev_raw, tokenizer, args.max_length)
    test = prepare_corpus(test_raw, tokenizer, args.max_length)
    labels = torch.tensor([ex.label for ex in test])

    majority = max(labels.float().mean().item(), 1 - labels.float().mean().item())

    print(f"Train : {len(train)} | Dev : {len(dev)} | Test : {len(test)}")
    print(f"Baseline classe majoritaire sur le test : {majority * 100:.2f}%")

    finetune_kwargs = dict(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    base = MaskedLM(base_model, tokenizer)
    base.predict_proba(test[:64])  # préchauffage GPU

    rows = {}

    def record(key, name, p_yes, train_time_s, inference_time_s, member_p=None):
        row = {
            "name": name,
            **quality(p_yes, labels),
            "train_time_s": train_time_s,
            "inference_time_s": inference_time_s,
        }

        if member_p is not None:
            hard = quality((member_p > 0.5).float().mean(0), labels, probabilistic=False)
            member_acc = torch.tensor([quality(p, labels)["accuracy"] for p in member_p])
            row["hard_accuracy"] = hard["accuracy"]
            row["hard_balanced_accuracy"] = hard["balanced_accuracy"]
            row["member_accuracy_mean"] = member_acc.mean().item()
            row["member_accuracy_std"] = member_acc.std().item() if len(member_acc) > 1 else 0.0

        rows[key] = row

    # --------------------------------------------------------
    # Modèle de base seul
    # --------------------------------------------------------

    p, t = timed(lambda: base.predict_proba(test), device)
    record("base", "Base (zero-shot)", p, 0.0, t)

    # --------------------------------------------------------
    # Ensembles
    # --------------------------------------------------------

    configs = [
        ("seeds", "FT, graines (sans masque)",
         dict(bootstrap=False, mask_fraction=0.0)),
        ("bagging", "FT, bagging (sans masque)",
         dict(bootstrap=True, mask_fraction=0.0)),
        ("forest", "Random LLM Forest",
         dict(bootstrap=True, mask_fraction=args.mask_fraction)),
    ]

    ensembles = {}

    for key, name, options in configs:
        print(f"\n=== {name} ({args.n_estimators} membres) ===")

        ensemble = RandomLLMForest(
            base_model,
            tokenizer,
            n_estimators=args.n_estimators,
            seed=args.seed,
            **options,
            **finetune_kwargs,
        ).fit(train, dev)
        ensembles[key] = ensemble

        member_p, t = timed(
            lambda: torch.stack([m.predict_proba(test) for m in ensemble.members]),
            device,
        )
        record(key, name, member_p.mean(0), ensemble.train_time_s, t, member_p)

        if key == "seeds":
            single = ensemble.members[0]
            p, t_single = timed(lambda: single.predict_proba(test), device)
            record("finetuned", "Fine-tuné seul", p, single.train_time_s, t_single)

    order = ["base", "finetuned", "seeds", "bagging", "forest"]
    rows = {key: rows[key] for key in order}

    # --------------------------------------------------------
    # Résultats
    # --------------------------------------------------------

    print("\n" + "=" * 128)
    print(
        f"{'Configuration':<28}{'Acc soft':>10}{'Acc hard':>10}{'Bal. acc':>10}"
        f"{'Log-loss':>10}{'% yes':>8}{'Membres (moy.)':>17}{'Gain':>8}"
        f"{'Entraîn. (s)':>14}{'Inférence (s)':>15}"
    )
    print("-" * 128)
    for row in rows.values():
        members = row.get("member_accuracy_mean")
        gain = "-" if members is None else f"{(row['accuracy'] - members) * 100:+.2f}"
        print(
            f"{row['name']:<28}{row['accuracy'] * 100:>9.2f}%"
            f"{_pct(row.get('hard_accuracy')):>10}"
            f"{row['balanced_accuracy'] * 100:>9.2f}%"
            f"{row['log_loss']:>10.4f}{row['yes_rate'] * 100:>7.1f}%"
            f"{_pct(members, row.get('member_accuracy_std')):>17}"
            f"{gain:>8}"
            f"{row['train_time_s']:>14.1f}{row['inference_time_s']:>15.2f}"
        )
    print("-" * 128)
    print(f"Classe majoritaire : {majority * 100:.2f}%")
    print("Gain = accuracy de l'ensemble (vote soft) - accuracy moyenne d'un membre seul.")

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "n_test": len(test),
                "majority_baseline": majority,
                "results": rows,
                "masks": {
                    key: [list(m.masked_dims) for m in e.members]
                    for key, e in ensembles.items()
                },
            },
            f,
            indent=2,
        )

    plot_comparison(
        rows,
        output_dir / "comparison.png",
        title=f"BoolQ : ensembles de {args.n_estimators} membres",
        accuracy_label="Accuracy (%)",
        reference=(majority * 100, "classe majoritaire"),
    )

    print(f"\nRésultats et graphique enregistrés dans {output_dir}/")


def _pct(value, std=None):
    if value is None:
        return "-"
    if std is None:
        return f"{value * 100:.2f}%"
    return f"{value * 100:.1f}±{std * 100:.1f}%"


# ============================================================
# Graphique
# ============================================================

BAR = "#2a78d6"
MEMBER = "#eb6834"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def plot_comparison(rows, path, title, accuracy_label, reference=None):
    names = [row["name"] for row in rows.values()][::-1]
    values = list(rows.values())[::-1]
    y = range(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE, sharey=True)

    panels = [
        (accuracy_label, [r["accuracy"] * 100 for r in values], "{:.1f}"),
        ("Temps d'entraînement (s)", [r["train_time_s"] for r in values], "{:.0f}"),
        ("Temps d'inférence sur le test (s)", [r["inference_time_s"] for r in values], "{:.2f}"),
    ]

    for ax, (panel_title, data, fmt) in zip(axes, panels):
        bars = ax.barh(y, data, color=BAR, height=0.6, label="Ensemble (vote soft)")
        ax.bar_label(bars, labels=[fmt.format(v) for v in data], padding=4,
                     fontsize=9, color=TEXT_SECONDARY)

        ax.set_facecolor(SURFACE)
        ax.set_title(panel_title, color=TEXT, fontsize=11, loc="left")
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.set_xlim(0, max(data) * 1.18 if max(data) > 0 else 1)

    axes[0].set_yticks(list(y), names, color=TEXT)

    member_y = [i for i, r in zip(y, values) if "member_accuracy_mean" in r]
    member_x = [r["member_accuracy_mean"] * 100 for r in values if "member_accuracy_mean" in r]
    axes[0].scatter(
        member_x, member_y, color=MEMBER, s=64, zorder=3,
        edgecolors=SURFACE, linewidths=2, label="Membre seul (moyenne)",
    )

    if reference is not None:
        axes[0].axvline(reference[0], color=TEXT_SECONDARY, linestyle=":", linewidth=1,
                        label=reference[1])

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=9, labelcolor=TEXT,
               loc="lower center", ncol=len(handles))

    fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
