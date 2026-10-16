"""
LAMBADA : combien d'epochs faut-il à chaque membre d'une Random LLM Forest ?

Pour chaque proportion de dimensions neutralisées, on entraîne N membres en
parallèle, une epoch à la fois (mêmes masques, bootstraps et graines que
RandomLLMForest). Après chaque epoch, on évalue sur la moitié "a" du test
(ensemble de sélection) et, pour comparaison, sur le dev :

- "fixe" : la forêt dont tous les membres ont exactement e epochs ;
- "early stopping" : la forêt dont chaque membre garde sa meilleure epoch
  (<= e) selon sa loss out-of-bag. C'est ce que fait RandomLLMForest avec un
  budget de e epochs.

Les forêts de 1, 3, 5... membres sont des sous-ensembles des N membres
(moyenne sur plusieurs tirages). Le dev vient des mêmes romans que le train :
il est plus facile que le test et favorise la mémorisation. On choisit donc
le nombre d'epochs sur la moitié "a" du test ; la moitié "b" reste réservée à
l'évaluation finale (forest_grid.py --test-half b).

Exemples :
    python lambada/forest_epochs.py
    python lambada/forest_epochs.py --quick
    python lambada/forest_epochs.py --max-epochs 30 --n-members 10
"""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import collate, load_lambada, prepare_corpus
from finetune import (
    MaskedLM, clone_model, evaluate, make_dim_mask, random_masked_dims, train_step,
)
from forest_grid import draw_subsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roneneldan/TinyStories-1M")
    parser.add_argument("--n-members", type=int, default=10)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.125])
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--subsets-per-size", type=int, default=20)
    parser.add_argument("--max-oob-samples", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--format", choices=["cimec", "openai"], default="cimec",
        help="openai = texte propre ; la sélection est alors une part de la moitié a (--dev-size)",
    )
    parser.add_argument("--dev-size", type=int, default=500, help="passages de sélection (format openai)")
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).parent / "outputs" / "forest_epochs")
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="petit test de bout en bout (peu d'exemples, 3 epochs, 3 membres)",
    )
    args = parser.parse_args()

    if args.quick:
        args.max_epochs = 3
        args.n_members = 3
        args.sizes = [1, 3]
        args.fractions = [0.0, 0.125]
        args.max_train_samples = args.max_train_samples or 512
        args.max_eval_samples = args.max_eval_samples or 128

    args.sizes = [k for k in args.sizes if k <= args.n_members]

    return args


class Member:
    """Un membre en cours d'entraînement, avec sa meilleure version out-of-bag."""

    def __init__(self, base_model, tokenizer, train_data, oob_data, masked_dims, seed, args):
        torch.manual_seed(seed)

        self.model = clone_model(base_model)
        self.best_model = clone_model(base_model)
        self.masked_dims = masked_dims
        self.dim_mask = make_dim_mask(
            masked_dims, base_model.config.hidden_size, next(base_model.parameters()).device
        )
        self.oob_data = oob_data
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id),
            generator=torch.Generator().manual_seed(seed),
        )
        self.tokenizer = tokenizer
        self.best_oob_loss = float("inf")
        self.best_epoch = 0

    def current(self):
        return MaskedLM(self.model, self.tokenizer, self.masked_dims)

    def best(self):
        return MaskedLM(self.best_model, self.tokenizer, self.masked_dims)

    def train_epoch(self):
        self.model.train()
        for batch in self.loader:
            train_step(self.model, self.optimizer, batch, self.dim_mask)
        # Libère les gradients entre deux epochs : avec N membres en parallèle
        # (et un modèle de 28M), ils pèseraient autant que les modèles.
        self.optimizer.zero_grad(set_to_none=True)
        self.model.eval()

    def update_best(self, epoch, max_length):
        """Loss out-of-bag de l'epoch courante ; garde les poids si c'est la meilleure."""

        oob_loss = evaluate([self.current()], self.oob_data, max_length=max_length)["soft"]["loss"]

        if oob_loss < self.best_oob_loss:
            self.best_oob_loss = oob_loss
            self.best_epoch = epoch
            self.best_model.load_state_dict(self.model.state_dict())

        return oob_loss


def make_members(base_model, tokenizer, data, fraction, args):
    """Mêmes tirages (masques, bootstraps, graines) que RandomLLMForest.fit."""

    hidden_size = base_model.config.hidden_size
    rng = random.Random(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    members = []

    for i in range(args.n_members):
        masked_dims = random_masked_dims(hidden_size, fraction, generator)
        indices = [rng.randrange(len(data)) for _ in range(len(data))]
        oob = sorted(set(range(len(data))) - set(indices))
        rng.shuffle(oob)

        members.append(Member(
            base_model,
            tokenizer,
            [data[j] for j in indices],
            [data[j] for j in oob[:args.max_oob_samples]],
            masked_dims,
            args.seed + i,
            args,
        ))

    return members


def forest_scores(models, eval_data, subsets, args):
    """Scores (loss, accuracy) de chaque taille de forêt, moyennés sur les sous-ensembles."""

    flat = [idx for k in args.sizes for idx in subsets[k]]
    results = iter(evaluate(models, eval_data, max_length=args.max_length, subsets=flat)["subsets"])

    scores = {}
    for k in args.sizes:
        rows = [next(results)["soft"] for _ in subsets[k]]
        scores[k] = {
            "loss": sum(r["loss"] for r in rows) / len(rows),
            "accuracy": sum(r["accuracy"] for r in rows) / len(rows),
        }

    return scores


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

    if args.format == "openai":
        # Texte propre : entraînement et sélection dans la moitié a, la moitié b
        # reste intacte pour l'évaluation finale.
        train_raw, selection_raw, _ = load_lambada(
            max_train=args.max_train_samples,
            max_eval=args.max_eval_samples,
            seed=args.seed,
            fmt="openai",
            openai_dev_size=args.dev_size,
        )
        train = prepare_corpus(train_raw, tokenizer, args.max_length)
        eval_sets = {"selection": prepare_corpus(selection_raw, tokenizer, args.max_length)}
    else:
        train_raw, dev_raw, selection_raw = load_lambada(
            max_train=args.max_train_samples,
            max_eval=args.max_eval_samples,
            seed=args.seed,
            test_half="a",
        )
        train = prepare_corpus(train_raw, tokenizer, args.max_length)
        eval_sets = {
            "selection": prepare_corpus(selection_raw, tokenizer, args.max_length),
            "dev": prepare_corpus(dev_raw, tokenizer, args.max_length),
        }

    print(
        f"Train : {len(train)} | "
        + " | ".join(f"{name} : {len(data)}" for name, data in eval_sets.items())
    )

    subsets = draw_subsets(args.n_members, args.subsets_per_size, random.Random(args.seed))
    history = {}

    for fraction in args.fractions:
        print(
            f"\n=== {fraction * 100:g} % neutralisé : {args.n_members} membres, "
            f"jusqu'à {args.max_epochs} epochs ===",
            flush=True,
        )

        members = make_members(base_model, tokenizer, train, fraction, args)
        rows = []

        for epoch in range(1, args.max_epochs + 1):
            start = time.perf_counter()
            for member in members:
                member.train_epoch()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_time = time.perf_counter() - start

            oob_losses = [m.update_best(epoch, args.max_length) for m in members]

            row = {
                "epoch": epoch,
                "train_time_s": epoch_time,
                "oob_loss_mean": sum(oob_losses) / len(oob_losses),
                "members_improving": sum(m.best_epoch == epoch for m in members),
            }
            for set_name, eval_data in eval_sets.items():
                row[set_name] = {
                    "fixed": forest_scores(
                        [m.current() for m in members], eval_data, subsets, args
                    ),
                    "early_stopping": forest_scores(
                        [m.best() for m in members], eval_data, subsets, args
                    ),
                }
            rows.append(row)

            big = args.sizes[-1]
            sel = row["selection"]
            dev_text = ""
            if "dev" in row:
                dev = row["dev"]
                dev_text = (
                    f" | dev : fixe {dev['fixed'][big]['accuracy'] * 100:5.2f}%, "
                    f"early stopping {dev['early_stopping'][big]['accuracy'] * 100:5.2f}%"
                )
            print(
                f"epoch {epoch:>2} | forêt de {big}, sélection : fixe "
                f"{sel['fixed'][big]['accuracy'] * 100:5.2f}% "
                f"(loss {sel['fixed'][big]['loss']:.3f}), early stopping "
                f"{sel['early_stopping'][big]['accuracy'] * 100:5.2f}% "
                f"(loss {sel['early_stopping'][big]['loss']:.3f}){dev_text} | "
                f"membre seul (sélection) {sel['fixed'][1]['accuracy'] * 100:5.2f}% | "
                f"{row['members_improving']}/{len(members)} membres en progrès (OOB) | "
                f"{epoch_time:.0f}s",
                flush=True,
            )

        history[fraction] = {
            "rows": rows,
            "best_oob_epochs": [m.best_epoch for m in members],
        }

        del members
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------

    print("\n" + "=" * 100)
    print("Meilleure epoch (forêt, vote soft), par taille de forêt :")
    summary = {}

    for fraction, h in history.items():
        summary[fraction] = {}
        print(f"\n{fraction * 100:g} % neutralisé "
              f"(meilleures epochs OOB des membres : {sorted(h['best_oob_epochs'])})")

        for set_name, set_label in (("selection", "sélection"), ("dev", "dev")):
            if set_name not in eval_sets:
                continue
            for k in args.sizes:
                for variant in ("fixed", "early_stopping"):
                    def score(r, metric):
                        return r[set_name][variant][k][metric]

                    best_loss = min(h["rows"], key=lambda r: score(r, "loss"))
                    best_acc = max(h["rows"], key=lambda r: score(r, "accuracy"))
                    summary[fraction][f"{set_name}_{variant}_{k}"] = {
                        "epoch_best_loss": best_loss["epoch"],
                        "best_loss": score(best_loss, "loss"),
                        "epoch_best_accuracy": best_acc["epoch"],
                        "best_accuracy": score(best_acc, "accuracy"),
                    }
                    name = "fixe" if variant == "fixed" else "early stopping"
                    print(
                        f"  {set_label:<9} {k:>3} membre(s), {name:<15} : loss min à "
                        f"l'epoch {best_loss['epoch']:>2} ({score(best_loss, 'loss'):.4f}) | "
                        f"accuracy max à l'epoch {best_acc['epoch']:>2} "
                        f"({score(best_acc, 'accuracy') * 100:.2f}%)"
                    )

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "history": {str(k): v for k, v in history.items()},
                "summary": {str(k): v for k, v in summary.items()},
            },
            f,
            indent=2,
        )

    plot_epochs(history, args, output_dir / "epochs.png")

    print(f"\nRésultats et graphique enregistrés dans {output_dir}/")


# ============================================================
# Graphique
# ============================================================

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


def plot_epochs(history, args, path):
    sizes = args.sizes[-4:]  # 4 tailles au plus + la courbe early stopping
    colors = RAMP[len(RAMP) - len(sizes):]
    fractions = list(history)

    fig, axes = plt.subplots(
        2, len(fractions), figsize=(6.5 * len(fractions), 8.5), facecolor=SURFACE, squeeze=False
    )

    for col, fraction in enumerate(fractions):
        rows = history[fraction]["rows"]
        epochs = [r["epoch"] for r in rows]

        for row_i, (metric, scale, ylabel) in enumerate(
            (("loss", 1, "Loss par token"), ("accuracy", 100, "Accuracy (%)"))
        ):
            ax = axes[row_i][col]

            for k, color in zip(sizes, colors):
                ax.plot(
                    epochs, [r["selection"]["fixed"][k][metric] * scale for r in rows],
                    color=color, linewidth=2, marker="o", markersize=4,
                    label=f"{k} membre{'s' if k > 1 else ''}, epochs fixes",
                )

            big = sizes[-1]
            ax.plot(
                epochs, [r["selection"]["early_stopping"][big][metric] * scale for r in rows],
                color=colors[-1], linewidth=2, linestyle="--",
                label=f"{big} membres, early stopping OOB",
            )

            _style(ax, f"{fraction * 100:g} % neutralisé", "Epochs (budget par membre)", ylabel)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=TEXT,
               loc="lower center", ncol=len(handles))
    fig.suptitle(
        "LAMBADA : qualité de la forêt (moitié a du test) selon le nombre d'epochs par membre",
        color=TEXT, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
