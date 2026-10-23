"""
Compose des forêts à partir des scores mis en cache par members.py.

Une forêt = une liste de vues (membre, masque d'inférence éventuel) avec des
poids ; on moyenne leurs distributions (vote souple), éventuellement après
une température appliquée à chaque vue. Rien n'est relancé sur les modèles.

Précision : un mot est juste si, pour chacun de ses tokens, l'argmax de la
distribution moyenne est le token cible. La distribution moyenne est
reconstruite à partir des TOPK tokens de chaque vue (exacte tant que le
gagnant figure dans le top-k d'au moins les vues qui comptent).
Perplexité : par mot, exacte (log-probabilités du token cible gardées).

Usage (courbes précision / perplexité en fonction de la taille) :
    python lambada/compose.py --dirs lambada/outputs/openai/members/28M_f0_s0 \
        lambada/outputs/openai/members/28M_f125_s0 --sizes 1 5 10 20 30 60
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch

from members import TEMPERATURES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_index(member_dir):
    config = json.loads((Path(member_dir) / "config.json").read_text(encoding="utf-8"))
    return torch.load(Path(member_dir).parent / f"index_{Path(config['model']).name}.pt")


def load_views(member_dir, k=0, views="own", limit=None, fraction=0.125, epoch=None):
    """
    Liste de vues, chacune {"test": {...}, "sel": {...}, "member": i}.
    views : "own" = chaque membre avec son masque d'entraînement ;
            "extra" = les k vues à masque d'inférence ; "all" = les deux.
    """

    out = []
    suffix = f"_scores_k{k}" + (f"_f{fraction:g}" if k else "")
    if epoch is not None:  # scores enregistrés à chaque époque (members.py train)
        paths = sorted(Path(member_dir).glob(f"member_*_e{epoch}{suffix}.pt"))[:limit]
    else:
        paths = sorted(p for p in Path(member_dir).glob(f"member_*{suffix}.pt") if "_e" not in p.stem)[:limit]
    for path in paths:
        data = torch.load(path)
        n_views = len(data["test"])
        chosen = {"own": [0], "extra": list(range(1, n_views)), "all": list(range(n_views))}[views]
        for v in chosen:
            out.append({"test": data["test"][v], "sel": data["sel"][v],
                        "member": data["meta"]["index"], "dir": str(member_dir)})
    return out


@torch.no_grad()
def ensemble(views, index, split="test", weights=None, t=1.0, chunk=1024):
    """Précision et perplexité (par mot) de la moyenne pondérée des vues."""

    info = index[split]
    targets = info["targets"].to(DEVICE)
    word_index = info["word_index"].to(DEVICE)
    n_words = info["n_words"]
    n_views = len(views)
    w = torch.full((n_views,), 1.0 / n_views, device=DEVICE) if weights is None else \
        torch.as_tensor(weights, dtype=torch.float, device=DEVICE) / sum(weights)
    t_idx = TEMPERATURES.index(t)

    # Perplexité : log sum_i w_i p_i(cible).
    target_logp = torch.stack([v[split]["target_logp"][:, t_idx] for v in views]).to(DEVICE)
    mixed = (w.log()[:, None] + target_logp).logsumexp(0)
    perplexity = math.exp(-mixed.double().sum().item() / n_words)

    # Précision : argmax de la distribution moyenne, reconstruite depuis les top-k.
    # La température change la forme des top-k : renormalisés à T sur le top-k.
    vocab = int(max(v[split]["topk_ids"].max() for v in views)) + 1
    vocab = max(vocab, int(targets.max()) + 1)
    correct_tok = torch.empty(len(targets), dtype=torch.bool, device=DEVICE)
    for start in range(0, len(targets), chunk):
        end = min(start + chunk, len(targets))
        dense = torch.zeros(end - start, vocab, device=DEVICE)
        for wi, v in zip(w, views):
            ids = v[split]["topk_ids"][start:end].to(DEVICE).long()
            logp = v[split]["topk_logp"][start:end].to(DEVICE).float()
            if t != 1.0:
                # p^(1/T) renormalisé : la masse hors top-k (petite) est ignorée.
                logp = logp / t
                logp = logp - logp.logsumexp(-1, keepdim=True)
            dense.scatter_add_(1, ids, wi * logp.exp())
        correct_tok[start:end] = dense.argmax(-1) == targets[start:end]

    wrong = torch.zeros(n_words, device=DEVICE)
    wrong.index_add_(0, word_index, (~correct_tok).float())
    return {"accuracy": (wrong == 0).float().mean().item(), "perplexity": perplexity}


def curve(views, index, sizes, n_subsets=20, seed=0, split="test", t=1.0, groups=None):
    """
    Précision et perplexité moyennes de sous-forêts tirées au hasard, par taille.
    groups : liste d'indices de vues par membre (les vues d'un même membre
    restent ensemble) ; par défaut une vue = un membre.
    """

    groups = groups or [[i] for i in range(len(views))]
    rng = random.Random(seed)
    rows = []
    for size in sizes:
        if size > len(groups):
            continue
        draws = [list(range(len(groups)))] if size == len(groups) else \
            [rng.sample(range(len(groups)), size) for _ in range(n_subsets)]
        scores = [ensemble([views[i] for g in d for i in groups[g]], index, split, t=t) for d in draws]
        acc = [s["accuracy"] for s in scores]
        logppl = [math.log(s["perplexity"]) for s in scores]
        mean = sum(acc) / len(acc)
        std = (sum((a - mean) ** 2 for a in acc) / (len(acc) - 1)) ** 0.5 if len(acc) > 1 else 0.0
        rows.append({"size": size, "accuracy": mean, "accuracy_std": std,
                     "perplexity": math.exp(sum(logppl) / len(logppl)), "n_subsets": len(draws)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dirs", nargs="+", required=True)
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--views", default="own", choices=["own", "extra", "all"])
    parser.add_argument("--fraction", type=float, default=0.125, help="masques d'inférence (avec --k)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    parser.add_argument("--subsets", type=int, default=20)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = {}
    for d in args.dirs:
        views = load_views(d, args.k, args.views, fraction=args.fraction)
        index = load_index(d)
        per_member = {}
        for i, v in enumerate(views):
            per_member.setdefault(v["member"], []).append(i)
        groups = list(per_member.values())
        results[d] = {split: curve(views, index, args.sizes, args.subsets, split=split, t=args.t, groups=groups)
                      for split in ("test", "sel")}
        for row in results[d]["test"]:
            print(f"{Path(d).name:24s} {row['size']:3d} membres | test {row['accuracy'] * 100:.2f} "
                  f"± {row['accuracy_std'] * 100:.2f} % | ppl {row['perplexity']:.0f}", flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
