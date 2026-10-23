"""
Analyse du protocole « même objectif » (outputs/openai/sweep, produit par members.py train).

Pour chaque taille de Pythia, chaque configuration et chaque taille de forêt k,
on choisit le taux d'apprentissage et l'époque qui maximisent la précision sur
la sélection (moyenne de sous-forêts de k membres), puis on lit le test à ce
réglage. Configurations :
- seeds : même modèle pré-entraîné, graines de fine-tuning seules (k = 1 : modèle seul) ;
- bag : + bootstrap ; bagdrop : + masque tiré à chaque pas ;
- pre : 10 modèles pré-entraînés avec des graines différentes, chacun fine-tuné ;
- seedsdrop / seedsdo : modèle seul avec masque tiré à chaque pas / dropout 0,1 (3 graines) ;
- zs : sans fine-tuning (10 graines de pré-entraînement).

Usage : python lambada/sweep_analysis.py   (écrit sweep/summary.json)
"""

import json
import random
import re
import statistics as st
from pathlib import Path

from compose import ensemble, load_index, load_views

SWEEP = Path(__file__).parent / "outputs" / "openai" / "sweep"
SIZES = ["14m", "31m", "70m", "160m"]
KS = [1, 2, 5, 10]


def forest_score(views, index, k, split, n_draws=10, seed=0):
    """Précision et perplexité moyennes de sous-forêts de k vues tirées au hasard."""

    rng = random.Random(seed)
    draws = [list(range(len(views)))] if k >= len(views) else [rng.sample(range(len(views)), k) for _ in range(n_draws)]
    scores = [ensemble([views[i] for i in d], index, split) for d in draws]
    return st.mean(s["accuracy"] for s in scores) * 100, st.mean(s["perplexity"] for s in scores)


def grid_dirs(size):
    """{config: {lr: [dossiers]}} ; « pre » regroupe ses 10 dossiers (un par graine de pré-entraînement)."""

    grid = {}
    for d in sorted(SWEEP.glob(f"{size}_lr*")):
        lr, config = re.match(rf"{size}_lr([^_]+)_(.+)$", d.name).groups()
        config = re.sub(r"_s\d+$", "", config)
        grid.setdefault(config, {}).setdefault(lr, []).append(d)
    return grid


def views_for(dirs, epoch):
    out = []
    for d in dirs:
        out += load_views(d, epoch=epoch)
    return out


def tune(dirs_by_lr, index, k, epochs):
    """Meilleur (lr, époque) sur la sélection pour une forêt de k membres ; renvoie le test à ce réglage."""

    best = None
    for lr, dirs in dirs_by_lr.items():
        for e in epochs:
            views = views_for(dirs, e)
            if not views or len(views) < k:
                continue
            sel = forest_score(views, index, k, "sel")[0]
            if best is None or sel > best[0]:
                best = (sel, lr, e, views)
    if best is None:
        return None
    sel, lr, e, views = best
    acc, ppl = forest_score(views, index, k, "test")
    return {"k": k, "lr": lr, "epoch": e, "sel": sel, "test": acc, "perplexity": ppl}


def main():
    summary = {}
    for size in SIZES:
        index = load_index(next(SWEEP.glob(f"{size}_lr*_seeds")))
        grid = grid_dirs(size)
        epochs = range(1, 5)
        res = summary[size] = {}
        for config, dirs_by_lr in grid.items():
            ks = [1] if config in ("seedsdrop", "seedsdo") else KS
            res[config] = [r for k in ks if (r := tune(dirs_by_lr, index, k, epochs))]
            for r in res[config]:
                print(f"{size:5s} {config:10s} k={r['k']:2d} lr {r['lr']:>5s} ép {r['epoch']} | "
                      f"sél {r['sel']:.2f} | test {r['test']:.2f} | ppl {r['perplexity']:.1f}", flush=True)
        # Sans fine-tuning : 10 graines de pré-entraînement.
        zs_dirs = sorted(SWEEP.glob(f"zs_{size}_s*"))
        zs = views_for(zs_dirs, 0)
        res["zs"] = [{"k": k, "test": forest_score(zs, index, k, "test")[0],
                      "perplexity": forest_score(zs, index, k, "test")[1]} for k in KS]
        for r in res["zs"]:
            print(f"{size:5s} zs         k={r['k']:2d} | test {r['test']:.2f} | ppl {r['perplexity']:.1f}", flush=True)

    # Modèles seuls plus gros.
    for size in ["410m", "1b", "1.4b"]:
        zs_dirs = sorted(SWEEP.glob(f"zs_{size}_s*"))
        index = load_index(zs_dirs[0])
        zs = views_for(zs_dirs, 0)
        entry = summary[size] = {"zs": [{"k": k, "test": forest_score(zs, index, k, "test")[0],
                                         "perplexity": forest_score(zs, index, k, "test")[1]}
                                        for k in KS if k <= len(zs)]}
        if size == "410m":
            entry["seeds"] = [tune({d.name.split("_lr")[1].split("_")[0]: [d]
                                    for d in SWEEP.glob("410m_lr*_seeds")}, index, 1, range(1, 3))]
        print(size, entry, flush=True)

    (SWEEP / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
