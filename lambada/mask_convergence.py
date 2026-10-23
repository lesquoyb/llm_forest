"""
Le masque tiré à chaque pas accélère-t-il le fine-tuning, ou le rend-il moins sensible au taux d'apprentissage ?

Pour un modèle seul (toutes les données d'entraînement), avec et sans masque
tiré à chaque pas, à plusieurs taux d'apprentissage et plusieurs graines, on
mesure la perte et la précision sur la sélection (500 passages) à intervalles
réguliers. On en tire, par configuration : le meilleur score, le pas où il
est atteint, le premier pas où l'on atteint 95 % de ce meilleur score, et la
perte de précision à la fin par rapport au meilleur point.

Usage :
    python lambada/mask_convergence.py --model roneneldan/TinyStories-28M --epochs 6 \
        --lrs 1e-5 2e-5 5e-5 1e-4 --seeds 0 1 2
"""

import argparse
import json
from pathlib import Path

import torch

from finetune import finetune
from members import load_base, load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="roneneldan/TinyStories-28M")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-5, 2e-5, 5e-5, 1e-4])
    parser.add_argument("--masks", type=float, nargs="+", default=[0.0, 0.125])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--evals-per-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    base, tok = load_base(args.model, device)
    train, dev, _ = load_data(tok)
    steps_per_epoch = -(-len(train) // args.batch_size)
    eval_every = max(1, steps_per_epoch // args.evals_per_epoch)

    out = Path(args.output)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"config": vars(args), "runs": []}
    done = {(r["lr"], r["mask"], r["seed"]) for r in results["runs"]}

    for lr in args.lrs:
        for mask in args.masks:
            for seed in args.seeds:
                if (lr, mask, seed) in done:
                    continue
                m = finetune(base, tok, train, dev, mask_dropout=mask, epochs=args.epochs, lr=lr,
                             batch_size=args.batch_size, seed=seed, eval_every=eval_every, verbose=False)
                h = m.history
                acc, steps = h["val_acc"], h["val_step"]
                best = max(range(len(acc)), key=acc.__getitem__)
                reach = next(i for i, a in enumerate(acc) if a >= 0.95 * acc[best])
                results["runs"].append({
                    "lr": lr, "mask": mask, "seed": seed, "steps": steps, "val_acc": acc,
                    "val_loss": h["val_loss"], "best_acc": acc[best], "best_step": steps[best],
                    "step_95": steps[reach], "final_drop": acc[best] - acc[-1],
                })
                out.write_text(json.dumps(results, indent=1), encoding="utf-8")
                print(f"lr {lr:g} masque {mask:g} graine {seed} | meilleur {acc[best] * 100:.2f} % au pas "
                      f"{steps[best]} | 95 % au pas {steps[reach]} | chute finale {(acc[best] - acc[-1]) * 100:.2f}",
                      flush=True)
                del m
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
