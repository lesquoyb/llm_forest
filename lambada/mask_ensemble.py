"""
Ensembles de masques à l'inférence : UN seul modèle, X masques aléatoires.

Chaque « membre » est le même modèle vu à travers son propre masque de
dimensions d'embedding (entrée + notation). Aucun entraînement par membre,
un seul jeu de poids en mémoire ; les X passes avant sont moyennées (vote
souple), comme pour la forêt. Sert à tester si le gain du masque dans la
forêt vient seulement de la diversité des votes.

Variantes (--variants) :
- base : modèle pré-entraîné, sans fine-tuning ;
- finetuned : fine-tuné une fois sur tout le corpus, sans masque ;
- dropout : fine-tuné une fois avec un nouveau masque aléatoire à chaque pas
  (--dropout-fraction), pour que le modèle supporte les masques d'inférence.

Pour chaque variante et chaque proportion de masque, on tire --n-masks
masques et on évalue des sous-ensembles de toutes tailles (comme forest_grid).

Usage :
    python lambada/mask_ensemble.py --model roneneldan/TinyStories-28M --epochs 6
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import load_lambada, prepare_corpus
from finetune import MaskedLM, evaluate, finetune, random_masked_dims
from forest_grid import draw_subsets, mean_std


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="roneneldan/TinyStories-28M")
    parser.add_argument("--variants", nargs="+", default=["finetuned", "dropout", "base"],
                        choices=["base", "finetuned", "dropout"])
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0625, 0.125, 0.1875, 0.25])
    parser.add_argument("--n-masks", type=int, default=30)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 5, 10, 15, 20, 30])
    parser.add_argument("--subsets-per-size", type=int, default=20)
    parser.add_argument("--dropout-fraction", type=float, default=0.125)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = args.model.split("/")[-1].replace("TinyStories-", "")
    output_dir = Path(args.output_dir or Path(__file__).parent / "outputs" / "openai" / f"mask_ensemble_{short}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    hidden_size = base_model.config.hidden_size

    train_raw, dev_raw, test_raw = load_lambada(seed=args.seed, fmt="openai", openai_dev_size=args.dev_size)
    train = prepare_corpus(train_raw, tokenizer, args.max_length)
    dev = prepare_corpus(dev_raw, tokenizer, args.max_length)
    test = prepare_corpus(test_raw, tokenizer, args.max_length)
    print(f"train {len(train)} | sélection {len(dev)} | test {len(test)} | hidden {hidden_size}", flush=True)

    results_path = output_dir / "results.json"
    results = {"config": vars(args), "rows": []}
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))
    done = {(r["variant"], r["fraction"], r["size"]) for r in results["rows"]}

    all_subsets = draw_subsets(args.n_masks, args.subsets_per_size, random.Random(args.seed))
    sizes = [k for k in args.sizes if k <= args.n_masks]
    subsets = [s for k in sizes for s in all_subsets[k]]

    common = dict(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                  max_length=args.max_length, seed=args.seed, verbose=False)

    for variant in args.variants:
        if all((variant, f, k) in done for f in [0.0] + args.fractions for k in sizes):
            print(f"{variant} : déjà fait", flush=True)
            continue

        if variant == "base":
            model, train_time = base_model, 0.0
        else:
            dropout = args.dropout_fraction if variant == "dropout" else 0.0
            print(f"\n=== Fine-tuning ({variant}, dropout de masque {dropout}) ===", flush=True)
            tuned = finetune(base_model, tokenizer, train, None, mask_dropout=dropout, **common)
            model, train_time = tuned.model, tuned.train_time_s
            print(f"  {train_time:.0f} s", flush=True)

        for fraction in [0.0] + args.fractions:
            if all((variant, fraction, k) in done for k in sizes):
                continue
            generator = torch.Generator().manual_seed(args.seed + 1000)
            n = 1 if fraction == 0 else args.n_masks
            members = [MaskedLM(model, tokenizer, random_masked_dims(hidden_size, fraction, generator))
                       for _ in range(n)]
            these = [[0]] if fraction == 0 else subsets
            test_scores = evaluate(members, test, batch_size=args.eval_batch_size, subsets=these)
            sel_scores = evaluate(members, dev, batch_size=args.eval_batch_size, subsets=these)

            it_test, it_sel = iter(test_scores["subsets"]), iter(sel_scores["subsets"])
            for k in ([1] if fraction == 0 else sizes):
                count = 1 if fraction == 0 else len(all_subsets[k])
                t = [next(it_test)["soft"] for _ in range(count)]
                s = [next(it_sel)["soft"] for _ in range(count)]
                row = {
                    "variant": variant, "fraction": fraction, "size": k,
                    "accuracy": mean_std([r["accuracy"] for r in t])[0],
                    "accuracy_std": mean_std([r["accuracy"] for r in t])[1],
                    "perplexity": math.exp(mean_std([math.log(r["perplexity"]) for r in t])[0]),
                    "sel_accuracy": mean_std([r["accuracy"] for r in s])[0],
                    "train_time_s": train_time,
                }
                results["rows"].append(row)
                print(f"{variant:9s} masque {fraction * 100:5.2f} % | {k:2d} masques | "
                      f"test {row['accuracy'] * 100:.2f} % | ppl {row['perplexity']:.0f} | "
                      f"sél {row['sel_accuracy'] * 100:.2f} %", flush=True)
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        if variant != "base":
            del model, tuned
            torch.cuda.empty_cache()

    print(f"\nRésultats enregistrés dans {results_path}", flush=True)


if __name__ == "__main__":
    main()
