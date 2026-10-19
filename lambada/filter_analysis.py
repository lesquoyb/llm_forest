"""
Quel « filtre » réalise un ensemble de masques à l'inférence ?

Un seul modèle, trois tests (sur le test B et la sélection de lambada_openai) :

1. Température : on divise les logits du modèle seul par T. Le T qui minimise
   la perplexité de la sélection est appliqué au test. Une température ne
   change pas l'argmax, donc pas la précision.
2. Version déterministe (règle d'inférence du dropout) : au lieu de tirer des
   masques, on multiplie les dimensions par la proportion gardée p = 1 - f,
   en entrée, en sortie, ou les deux. Une seule passe.
3. Décomposition : N masques aléatoires appliqués en entrée seule, en sortie
   seule, ou les deux.

Variantes de modèle comme mask_ensemble.py : base, finetuned, dropout.

Usage :
    python lambada/filter_analysis.py --model roneneldan/TinyStories-28M --epochs 6
"""

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import collate, load_lambada, prepare_corpus
from finetune import (MaskedLM, _correct_words, evaluate, finetune, make_dim_mask,
                      random_masked_dims, target_logits)


class FixedMaskLM(MaskedLM):
    """MaskedLM dont le masque est donné tel quel (couple entrée/sortie, valeurs réelles)."""

    def __init__(self, model, tokenizer, mask):
        super().__init__(model, tokenizer)
        self.mask = mask

    def dim_mask(self):
        return self.mask


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="roneneldan/TinyStories-28M")
    parser.add_argument("--variants", nargs="+", default=["finetuned", "dropout", "base"],
                        choices=["base", "finetuned", "dropout"])
    parser.add_argument("--fraction", type=float, default=0.125)
    parser.add_argument("--dropout-fraction", type=float, default=0.125)
    parser.add_argument("--n-masks", type=int, default=30)
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0, 3.5, 4.0])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dev-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


@torch.no_grad()
def temperature_scores(model, tokenizer, corpus, temperatures, batch_size, max_length):
    """Précision et perplexité par mot du modèle seul pour chaque température."""

    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(corpus, batch_size=batch_size, shuffle=False,
                        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id))
    temps = torch.tensor(temperatures, device=device)
    nll = torch.zeros(len(temperatures), dtype=torch.float64, device=device)
    correct, n_words_total = 0, 0

    for batch in loader:
        labels = batch["labels"].to(device)
        logits = target_logits(model, batch["input_ids"].to(device),
                               batch["attention_mask"].to(device), labels, None)
        is_target = labels != -100
        targets = labels[is_target]
        word_index = is_target.nonzero()[:, 0]
        correct += _correct_words(logits.argmax(-1)[None], targets, word_index, labels.shape[0]).item()
        for i, t in enumerate(temps):
            log_probs = (logits / t).log_softmax(-1)
            nll[i] -= log_probs.gather(1, targets[:, None]).sum().double()
        n_words_total += labels.shape[0]

    return {"accuracy": correct / n_words_total,
            "perplexity": [math.exp(v / n_words_total) for v in nll.tolist()]}


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    short = args.model.split("/")[-1].replace("TinyStories-", "")
    output_dir = Path(args.output_dir or Path(__file__).parent / "outputs" / "openai" / f"filter_analysis_{short}")
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

    common = dict(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                  max_length=args.max_length, seed=args.seed, verbose=False)
    keep = 1.0 - args.fraction
    results = {"config": vars(args), "variants": {}}

    for variant in args.variants:
        if variant == "base":
            model = base_model
        else:
            dropout = args.dropout_fraction if variant == "dropout" else 0.0
            print(f"\n=== Fine-tuning ({variant}) ===", flush=True)
            model = finetune(base_model, tokenizer, train, None, mask_dropout=dropout, **common).model
        out = results["variants"][variant] = {}

        # 1. Température.
        sel_t = temperature_scores(model, tokenizer, dev, args.temperatures, args.eval_batch_size, args.max_length)
        test_t = temperature_scores(model, tokenizer, test, args.temperatures, args.eval_batch_size, args.max_length)
        best = min(range(len(args.temperatures)), key=lambda i: sel_t["perplexity"][i])
        out["temperature"] = {
            "temperatures": args.temperatures,
            "sel_perplexity": sel_t["perplexity"], "test_perplexity": test_t["perplexity"],
            "best_T": args.temperatures[best],
            "test_accuracy": test_t["accuracy"],
            "test_perplexity_T1": test_t["perplexity"][args.temperatures.index(1.0)],
            "test_perplexity_best": test_t["perplexity"][best],
        }
        print(f"{variant:9s} température : T* = {args.temperatures[best]} | test {test_t['accuracy'] * 100:.2f} % | "
              f"ppl T=1 {out['temperature']['test_perplexity_T1']:.0f} -> T* {test_t['perplexity'][best]:.0f}",
              flush=True)

        # 2. Masque déterministe = espérance des masques (dimensions multipliées par p).
        scale = torch.full((hidden_size,), keep, device=device)
        for name, mask in (("entrée", (scale, None)), ("sortie", (None, scale)), ("les deux", (scale, scale))):
            for split, corpus in (("test", test), ("sel", dev)):
                r = evaluate([FixedMaskLM(model, tokenizer, mask)], corpus, batch_size=args.eval_batch_size)["soft"]
                out.setdefault("deterministic", {}).setdefault(name, {})[split] = r
            r = out["deterministic"][name]["test"]
            print(f"{variant:9s} échelle p={keep:.4f} ({name:8s}) | test {r['accuracy'] * 100:.2f} % | "
                  f"ppl {r['perplexity']:.0f}", flush=True)

        # 3. N masques aléatoires : entrée seule, sortie seule, les deux.
        for name in ("entrée", "sortie", "les deux"):
            generator = torch.Generator().manual_seed(args.seed + 1000)  # mêmes masques que mask_ensemble
            members = []
            for _ in range(args.n_masks):
                m = make_dim_mask(random_masked_dims(hidden_size, args.fraction, generator), hidden_size, device)
                members.append(FixedMaskLM(model, tokenizer,
                                           {"entrée": (m, None), "sortie": (None, m), "les deux": (m, m)}[name]))
            for split, corpus in (("test", test), ("sel", dev)):
                r = evaluate(members, corpus, batch_size=args.eval_batch_size)
                out.setdefault("random", {}).setdefault(name, {})[split] = {
                    "soft": r["soft"],
                    "member_mean_accuracy": sum(x["accuracy"] for x in r["members"]) / len(r["members"]),
                }
            r = out["random"][name]["test"]
            print(f"{variant:9s} {args.n_masks} masques ({name:8s}) | test {r['soft']['accuracy'] * 100:.2f} % | "
                  f"ppl {r['soft']['perplexity']:.0f} | membre moyen {r['member_mean_accuracy'] * 100:.2f} %",
                  flush=True)

        (output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        if variant != "base":
            del model
            torch.cuda.empty_cache()

    print(f"\nRésultats enregistrés dans {output_dir / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
