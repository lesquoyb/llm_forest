"""
Membres de forêt sauvegardés sur disque, et scores mis en cache.

Deux étapes, séparées pour pouvoir composer n'importe quelle forêt (grande,
hétérogène, pondérée, avec température) sans rien réentraîner ni relancer
les modèles :

1. `train` : entraîne les membres un par un et enregistre leurs poids
   (float16) dans `<dir>/member_XXX.pt`, avec `<dir>/member_XXX.json`
   (masque, graine, temps...). Reprend là où il s'était arrêté. Les tirages
   (masques, bootstraps, graines) sont ceux de RandomLLMForest : avec la même
   graine, les membres sont les mêmes que ceux de forest_grid.py.

2. `score` : charge chaque membre à son tour sur le GPU et enregistre, pour
   chaque « vue » (le membre avec son propre masque, ou avec un masque
   d'inférence supplémentaire) et chaque corpus (test B, sélection) :
   - les TOPK tokens les plus probables à chaque position cible et leurs
     log-probabilités (pour l'argmax de la distribution moyenne) ;
   - la log-probabilité du token cible à plusieurs températures.
   C'est quelques Mo par membre ; compose.py en fait des forêts.

Usage :
    python lambada/members.py train --model roneneldan/TinyStories-28M --n 30 \
        --mask-fraction 0.125 --epochs 6 --dir lambada/outputs/openai/members/28M_f125_s0
    python lambada/members.py score --dir lambada/outputs/openai/members/28M_f125_s0
"""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import collate, load_lambada, prepare_corpus
from finetune import finetune, make_dim_mask, random_masked_dims, target_logits

TOPK = 64
TEMPERATURES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0]


def load_data(tokenizer, dev_size=500, seed=0, max_length=256):
    """Entraînement (moitié a), sélection (dans la moitié a), test (moitié b)."""

    train_raw, dev_raw, test_raw = load_lambada(seed=seed, fmt="openai", openai_dev_size=dev_size)
    return tuple(prepare_corpus(c, tokenizer, max_length) for c in (train_raw, dev_raw, test_raw))


def load_base(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # float32 explicite : les poids Pythia sont publiés en float16, chargés tels quels sinon.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


# ============================================================
# Entraînement
# ============================================================

def member_plan(n, data_size, hidden_size, mask_fraction, bootstrap, seed):
    """Masques et bootstraps de chaque membre, tirés comme dans RandomLLMForest.fit."""

    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    plan = []
    for i in range(n):
        masked_dims = random_masked_dims(hidden_size, mask_fraction, generator)
        indices = None
        if bootstrap:
            indices = [rng.randrange(data_size) for _ in range(data_size)]
            oob = sorted(set(range(data_size)) - set(indices))
            rng.shuffle(oob)  # garde la même suite de tirages que la forêt
        plan.append({"index": i, "masked_dims": list(masked_dims), "indices": indices, "seed": seed + i})
    return plan


def save_member(model, path):
    """Poids en float16, sans les doublons (embedding et lm_head sont liés)."""

    torch.save({name: p.detach().half().cpu() for name, p in model.named_parameters()}, path)


def load_member(base_model, path):
    state = torch.load(path, map_location="cpu")
    model = base_model  # chargé en place : un seul jeu de poids sur le GPU
    with torch.no_grad():
        params = dict(model.named_parameters())
        for name, value in state.items():
            params[name].copy_(value.to(params[name].dtype))
    return model


def train(args):
    """
    Entraîne les membres et enregistre leurs scores (sélection et test) à la
    fin de CHAQUE époque : member_XXX_e{époque}_scores_k0.pt. On choisit
    ensuite l'époque d'arrêt sur la sélection, sans réentraîner. Les poids ne
    sont enregistrés qu'avec --save-weights (une centaine de Mo par membre).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    base_model, tokenizer = load_base(args.model, device)
    train_data, dev, test = load_data(tokenizer, args.dev_size)
    hidden_size = base_model.config.hidden_size
    if args.dropout:
        # Dropout classique (couches nn.Dropout du modèle : embedding, résidus,
        # attention selon l'architecture), pour comparer au masque tiré à chaque pas.
        for module in base_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = args.dropout

    config = {k: v for k, v in vars(args).items() if k != "func"}
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    index_path = out.parent / f"index_{Path(args.model).name}.pt"
    if not index_path.exists():
        torch.save({"test": corpus_index(test, tokenizer), "sel": corpus_index(dev, tokenizer)}, index_path)

    plan = member_plan(args.n, len(train_data), hidden_size, args.mask_fraction,
                       not args.no_bootstrap, args.seed)
    for entry in plan:
        i = entry["index"]
        meta_path = out / f"member_{i:03d}.json"
        if meta_path.exists():
            continue
        own = make_dim_mask(tuple(entry["masked_dims"]), hidden_size, device)
        meta = {"model": args.model, "index": i, "seed": entry["seed"], "lr": args.lr,
                "masked_dims": entry["masked_dims"], "mask_fraction": args.mask_fraction,
                "mask_dropout": args.mask_dropout, "dropout": args.dropout,
                "bootstrap": not args.no_bootstrap, "epochs": args.epochs}

        def record(model, epoch):
            result = {"meta": meta | {"epoch": epoch}, "inference_fraction": 0.0}
            for name, corpus in (("test", test), ("sel", dev)):
                result[name] = score_views(model, tokenizer, corpus, [own], args.batch_size)
            torch.save(result, out / f"member_{i:03d}_e{epoch}_scores_k0.pt")

        if args.epochs == 0:  # modèle de base tel quel (référence sans fine-tuning)
            record(base_model, 0)
            meta_path.write_text(json.dumps(meta | {"train_time_s": 0.0}), encoding="utf-8")
            continue
        data = [train_data[j] for j in entry["indices"]] if entry["indices"] else train_data
        member = finetune(
            base_model, tokenizer, data, None,
            masked_dims=tuple(entry["masked_dims"]), mask_dropout=args.mask_dropout,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            seed=entry["seed"], copy_model=True, verbose=False, on_epoch_end=record,
        )
        if args.save_weights:
            save_member(member.model, out / f"member_{i:03d}.pt")
        meta_path.write_text(json.dumps(meta | {"train_time_s": member.history["compute_s"]}), encoding="utf-8")
        print(f"membre {i + 1}/{args.n} : {member.history['compute_s']:.0f} s", flush=True)
        del member
        torch.cuda.empty_cache()
    print(f"Scores enregistrés dans {out}", flush=True)


# ============================================================
# Scores
# ============================================================

@torch.no_grad()
def score_views(model, tokenizer, corpus, masks, batch_size=16):
    """
    Pour chaque masque de `masks` (None, vecteur, ou couple entrée/sortie) :
    {"topk_ids" [n_tokens, TOPK] int32, "topk_logp" [n_tokens, TOPK] float16,
     "target_logp" [n_tokens, len(TEMPERATURES)] float32}.
    """

    device = next(model.parameters()).device
    model.eval()
    loader = DataLoader(corpus, batch_size=batch_size, shuffle=False,
                        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id))
    temps = torch.tensor(TEMPERATURES, device=device)
    parts = [{"topk_ids": [], "topk_logp": [], "target_logp": []} for _ in masks]

    for batch in loader:
        labels = batch["labels"].to(device)
        targets = labels[labels != -100]
        for part, mask in zip(parts, masks):
            logits = target_logits(model, batch["input_ids"].to(device),
                                   batch["attention_mask"].to(device), labels, mask).float()
            logp = logits.log_softmax(-1)
            values, ids = logp.topk(TOPK, dim=-1)
            part["topk_ids"].append(ids.int().cpu())
            part["topk_logp"].append(values.half().cpu())
            scaled = (logits[None] / temps[:, None, None]).log_softmax(-1)  # [T, n, V]
            part["target_logp"].append(
                scaled.gather(2, targets.expand(len(temps), -1)[..., None]).squeeze(-1).T.cpu())

    return [{k: torch.cat(v) for k, v in part.items()} for part in parts]


def corpus_index(corpus, tokenizer, batch_size=16):
    """Tokens cibles et numéro du passage de chacun, dans l'ordre des scores."""

    loader = DataLoader(corpus, batch_size=batch_size, shuffle=False,
                        collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id))
    targets, words, offset = [], [], 0
    for batch in loader:
        labels = batch["labels"]
        is_target = labels != -100
        targets.append(labels[is_target])
        words.append(is_target.nonzero()[:, 0] + offset)
        offset += labels.shape[0]
    return {"targets": torch.cat(targets), "word_index": torch.cat(words), "n_words": offset}


def score(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.dir)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    base_model, tokenizer = load_base(config["model"], device)
    _, dev, test = load_data(tokenizer, config.get("dev_size", 500))
    hidden_size = base_model.config.hidden_size
    corpora = {"test": test, "sel": dev}

    index_path = out.parent / f"index_{Path(config['model']).name}.pt"
    if not index_path.exists():
        torch.save({name: corpus_index(c, tokenizer) for name, c in corpora.items()}, index_path)

    for meta_path in sorted(out.glob("member_*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        suffix = f"_scores_k{args.inference_masks}" + (
            f"_f{args.inference_fraction:g}" if args.inference_masks else "")
        scores_path = out / meta_path.name.replace(".json", f"{suffix}.pt")
        if scores_path.exists():
            continue
        model = load_member(base_model, out / meta_path.name.replace(".json", ".pt"))

        own = make_dim_mask(tuple(meta["masked_dims"]), hidden_size, device)
        views = [own]  # le membre tel qu'il a été entraîné (masque en entrée et en sortie)
        # Masques d'inférence : en entrée seulement (en sortie, ce n'est qu'une température).
        generator = torch.Generator().manual_seed(10_000 + meta["index"])
        for _ in range(args.inference_masks):
            extra = make_dim_mask(random_masked_dims(hidden_size, args.inference_fraction, generator),
                                  hidden_size, device)
            views.append((extra if own is None else own * extra, own))

        result = {"meta": meta, "inference_fraction": args.inference_fraction}
        for name, corpus in corpora.items():
            result[name] = score_views(model, tokenizer, corpus, views, args.batch_size)
        torch.save(result, scores_path)
        print(f"{meta_path.stem} : {len(views)} vues", flush=True)
    print(f"Scores enregistrés dans {out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("train")
    p.add_argument("--model", default="roneneldan/TinyStories-28M")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--mask-fraction", type=float, default=0.0)
    p.add_argument("--mask-dropout", type=float, default=0.0,
                   help="nouveau masque aléatoire de cette proportion à chaque pas")
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0, help="dropout classique pendant le fine-tuning")
    p.add_argument("--save-weights", action="store_true", help="enregistre aussi les poids (float16)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dev-size", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dir", required=True)
    p.set_defaults(func=train)

    p = sub.add_parser("score")
    p.add_argument("--dir", required=True)
    p.add_argument("--inference-masks", type=int, default=0,
                   help="vues supplémentaires par membre, chacune avec un masque d'entrée aléatoire")
    p.add_argument("--inference-fraction", type=float, default=0.125)
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(func=score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
