"""
Forêts et génération d'histoires (protocole GPT-Eval de TinyStories, notation par Laya).

LAMBADA ne mesure que la prédiction d'un mot : ici on vérifie si la forêt
écrit de meilleures histoires. Données : TinyStories-valid.txt (21 990
histoires), mélangées une fois (graine 0) puis découpées en
- 10 000 histoires pour entraîner les membres (chacun sur un bootstrap) ;
- 1 000 histoires pour la perplexité ;
- 50 histoires dont le début (coupé au milieu d'une phrase) sert d'amorce.

Étapes (sous-commandes) :
1. train : membres fine-tunés une époque sur un bootstrap des 10 000 histoires
   (perte sur tous les tokens), poids sur disque comme members.py ;
2. generate : pour chaque configuration (modèle seul, forêt, masques
   d'inférence), 10 suites par amorce à la température 1. À chaque token, la
   forêt moyenne les distributions de ses membres (chacun garde son cache) ;
   perplexité par token sur les 1 000 histoires ;
3. judge : notes de Laya (grammaire, cohérence, créativité, intrigue ; 0 à 4)
   et mesures automatiques (longueur, répétitions, distinct-n).

Usage :
    python lambada/story_forest.py train --n 10 --mask-fraction 0.125 --dir .../story_28M_f125
    python lambada/story_forest.py generate --out .../stories.jsonl
    python lambada/story_forest.py judge --stories .../stories.jsonl
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from data import Example
from finetune import finetune, make_dim_mask, masked_hidden_states, random_masked_dims
from members import load_base, load_member, save_member

ROOT = Path(__file__).parent / "outputs" / "stories"
MAX_TOKENS = 256


def load_stories(seed=0):
    path = hf_hub_download("roneneldan/TinyStories", "TinyStories-valid.txt", repo_type="dataset")
    stories = [s.strip() for s in Path(path).read_text(encoding="utf-8").split("<|endoftext|>") if s.strip()]
    random.Random(seed).shuffle(stories)
    return {"train": stories[:10_000], "ppl": stories[10_000:11_000], "prompts": stories[-50:]}


def make_prompt(story):
    """Début d'histoire coupé au milieu d'une phrase, vers 40 % des mots."""

    words = story.split()
    cut = max(8, int(len(words) * 0.4))
    return " ".join(words[:cut])


def lm_examples(stories, tokenizer):
    """
    Histoires concaténées (séparées par <|endoftext|>) puis découpées en blocs
    de MAX_TOKENS + 1 tokens, en `Example` : perte sur tous les tokens du bloc
    sauf le premier. Des blocs de taille fixe gardent des tenseurs de même
    forme : avec des histoires de longueurs différentes, la mémoire du GPU se
    fragmentait jusqu'à déborder (3 h par membre au lieu d'une minute).
    """

    stream = []
    for ids in tokenizer(stories, add_special_tokens=False)["input_ids"]:
        stream += ids + [tokenizer.eos_token_id]
    size = MAX_TOKENS + 1
    return [Example(stream[i:i + 1], stream[i + 1:i + size], "")
            for i in range(0, len(stream) - size + 1, size)]


# ============================================================
# 1. Entraînement
# ============================================================

def train(args):
    device = torch.device("cuda")
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    base, tok = load_base(args.model, device)
    data = lm_examples(load_stories()["train"], tok)
    (out / "config.json").write_text(json.dumps(vars(args) | {"func": None}, indent=2), encoding="utf-8")
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    for i in range(args.n):
        dims = random_masked_dims(base.config.hidden_size, args.mask_fraction, gen)
        idx = [rng.randrange(len(data)) for _ in range(len(data))]
        path = out / f"member_{i:03d}.pt"
        if path.exists():
            continue
        m = finetune(base, tok, [data[j] for j in idx], None, masked_dims=dims, epochs=args.epochs,
                     lr=args.lr, batch_size=args.batch_size, seed=args.seed + i, verbose=False)
        save_member(m.model, path)
        (out / f"member_{i:03d}.json").write_text(json.dumps(
            {"index": i, "masked_dims": list(dims), "train_time_s": m.train_time_s}), encoding="utf-8")
        print(f"membre {i + 1}/{args.n} : {m.train_time_s:.0f} s", flush=True)
        del m
        torch.cuda.empty_cache()


# ============================================================
# 2. Génération par ensemble
# ============================================================

class Member:
    """Un modèle sur le GPU, son masque d'entrée/sortie et son cache de contexte."""

    def __init__(self, model, mask):
        self.model, self.mask, self.past = model, mask, None

    @torch.no_grad()
    def step(self, input_ids):
        """Log-probabilités du token suivant [batch, vocab] ; input_ids = nouveaux tokens."""

        in_mask, out_mask = self.mask if isinstance(self.mask, tuple) else (self.mask, self.mask)
        embeds = self.model.get_input_embeddings()(input_ids)
        if in_mask is not None:
            embeds = embeds * in_mask
        outputs = self.model.base_model(inputs_embeds=embeds, past_key_values=self.past, use_cache=True)
        self.past = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1]
        if out_mask is not None:
            hidden = hidden * out_mask
        return self.model.get_output_embeddings()(hidden).float().log_softmax(-1)


@torch.no_grad()
def generate(members, tokenizer, prompt, n_samples, max_new, temperature=1.0, seed=0):
    device = next(members[0].model.parameters()).device
    ids = torch.tensor([tokenizer(prompt, add_special_tokens=False)["input_ids"]] * n_samples, device=device)
    for m in members:
        m.past = None
    gen = torch.Generator(device=device).manual_seed(seed)
    w = math.log(len(members))
    new_tokens, done = [], torch.zeros(n_samples, dtype=torch.bool, device=device)
    step_in = ids
    for _ in range(max_new):
        logp = torch.stack([m.step(step_in) for m in members]).logsumexp(0) - w  # moyenne des probabilités
        probs = (logp / temperature).softmax(-1)
        nxt = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        nxt = torch.where(done, torch.full_like(nxt, tokenizer.eos_token_id), nxt)
        new_tokens.append(nxt)
        done |= nxt == tokenizer.eos_token_id
        if done.all():
            break
        step_in = nxt[:, None]
    out = torch.stack(new_tokens, 1).tolist()
    texts = []
    for row in out:
        if tokenizer.eos_token_id in row:
            row = row[:row.index(tokenizer.eos_token_id)]
        texts.append(tokenizer.decode(row))
    return texts


@torch.no_grad()
def perplexity(members, tokenizer, stories, batch=16):
    """Perplexité par token de la distribution moyenne sur des histoires jamais vues."""

    device = next(members[0].model.parameters()).device
    nll, count = 0.0, 0
    w = math.log(len(members))
    encoded = [ids[:MAX_TOKENS] for ids in tokenizer(stories, add_special_tokens=False)["input_ids"]]
    for start in range(0, len(encoded), batch):
        rows = encoded[start:start + batch]
        length = max(len(r) for r in rows)
        x = torch.full((len(rows), length), tokenizer.eos_token_id, device=device)
        att = torch.zeros_like(x)
        for i, r in enumerate(rows):
            x[i, :len(r)] = torch.tensor(r)
            att[i, :len(r)] = 1
        logps = []
        for m in members:
            h = masked_hidden_states(m.model, x, att, m.mask)
            logps.append(m.model.get_output_embeddings()(h[:, :-1]).float().log_softmax(-1)
                         .gather(2, x[:, 1:, None]).squeeze(-1))
        mixed = torch.stack(logps).logsumexp(0) - w
        valid = att[:, 1:].bool()
        nll -= mixed[valid].double().sum().item()
        count += valid.sum().item()
    return math.exp(nll / count)


def generate_all(args):
    device = torch.device("cuda")
    data = load_stories()
    prompts = [make_prompt(s) for s in data["prompts"]][:args.max_prompts]
    out_path = Path(args.out)
    done = set()
    if out_path.exists():
        done = {json.loads(l)["config"] for l in out_path.read_text(encoding="utf-8").splitlines()}
    summary_path = out_path.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    base, tok = load_base(args.model, device)
    hidden = base.config.hidden_size
    configs = json.loads(Path(args.configs).read_text(encoding="utf-8"))

    for cfg in configs:
        name = cfg["name"]
        if name in done:
            continue
        # Membres : poids sur disque (dossier + nombre) ou modèle de base ; masques d'inférence éventuels.
        models = []
        if cfg.get("dir"):
            for p in sorted(Path(cfg["dir"]).glob("member_*.pt"))[:cfg["n"]]:
                meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
                m, _ = load_base(cfg.get("model", args.model), device)
                load_member(m, p)
                models.append((m, make_dim_mask(tuple(meta["masked_dims"]), hidden, device)))
        else:
            m, _ = load_base(cfg.get("model", args.model), device)
            models.append((m, None))
        members = []
        g = torch.Generator().manual_seed(1234)
        for m, own in models:
            if cfg.get("inference_masks"):
                for _ in range(cfg["inference_masks"]):
                    extra = make_dim_mask(random_masked_dims(hidden, cfg["inference_fraction"], g), hidden, device)
                    members.append(Member(m, (extra if own is None else own * extra, own)))
            else:
                members.append(Member(m, own))

        start = time.perf_counter()
        ppl = perplexity(members, tok, data["ppl"][:args.ppl_stories])
        with out_path.open("a", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts):
                texts = generate(members, tok, prompt, args.samples, args.max_new, cfg.get("temperature", 1.0), seed=i)
                for j, t in enumerate(texts):
                    f.write(json.dumps({"config": name, "prompt_id": i, "sample": j,
                                        "prompt": prompt, "completion": t}, ensure_ascii=False) + "\n")
        summary[name] = {"perplexity": ppl, "n_members": len(models), "n_views": len(members),
                         "time_s": time.perf_counter() - start}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{name:28s} ppl {ppl:.3f} | {time.perf_counter() - start:.0f} s", flush=True)
        del members, models
        torch.cuda.empty_cache()


# ============================================================
# 3. Notation
# ============================================================

LEVELS = ["very poor", "poor", "acceptable", "good", "excellent"]
QUESTIONS = {
    "grammar": {"type": "score", "criteria": LEVELS, "instructions":
                "A student completed the beginning of a children's story. Rate the grammar and spelling of the completion."},
    "consistency": {"type": "score", "criteria": LEVELS, "instructions":
                    "A student completed the beginning of a children's story. Is the completion consistent with the "
                    "beginning (characters, events, the sentence that was cut off finished correctly)?"},
    "creativity": {"type": "score", "criteria": LEVELS, "instructions":
                   "A student completed the beginning of a children's story. How creative is the completion?"},
    "plot": {"type": "score", "criteria": LEVELS, "instructions":
             "A student completed the beginning of a children's story. Does the plot make sense and come to an ending?"},
}


def distinct_n(texts, n):
    grams = [tuple(t.split()[i:i + n]) for t in texts for i in range(max(0, len(t.split()) - n + 1))]
    return len(set(grams)) / max(1, len(grams))


def judge(args):
    from laya import Router

    router = Router()
    rows = [json.loads(l) for l in Path(args.stories).read_text(encoding="utf-8").splitlines()]
    scores_path = Path(args.stories).with_suffix(".laya.jsonl")
    seen = set()
    if scores_path.exists():
        seen = {(r["config"], r["prompt_id"], r["sample"]) for r in map(json.loads, scores_path.read_text(encoding="utf-8").splitlines())}
    with scores_path.open("a", encoding="utf-8") as f:
        for r in rows:
            key = (r["config"], r["prompt_id"], r["sample"])
            if key in seen:
                continue
            state = f"Beginning of the story: {r['prompt']}\n\nStudent's completion: {r['completion']}"
            out = router.predict(state, QUESTIONS, model="english")["answers"]
            words = r["completion"].split()
            rep = 1 - len(set(zip(words, words[1:], words[2:]))) / max(1, len(words) - 2)
            f.write(json.dumps({"config": r["config"], "prompt_id": r["prompt_id"], "sample": r["sample"],
                                **{k: out[k]["score"] for k in QUESTIONS}, "words": len(words),
                                "repeated_trigrams": rep}) + "\n")

    # Tableau par configuration.
    scored = [json.loads(l) for l in scores_path.read_text(encoding="utf-8").splitlines()]
    texts = {}
    for r in rows:
        texts.setdefault(r["config"], []).append(r["completion"])
    summary = json.loads(Path(args.stories).with_suffix(".summary.json").read_text(encoding="utf-8"))
    table = {}
    for cfg in dict.fromkeys(r["config"] for r in scored):
        sub = [r for r in scored if r["config"] == cfg]
        table[cfg] = {k: sum(r[k] for r in sub) / len(sub) for k in [*QUESTIONS, "words", "repeated_trigrams"]}
        table[cfg]["distinct_2"] = distinct_n(texts[cfg], 2)
        table[cfg]["perplexity"] = summary.get(cfg, {}).get("perplexity")
        print(cfg, {k: round(v, 3) if isinstance(v, float) else v for k, v in table[cfg].items()}, flush=True)
    Path(args.stories).with_suffix(".table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(required=True)
    p = sub.add_parser("train")
    p.add_argument("--model", default="roneneldan/TinyStories-28M")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--mask-fraction", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dir", required=True)
    p.set_defaults(func=train)
    p = sub.add_parser("generate")
    p.add_argument("--model", default="roneneldan/TinyStories-28M")
    p.add_argument("--configs", required=True, help="JSON : liste de {name, dir?, n?, model?, inference_masks?, ...}")
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--max-new", type=int, default=300)
    p.add_argument("--ppl-stories", type=int, default=1000)
    p.add_argument("--max-prompts", type=int, default=50)
    p.set_defaults(func=generate_all)
    p = sub.add_parser("judge")
    p.add_argument("--stories", required=True)
    p.set_defaults(func=judge)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
